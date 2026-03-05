# -*- coding: utf-8 -*-
"""
Proper implementation of the ACIL [1].

This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
"""

import torch
import logging
import numpy as np
from tqdm import tqdm
from os import path, makedirs
from models.base import BaseLearner
from typing import Dict, Any, Optional, Sized
from torch.utils.data import DataLoader, Sampler
from utils.data_manager import DataManager, DummyDataset
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.inc_net import Task_Spec_Resnet
from utils.toolkit import count_parameters, tensor2numpy
import os
from collections import defaultdict
from utils.heatmap import save_feature_heatmap_on_image

__all__ = [
    "ACIL_Spec_Branch_Resnet",
]


class _Extract(torch.nn.Module):
    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name

    def forward(self, X: Dict[str, Any]) -> torch.Tensor:
        return X[self.name]

class _Extract_last(torch.nn.Module):
    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name

    def forward(self, X: Dict[str, Any]) -> torch.Tensor:
        fmaps = X[self.name]
        return fmaps[-1]

class InplaceRepeatSampler(Sampler):
    def __init__(self, data_source: Sized, num_repeats: int = 1):
        self.data_source = data_source
        self.num_repeats = num_repeats

    def __iter__(self):
        for i in range(len(self.data_source)):
            for _ in range(self.num_repeats):
                yield i

    def __len__(self):
        return len(self.data_source) * self.num_repeats


class ACIL_Spec_Branch_Resnet(BaseLearner):
    """
    Training process of the ACIL [1].

    This implementation refers to the official implementation https://github.com/ZHUANGHP/Analytic-continual-learning.

    References:
    [1] Zhuang, Huiping, et al.
        "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
        Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
    """
    def __init__(self, args: Dict[str, Any]) -> None:
        args.update(args["configurations"][args["dataset_name"]])

        if "memory_size" not in args:
            args["memory_size"] = 0
        elif args["memory_size"] != 0:
            raise ValueError(
                f"{self.__class__.__name__} is an exemplar-free method,"
                "so the `memory_size` must be 0."
            )
        super().__init__(args)
        self.parse_args(args)

        makedirs(self.save_path, exist_ok=True)

        """ Create the network """
        self.create_network()

    def parse_args(self, args: Dict[str, Any]) -> None:
        """ Base training hyper-parameters
        For small datasets like CIFAR-100 without powerful image augmentation provided here,
        we suggest using the MultiStepLR scheduler to get a more generalizable model.
        """
        self.num_workers: int = args.get("num_workers", 8)

        self.train_eval_freq: int = args.get("train_eval_freq", 1)
        # Batch size
        self.init_batch_size: int = args.get("init_batch_size", 256)
        # Learning rate information
        self.lr_info: Dict[str, Any] = args["scheduler"]
        # 5e-4 for CIFAR and 5e-5 for ImageNet
        self.weight_decay: float = args.get("init_weight_decay", 1e-4)

        """ Incremental learning hyper-parameters"""
        # Bigger batch size leads faster learning speed, >= 4096 for ImageNet.
        self.IL_batch_size: int = args.get("IL_batch_size", self.init_batch_size)
        # 8192 for CIFAR-100, and 16384 for ImageNet
        self.buffer_size: int = args["buffer_size"]
        # Regularization term of the regression
        self.gamma: float = args["gamma"]
        # Inplace repeat sampler for the training data loader during incremental learning
        self.inplace_repeat: int = args.get("inplace_repeat", 1)

        # Set the log path to save the base training backbone
        self.seed: int = args["seed"]
        self.conv_type: str = args["convnet_type"]
        self.save_path = (
            f"logs/{args['model_name']}/{args['dataset_name']}/{args['init_cls']}"
        )
        self.init_cls = args["init_cls"]
        self.increment = args["increment"]
        
        
        self.logits_thr = args["logits_thr"]
        
        self.unsupervision_pretrain = args["unsupervision_pretrain"]
        self.draw_heatmap = args["draw_heatmap"]

    def create_network(self) -> None:
        self._network = Task_Spec_Resnet(
            self.args,
            buffer_size=self.buffer_size,
            pretrained=self.unsupervision_pretrain,
            gamma=self.gamma,
            device=self._device,
        )

    def incremental_train(self, data_manager: DataManager) -> None:
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        print(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        if self._cur_task == 0:
            self.task_class_range=[self._known_classes, self._total_classes]
        else :
            self.task_class_range.append(self._total_classes)

        test_dataset: DummyDataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
            ret_data=False,
        )  # type: ignore

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.init_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers == 0),
        )

        self.classes = data_manager.classes
        self.class_to_idx = data_manager.class_to_idx
        self.order = data_manager._class_order
        self.new_classes = self.order[self._known_classes:self._total_classes]

        if self._cur_task == 0:
            assert self._known_classes == 0
            # You can specify the base weight in configuration file to skip the base training process.
            # This is helpful when you want to compare the performance of different methods fairly.
            if self.args.get("base_weight", None) is not None:
                base_weight_path = self.args["base_weight"]
                assert path.isfile(base_weight_path), "The base weight is not found."
                logging.info(
                    f"Loading the base model from the provided weight: {base_weight_path}. "
                    f"The base training process is skipped."
                )
                self._network.convnet.load_state_dict(torch.load(base_weight_path))
                self._network.to(self._device)
            else:
                train_dataset_init: DummyDataset = data_manager.get_dataset(
                    np.arange(0, self._total_classes),
                    source="train",
                    mode="train",
                    ret_data=False,
                )  # type: ignore
                train_loader_init = DataLoader(
                    train_dataset_init,
                    batch_size=self.init_batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                )
                self._init_train(train_loader_init, self.test_loader)
            self._network.generate_buffer()
            self._network.generate_fc()
        self._network.to(self._device)

        train_dataset: DummyDataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
            ret_data=False,
        )  # type: ignore

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.IL_batch_size,
            num_workers=self.num_workers,
            sampler=InplaceRepeatSampler(train_dataset, self.inplace_repeat),
        )

        self._train(
            train_loader,
            desc="Base Re-align" if self._cur_task == 0 else "Incremental Learning",
        )

        if self._cur_task == 0:
            task_class = self.init_cls
        elif self._cur_task > 0:
            task_class = self.increment
        train_dataset_sub: DummyDataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            ret_data=False,
            # appendent=self._get_memory(),
        )  # type: ignore
        train_loader_sub = DataLoader(
            train_dataset_sub,
            batch_size=self.init_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        test_dataset_sub: DummyDataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="test",
            mode="test",
            ret_data=False,
        )  # type: ignore

        test_loader_sub = DataLoader(
            test_dataset_sub,
            batch_size=self.init_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers == 0),
        )

        self._network.update_adapter(task_class)
        
        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )
        print("All params: {}".format(count_parameters(self._network)))
        print("Trainable params: {}".format(count_parameters(self._network, True)))
        
        taskMLP = self._network.adapter_list[-1]
        optimizer = torch.optim.SGD(
            taskMLP.parameters(),
            lr=self.lr_info["init_lr"],
            momentum=0.9,
            weight_decay=self.weight_decay,                
        )
        init_epochs = self.lr_info["init_epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=init_epochs
        )
        total_batches = init_epochs * (len(train_loader_sub))
        criterion = torch.nn.CrossEntropyLoss().to(self._device)
        process_bar = tqdm(total=total_batches, desc="Sub Branch Training", unit="batches")
        for epoch in range(init_epochs):
            taskMLP.train()
            losses = 0.0
            correct, total = 0, 0
            for _, X, y in train_loader_sub:
                X: torch.Tensor = X.to(self._device, non_blocking=True)
                y: torch.Tensor = y.to(self._device, non_blocking=True).long()

                l, r = self.task_class_range[self._cur_task], self.task_class_range[self._cur_task+1]

                out = self._network(X)                
                features = out["features"]
                main_logits = out["logits"].to(self._device)

                optimizer.zero_grad(set_to_none=True)

                y_hat = taskMLP(features, main_logits[:, l:r])['logits']
                
                loss: torch.Tensor = criterion(y_hat, y-self._known_classes)
                loss.backward()
                optimizer.step()
                process_bar.update(1)

                losses += loss.item()
                _, preds = torch.max(y_hat, dim=1)
                preds = preds+self._known_classes
                correct += preds.eq(y.expand_as(preds)).cpu().sum()
                total += len(y)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            taskMLP.eval()
            if (epoch + 1) % self.train_eval_freq == 0:                   
                with logging_redirect_tqdm():
                    logging.info(
                        f"Epoch {epoch + 1}/{init_epochs} - "
                        f"Train Loss: {(losses / len(train_loader_sub)):.4f}, "
                        f"Train Acc@1: {train_acc:.3f}%, "
                    )
                            
            test_losses = 0.0
            test_correct, test_total = 0, 0
            for _, X, y in test_loader_sub:
                X: torch.Tensor = X.to(self._device, non_blocking=True)
                y: torch.Tensor = y.to(self._device, non_blocking=True).long()

                out = self._network(X)
                features = out["features"]
                main_logits = out["logits"].to(self._device)
                main_logits = torch.nn.functional.softmax(main_logits, dim=1)

                l, r = self.task_class_range[self._cur_task], self.task_class_range[self._cur_task+1]
                
                y_hat = taskMLP(features, main_logits[:, l:r])['logits']
                test_loss: torch.Tensor = criterion(y_hat, y-self._known_classes)

                test_losses += test_loss.item()
                _, preds = torch.max(y_hat, dim=1)
                preds = preds+self._known_classes
                test_correct += preds.eq(y.expand_as(preds)).cpu().sum()
                test_total += len(y)
            test_acc = np.around(tensor2numpy(test_correct) * 100 / test_total, decimals=2)
            with logging_redirect_tqdm():
                logging.info(
                    f"Epoch {epoch + 1}/{init_epochs} - "
                    f"Test Loss: {(test_losses / len(test_loader_sub)):.4f}, "
                    f"Test Acc@1: {test_acc:.3f}%, "
                )
            scheduler.step()
        taskMLP.freeze()



    @torch.no_grad()
    def _train(
        self, train_loader: DataLoader, desc: str = "Incremental Learning"
    ) -> None:
        self._network.eval()
        self._network.update_fc(self._total_classes)
        for _, X, y in tqdm(train_loader, desc=desc):
            X: torch.Tensor = X.to(self._device, non_blocking=True)
            y: torch.Tensor = y.to(self._device, non_blocking=True).long()
            self._network.fit(X, y)
        self._network.after_task()
        

    def _init_train(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        if self.args['sub_fusion_mode'] in ['cat', 'add', 'fc', 'arc_cat']:            
            mlp_layer = torch.nn.Sequential(
                _Extract("features"),
                torch.nn.Linear(self._network.feature_dim, self._total_classes, bias=False),
            )
        elif self.args['sub_fusion_mode'] in ['cat_kan', 'add_kan', 'kan']:
            hid_dim = self.args["sub_mlp_hid_dim"] if (self.args["sub_mlp_hid_dim"] is not None) else self.rgs["sub_hid_dim"]
            if hid_dim == 0:
                mlp_layer = torch.nn.Sequential(
                    _Extract("features"),
                    torch.nn.Flatten(), 
                    # KAN([self._network.feature_dim, self._total_classes]),
                )
            else:
               mlp_layer =  torch.nn.Sequential(
                   _Extract_last("fmaps"),
                    torch.nn.Flatten(),
                    # KAN([self._network.feature_dim, self.args['sub_mlp_hid_dim'], self._total_classes])
               )
        model = torch.nn.Sequential(
            self._network.convnet,
            mlp_layer,
        ).to(self._device)
        if len(self._multiple_gpus) > 1:
            model = torch.nn.DataParallel(model, self._multiple_gpus)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr_info["init_lr"],
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        criterion = torch.nn.CrossEntropyLoss().to(self._device)

        # Scheduler with linear warmup
        scheduler_type = self.lr_info["type"]
        if scheduler_type == "MultiStep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_info["milestones"],
                gamma=self.lr_info["decay"],
            )
        elif scheduler_type == "CosineAnnealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.lr_info["init_epochs"], eta_min=1e-6
            )
        else:
            raise ValueError(f"Unsupported LR scheduler type: {scheduler_type}")

        if self.lr_info.get("warmup", 0) > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=self.lr_info["warmup"],
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, scheduler], [self.lr_info["warmup"]]
            )

        init_epochs = self.lr_info["init_epochs"]

        total_batches = init_epochs * (len(train_loader) + len(test_loader)) + (
            init_epochs // self.train_eval_freq
        ) * len(train_loader)
        process_bar = tqdm(total=total_batches, desc="Base Training", unit="batches")

        for epoch in range(init_epochs):
            model.train()
            for _, X, y in train_loader:
                X: torch.Tensor = X.to(self._device, non_blocking=True)
                y: torch.Tensor = y.to(self._device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)

                y_hat = model(X)
                loss: torch.Tensor = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                process_bar.update(1)

            model.eval()
            if (epoch + 1) % self.train_eval_freq == 0:
                train_metrics = _evaluate(
                    model, train_loader, process_bar, self._device
                )
                with logging_redirect_tqdm():
                    logging.info(
                        f"Epoch {epoch + 1}/{init_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc@1: {train_metrics['acc@1'] * 100:.3f}%, "
                        # f"Train Acc@5: {train_metrics['acc@5'] * 100:.3f}%, "
                        f"LR: {scheduler.get_last_lr()[0]}"
                    )
            
            test_metrics = _evaluate(model, test_loader, process_bar, self._device)
            with logging_redirect_tqdm():
                logging.info(
                    f"Epoch {epoch + 1}/{init_epochs} - "
                    f"Test Loss: {test_metrics['loss']:.4f}, "
                    f"Test Acc@1: {test_metrics['acc@1'] * 100:.3f}%, "
                )

            scheduler.step()
        self._network.eval()
        saving_file = path.join(
            self.save_path,
            f"{self.conv_type}_{self.seed}_{round(test_metrics['acc@1'] * 10000)}.pth",
        )
        torch.save(self._network.convnet.state_dict(), saving_file)
        self._network.freeze()

    def after_task(self) -> None:
        self._known_classes = self._total_classes
        self._network.after_task()


    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        spec_cnt = 0
        total_cnt = 0
        for _, (_, inputs, targets) in enumerate(tqdm(loader, desc="Eval cnn", unit="batches")):
            inputs = inputs.to(self._device)
            total_cnt += targets.shape[0]            
            with torch.no_grad():
                out = self._network(inputs)
                main_outputs = out["logits"]
                features = out['features']
                outputs= torch.nn.functional.softmax(main_outputs, dim=1)
                
                adapt_outputs = []
                for idx, line in enumerate(outputs):
                    spec_flag = False
                    for i in range(self._cur_task+1):
                        l, r = self.task_class_range[i], self.task_class_range[i+1]
                        taskMLP = self._network.adapter_list[i]
                        spec_branch = taskMLP([f[idx].unsqueeze(0) for f in features], line[l:r].unsqueeze(0))
                        spec_out = spec_branch['fmaps']
                        spec_out= torch.nn.functional.softmax(spec_out, dim=1)
                        
                        if self.logits_thr == -1:
                            thr = (1/self.init_cls) if i == 0 else (1/self.increment)
                            spec_pred = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[0]
                            if spec_pred[0]>thr:
                                line[l:r] = 0.5*line[l:r] + 0.5*spec_out
                                spec_flag = True
                        elif self.logits_thr == 0: # 强制按均等融合
                            line[l:r] = 0.5*line[l:r] + 0.5*spec_out
                            spec_flag = True
                        elif self.logits_thr > 0: # 指定阈值
                            thr = self.logits_thr
                            spec_pred = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[0]
                            if spec_pred[0]>thr:
                                line[l:r] = 0.5*line[l:r] + 0.5*spec_out
                                spec_flag = True

                    adapt_outputs.append(line)
                    if spec_flag:
                        spec_cnt+=1

                outputs = torch.stack(adapt_outputs)

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        print(f'spec_branch: {spec_cnt} / {total_cnt} = {((spec_cnt/total_cnt)*100):.2f} %')
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        if self.draw_heatmap and self._total_classes==34:
            heatmap_path = "./logs/acil_spec/isar/6/4/heatmap"
            if not os.path.exists(heatmap_path):
                os.mkdir(heatmap_path)
            layer3_path = os.path.join(heatmap_path, 'layer3')
            if not os.path.exists(layer3_path):
                os.mkdir(layer3_path)
            layer4_path = os.path.join(heatmap_path, 'layer4')
            if not os.path.exists(layer4_path):
                os.mkdir(layer4_path)
            
            n = 5 
            counter = defaultdict(int)
            for batch, (_, inputs, targets) in enumerate(tqdm(self.test_loader, desc="draw heatmap", unit="batches")):
                inputs = inputs.to(self._device)           
                with torch.no_grad():
                    outs = self._network(inputs)
                    features = outs['features']
                    for i,img  in (enumerate(inputs)):
                        target = targets[i].item()  # 取出单个 target（假设是tensor）
                        if counter[target] < n:                            
                            counter[target] += 1
                            save_feature_heatmap_on_image(
                                original_img=img,
                                feature_map=features[2][i],
                                save_path=os.path.join(layer3_path, f"layer3_{batch:03d}_{i:04d}_{targets[i]}.png"),
                            )
                            save_feature_heatmap_on_image(
                                original_img=img,
                                feature_map=features[3][i],
                                save_path=os.path.join(layer4_path, f"layer4_{batch:03d}_{i:04d}_{targets[i]}.png"),
                            )

        return cnn_accy, nme_accy


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    progress_bar: Optional[tqdm] = None,
    device=None,
) -> Dict[str, float]:
    """Evaluate the model on the given data loader."""
    model.eval()
    acc1_cnt, acc5_cnt = 0, 0
    sample_cnt = 0
    total_loss = 0.0

    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    for _, X, y in loader:
        X: torch.Tensor = X.to(device, non_blocking=True)
        y: torch.Tensor = y.to(device, non_blocking=True).long()

        logits: torch.Tensor = model(X)
        acc1_cnt += (logits.argmax(dim=1) == y).sum().item()
        sample_cnt += y.size(0)
        total_loss += float(criterion(logits, y).item())

        if progress_bar is not None:
            progress_bar.update(1)

    return {
        "acc@1": acc1_cnt / sample_cnt,
        # "acc@5": acc5_cnt / sample_cnt,
        "loss": total_loss / sample_cnt,
    }
