import torch
import torch.nn as nn

class TaskMLP(nn.Module):
    def __init__(self, args, device, out_dim=10):
        super().__init__()
        self.args = args
        self.device = device
        self.out_dim = out_dim

        self.fusion_mode = args.get("sub_fusion_mode", "cat")
        if self.fusion_mode == "cat":
            self.sub_branch = SubCat(self.args, self.device, self.out_dim)
        elif self.fusion_mode == "add":
            self.sub_branch = SubAdd(self.args, self.device, self.out_dim)
        else:
            raise ValueError(f"Fusion mode {self.fusion_mode} not supported in clean version.")
        
    def forward(self, features, main_logits):
        main_logits = torch.nn.functional.softmax(main_logits, dim=1)
        sub_logits = self.sub_branch(features)["logits"]
        logits = sub_logits + main_logits # Loss
        return {
            'fmaps': sub_logits,
            'logits': logits
        }
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self        

class SubCat(nn.Module):
    def __init__(self, args, device, out_dim=10):
        super().__init__()
        self.args = args
        self.device = device
        self.sub_out_dim = out_dim
        self.sub_x_dim = args.get("commom_hid_dim", [64, 64, 64, 64])
        self.sub_hid_dim = args.get("sub_hid_dim", [64, 64, 64])
        self.sub_mlp_mode = args.get("sub_mlp_mode", "fc1")
        
        # block1: [B, 64, 16, 16] -> [B, 64, 8, 8]
        self.block1 = nn.Sequential(
            nn.Conv2d(self.sub_x_dim[-2], self.sub_hid_dim[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.sub_hid_dim[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # block2: [B, 128, 8, 8] -> [B, 64, 4, 4]
        self.block2 = nn.Sequential(
            nn.Conv2d(self.sub_hid_dim[0]+self.sub_x_dim[-1], self.sub_hid_dim[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.sub_hid_dim[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # block3: [B, 64, 4, 4] -> [B, 64, 2, 2]
        self.block3 = nn.Sequential(
            nn.Conv2d(self.sub_hid_dim[1], self.sub_hid_dim[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.sub_hid_dim[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # mlp_layer [B, 64, 2, 2] -> [B, 256] -> [B, sub_out_dim]
        self.mlp_layer = SubMLP(
            self.args,
            self.device,
            self.sub_out_dim
        )
    
    def forward(self, features):
        x = features[-2].to(self.device)
        x_1 = self.block1(x)
        x_1 = torch.cat([x_1, features[-1].to(self.device)], dim=1)
        x_2 = self.block2(x_1)
        x_3 = self.block3(x_2)

        logits = self.mlp_layer(x_3)

        return {
            "logits": logits
        }

class SubAdd(nn.Module):
    def __init__(self, args, device, out_dim=10):
        super().__init__()
        self.args = args
        self.device = device
        self.sub_out_dim = out_dim
        self.sub_x_dim = args.get("commom_hid_dim", [64, 64, 64, 64])
        self.sub_hid_dim = args.get("sub_hid_dim", [64, 64, 64])
        self.sub_mlp_mode = args.get("sub_sub_mlp_mode", "fc1")

        # block1: [B, 64, 16, 16] -> [B, 64, 8, 8]
        self.block1 = nn.Sequential(
            nn.Conv2d(self.sub_x_dim[-2], self.sub_hid_dim[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.sub_hid_dim[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # block2: [B, 64, 8, 8] -> [B, 64, 4, 4]
        self.block2 = nn.Sequential(
            nn.Conv2d(self.sub_x_dim[-1], self.sub_hid_dim[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.sub_hid_dim[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # block3: [B, 64, 4, 4] -> [B, 64, 2, 2]
        self.block3 = nn.Sequential(
            nn.Conv2d(self.sub_hid_dim[1], self.sub_hid_dim[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.sub_hid_dim[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # mlp_layer [B, 64, 2, 2] -> [B, 256] -> [B, sub_out_dim]
        self.mlp_layer = SubMLP(
            self.args,
            self.device,
            self.sub_out_dim
        )

    def forward(self, features):
        x = features[-2].to(self.device)
        x_1 = self.block1(x)
        x_1 = x_1 + features[-1].to(self.device)
        x_2 = self.block2(x_1)
        x_3 = self.block3(x_2)

        logits = self.mlp_layer(x_3)

        return {
            'logits': logits        
        }


class SubMLP(nn.Module):
    def __init__(self, args, device, out_dim=10):
        super().__init__()
        self.args = args
        self.device = device
        self.out_dim = out_dim
        self.mlp_mode = args.get("sub_mlp_mode", "fc1")
        self.in_dim = args.get("sub_feature_dim", 256)

        if self.mlp_mode == 'fc1':
            self.mlp_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.in_dim, self.out_dim),
            )
        elif self.mlp_mode == 'kan':
            self.mlp_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.in_dim, self.out_dim),
            )
        else:
            raise NotImplementedError(f"Mode {self.mlp_mode} not implemented in clean version.")
        
    def forward(self, feature):
        logits = self.mlp_layer(feature)
        return logits
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval
        return self
    
