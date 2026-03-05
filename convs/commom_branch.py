import torch.nn as nn

class ISARConv4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_dim = args.get("x_dim", 3)
        self.commom_hid_dim = args.get("commom_hid_dim", [64, 64, 64, 64])
        self.commom_out_dim = args.get("commom_out_dim", 128)

        # block1: [B, 3, 128, 128] -> [B, 64(self.commom_hid_dim), 64, 64]
        self.block1 = nn.Sequential(
            nn.Conv2d(self.x_dim, self.commom_hid_dim[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.commom_hid_dim[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # block2: [B, 64, 64, 64] -> [B, 64, 32, 32]
        self.block2 = nn.Sequential(
            nn.Conv2d(self.commom_hid_dim[0], self.commom_hid_dim[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.commom_hid_dim[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # block3: [B, 64, 32, 32] -> [B, 64, 16, 16]
        self.block3 = nn.Sequential(
            nn.Conv2d(self.commom_hid_dim[1], self.commom_hid_dim[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.commom_hid_dim[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # block4: [B, 64, 16, 16] -> [B, 64, 8, 8]
        self.block4 = nn.Sequential(
            nn.Conv2d(self.commom_hid_dim[2], self.commom_hid_dim[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.commom_hid_dim[3]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # flatten: [B, 64, 8, 8] -> [B, 4096] -> [B, 128]
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*self.commom_hid_dim[3], self.commom_out_dim),
            nn.ReLU(),
        )
        self.out_dim_for_no_buffer_fc = 8*8*self.commom_hid_dim[3]
        self.out_dim_for_buffer = self.commom_out_dim

    def forward(self, x):
        x_1 = self.block1(x)
        x_2 = self.block2(x_1)
        x_3 = self.block3(x_2)
        x_4 = self.block4(x_3)

        features = self.flatten(x_4)
        # 返回字典：
        # fmaps: 中间层特征图 (给 sub_branch 用于融合特征)
        # features: 最终特征向量 (给 linears 层最终分类出logits用)
        # 训练逻辑: 先用临时头训练好主的卷积网络权重参数 再换成解析头
        return {
            'fmaps': [x_1, x_2, x_3, x_4],
            'features': features
        }
