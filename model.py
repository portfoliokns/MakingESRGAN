import torch.nn as nn

class RRDB(nn.Module):
    def __init__(self, nf=64):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.conv3(x2)
        return x + x3

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.ReLU()

        self.residualLayer = nn.Sequential(
            *[RRDB() for _ in range(16)]
        )

        self.residualConv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.residualBN = nn.BatchNorm2d(64)

        self.pixelShuffle = nn.Sequential(
            nn.Conv2d(64, 64*4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        skip = self.relu(x)

        x = self.residualLayer(skip)
        x = self.residualConv(x)
        x = self.residualBN(x)
        x = x + skip
        x = self.pixelShuffle(x)
        return x



class Discriminator(nn.Module):
    def __init__(self, size=256):
        super(Discriminator, self).__init__()

        size = size // 16  # 4回のダウンサンプリングで16分の1になる

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 1024chへ
            nn.LeakyReLU(0.2),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * size * size, 512)  # 入力ユニット数を512に変更
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(512, 1)  # 最終出力
        self.sigmoid = nn.Sigmoid()  # Sigmoidでスコア化

    def forward(self, x):
        x = self.net(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)  # FC(1024)の後にLeakyReLU
        x = self.fc2(x)
        x = self.sigmoid(x)  # Sigmoidでスコア化
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


import torchvision.models as models
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        # VGG19を特徴抽出用に設定
        self.vgg = models.vgg19(pretrained=True).features
        self.vgg.eval()  # 評価モードに設定
    
    def forward(self, x):
        return self.vgg(x)