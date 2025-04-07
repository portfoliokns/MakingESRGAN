import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PairedTransform, SuperResolutionDataset
from model import Generator, Discriminator, VGGFeatureExtractor
import os
from torchvision.utils import save_image
import torch.nn as nn

# ハイパーパラメータ
epochs = 11  # 学習回数
batch_size = 10  # バッチサイズ（GPUのメモリに依存）
lr_g = 0.1e-4  # Generatorの学習率
lr_d = 0.1e-6  # Discriminatorの学習率
λ1, λ2, λ3 = 0.5, 0.1, 0.08  # L1損失
 
# データセットの作成
transform = PairedTransform()
dataset = SuperResolutionDataset(low_res_dir="data/train_low", high_res_dir="data/train_high", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルのインスタンス化
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# オプティマイザーの設定
optim_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))

# 損失関数
criterion = torch.nn.BCEWithLogitsLoss()
mse_loss = torch.nn.MSELoss() 
vgg_extractor = VGGFeatureExtractor().cuda()
l1_loss_fn = torch.nn.L1Loss()

# 途中から再開するための変数
start_epoch = 1  # デフォルトは1から開始

# もしチェックポイントファイルが存在すれば、読み込む
checkpoint_path = "checkpoint/checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # 次のエポックから開始
    print(f"✅ 学習を {start_epoch} エポック目から再開します。")
else:
    print("🚀 チェックポイントも学習済みモデルも見つかりませんでした。新規学習を開始します。")

# 学習ループ
for epoch in range(start_epoch, epochs + 1):
    for batch_idx, (lr, hr) in enumerate(dataloader):
        # データをGPUに送る
        lr, hr = lr.cuda(), hr.cuda()

        # -----------------
        # Discriminatorの訓練
        # -----------------
        optim_d.zero_grad()

        # 本物の画像と生成された画像
        real_output = discriminator(hr)
        fake_hr = generator(lr)
        fake_output = discriminator(fake_hr.detach())  # 学習をバックプロパゲーションしない

      # 学習に用いられた画像の保存
        save_image(lr, f"real_low_images/batch_{batch_idx}.png", normalize=True)
        save_image(fake_hr, f"fake_images/batch_{batch_idx}.png", normalize=True)
        save_image(hr, f"real_high_images/batch_{batch_idx}.png", normalize=True)
        
        # 本物と偽物の判定損失
        real_loss = criterion(real_output, torch.ones_like(real_output))
        fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optim_d.step()

        # -----------------
        # Generatorの訓練
        # -----------------
        optim_g.zero_grad()

        # Generatorの出力をDiscriminatorに通して、損失を計算
        fake_output = discriminator(fake_hr)
        perceptual_loss = nn.MSELoss()(vgg_extractor(fake_hr), vgg_extractor(hr))
        adversarial_loss = nn.BCELoss()(fake_output, torch.ones_like(fake_output))
        l1_loss = l1_loss_fn(fake_hr, hr)
        g_loss = l1_loss *λ1 + perceptual_loss * λ2 + adversarial_loss * λ3

        g_loss.backward()
        optim_g.step()

        # ログの表記
        if batch_idx % 2 == 0:
            print(f"{epoch},{batch_idx},{d_loss.item()},{l1_loss},{perceptual_loss},{adversarial_loss},{g_loss.item()},{lr_d},{lr_g}")

        # .pthの保存
        if batch_idx % 10 == 0:
            torch.save(generator.state_dict(), f"generator/generator_batch_{epoch}_{batch_idx}.pth")
    
    torch.save({
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
    }, f"checkpoint/checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
    }, "checkpoint/checkpoint.pth")
    print(f"💾 チェックポイントを保存しました（Epoch {epoch}）")
