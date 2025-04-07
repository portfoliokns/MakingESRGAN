import torch
from model import Generator
from PIL import Image
import torchvision.transforms as transforms

# デノーマライズ用の関数
def denormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).cuda()
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).cuda()
    # mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).cpu()
    # std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).cpu()

    return tensor * std + mean

# 学習済みモデルの設定
# generator = Generator().cpu()
generator = Generator().cuda()
generator.load_state_dict(torch.load("generator/generator_batch_7_280.pth", weights_only=True))
generator.eval()  # 評価モードに設定

# 個別にアップスケールしたい画像を設定
num = 10
image = Image.open("w_test" + str(num) +".png")

original_width, original_height = image.size

if image.mode == "RGBA":
    image = image.convert("RGB")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
image = transform(image).unsqueeze(0).cuda()
# image = transform(image).unsqueeze(0).cpu()

with torch.no_grad():
    fake_hr = generator(image)

# 生成された画像を保存
fake_hr = fake_hr.squeeze(0)
fake_hr = denormalize(fake_hr)
fake_hr = torch.clamp(fake_hr, 0, 1)
fake_hr = transforms.ToPILImage()(fake_hr)
fake_hr.save("w_output" + str(num) +".jpg")