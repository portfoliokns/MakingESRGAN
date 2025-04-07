import os
from glob import glob
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as F

class SuperResolutionDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        """
        低画質（LR）と高画質（HR）の画像をペアで読み込むデータセット
        """
        self.low_res_images = sorted(glob(os.path.join(low_res_dir, "*.*")))
        self.high_res_images = sorted(glob(os.path.join(high_res_dir, "*.*")))

        self.transform_lr = transforms.Compose([
            transforms.Resize((128, 128)),
        ])

        self.transform_hr = transforms.Compose([
            transforms.Resize((256, 256)),
        ])

        self.transform = transform

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        """ LR（低画質）と HR（高画質）のペアを取得 """
        lr_image = Image.open(self.low_res_images[idx]).convert("RGB")
        hr_image = Image.open(self.high_res_images[idx]).convert("RGB")

        lr_image, hr_image = self.transform(lr_image, hr_image)

        lr_image = self.transform_lr(lr_image)
        hr_image = self.transform_hr(hr_image)

        return lr_image, hr_image
    
class PairedTransform:
    def __init__(self, crop_size=256):
        self.crop_size = crop_size
        self.base_transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomChoice([
                transforms.Lambda(lambda img: img.rotate(0)),
                transforms.Lambda(lambda img: img.rotate(90, expand=True)),
                transforms.Lambda(lambda img: img.rotate(180)),
                transforms.Lambda(lambda img: img.rotate(270, expand=True)),
            ]),
        ])
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __call__(self, lr_image, hr_image):

        seed = random.randint(0, 99999)  # 乱数のシードを固定

        # # ランダムクロップのパラメータを取得
        i, j, h, w = transforms.RandomCrop.get_params(lr_image, output_size=(self.crop_size, self.crop_size))

        # クロップを同じ位置で両方の画像に適用
        hr_image = F.crop(hr_image, i * 2, j * 2, h * 2, w * 2)
        lr_image = F.crop(lr_image, i, j, h, w)

        random.seed(seed)
        torch.manual_seed(seed)
        lr_image = self.base_transform(lr_image)

        random.seed(seed)
        torch.manual_seed(seed)
        hr_image = self.base_transform(hr_image)

        #ランダムな明るさの設定 #白と黒がある部分にRGBのアーティファクトのノイズが目立つためコメントアウト
        brightness_factor = random.uniform(0.8, 1.05)
        enhancer = ImageEnhance.Brightness(lr_image)
        lr_image = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Brightness(hr_image)
        hr_image = enhancer.enhance(brightness_factor)

        # ランダムなコントラストの設定
        contract_factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(lr_image)
        lr_image = enhancer.enhance(contract_factor)
        enhancer = ImageEnhance.Contrast(hr_image)
        hr_image = enhancer.enhance(contract_factor)

        # 最後にテンソル変換と正規化を適用
        lr_image = self.tensor_transform(lr_image)
        hr_image = self.tensor_transform(hr_image)

        return lr_image, hr_image
