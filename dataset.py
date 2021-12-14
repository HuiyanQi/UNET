import os
from PIL import Image
from torch.utils.data import Dataset

class CarvanaDataset(Dataset):
    def __init__(self,image_dir,mask_dir,image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = os.listdir(image_dir)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir,self.images[item])
        mask_path = os.path.join(self.mask_dir,self.images[item].replace(".jpg", "_mask.gif"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
            mask[mask == 255.0] = 1.0
        return image, mask
