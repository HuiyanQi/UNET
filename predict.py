import torchvision.utils
from tqdm import tqdm
import argparse
import torch
from torchvision import transforms
from model import UNET
import os
from PIL import Image
from torch.utils.data import Dataset
def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default = os.cpu_count())
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=160) #1280
    parser.add_argument('--image_width', type=int, default=240) #1918
    parser.add_argument('--test_image_dir', type=str, default="/workspace/qhy/UNET/data/test/test/")
    parser.add_argument('--save_images_folder', type=str, default="./test_results/")
    parser.add_argument('--model_weights_dir', type=str, default='./weights')
    args = parser.parse_args()
    return args
args = parse()

class testDataset(Dataset):
    def __init__(self,image_dir,image_transform=None):
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.images = os.listdir(image_dir)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir,self.images[item])
        image = Image.open(img_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image

def test(model, data_loader):
    model.eval()
    data_loader = tqdm(data_loader)
    save_dir = args.save_images_folder
    for step, images in enumerate(data_loader):
        images = images.cuda()
        with torch.no_grad():
            pred = model(images)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            data_loader.desc = "step:{}".format(step)
        if os.path.exists(save_dir):
            torchvision.utils.save_image(pred, f"{save_dir}/pred_{step}.png")
        else:
            os.makedirs(save_dir)
            torchvision.utils.save_image(pred, f"{save_dir}/pred_{step}.png")

def main():
    model = os.listdir(args.model_weights_dir)
    model.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    model_weight_path = model[len(model) - 1]
    model_weight_path = os.path.join(args.model_weights_dir, model_weight_path)
    test_transform = transforms.Compose([transforms.Resize((args.image_height, args.image_width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])])


    test_dataset = testDataset(image_dir=args.test_image_dir, image_transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)
    model = UNET(in_channels=3,out_channels=1).cuda()
    model.load_state_dict(torch.load(model_weight_path, map_location='cuda:0'))
    test(model=model,data_loader=test_loader)

if __name__ == "__main__":
    main()
