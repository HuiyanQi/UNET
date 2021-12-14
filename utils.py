import os
import torch
import torchvision.utils
from tqdm import tqdm
import argparse

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default = os.cpu_count())
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_height', type=int, default=160) #1280
    parser.add_argument('--image_width', type=int, default=240) #1918
    parser.add_argument('--train_image_dir', type=str, default="./data/train_images/")
    parser.add_argument('--train_mask_dir', type=str, default="./data/train_masks/")
    parser.add_argument('--val_image_dir', type=str, default="./data/val_images/")
    parser.add_argument('--val_mask_dir', type=str, default="./data/val_masks/")
    parser.add_argument('--save_images_folder', type=str, default="./saved_images/")
    parser.add_argument('--model_weights_dir', type=str, default='./weights')

    args = parser.parse_args()

    return args
args = parse()
def train_one_epoch(epoch, data_loader, model, optimizer, loss_fn):
    model.train()
    data_loader = tqdm(data_loader)
    total_train_loss = 0
    for step,data in enumerate(data_loader):
        images, targets = data
        images, targets = images.cuda(), targets.float().cuda()
        pred = model(images)
        loss = loss_fn(pred,targets)
        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch,total_train_loss / (step + 1))
    return total_train_loss / (step + 1)


def evaluate(model, data_loader, epoch, loss_fn):
    num_corract = 0
    num_pixels = 0
    total_test_loss = 0
    dice_score = 0
    model.eval()
    data_loader= tqdm(data_loader)
    save_dir = os.path.join(args.save_images_folder,str(epoch))
    for step, data in enumerate(data_loader):
        images, targets = data
        images, targets = images.cuda(), targets.float().cuda()
        with torch.no_grad():
            pred = model(images)
            loss = loss_fn(pred, targets)
            total_test_loss += loss.item()
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            num_corract += (pred == targets).sum()
            num_pixels += torch.numel(pred)
            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.2f}%, dice score: {:.2f}".format(epoch, total_test_loss / (step + 1), num_corract / num_pixels * 100, dice_score / (step + 1))
            dice_score += (2 * (pred * targets).sum()) / ((pred + targets).sum() + 1e-8)
        if os.path.exists(save_dir):
            torchvision.utils.save_image(pred, f"{save_dir}/pred_{step}.png")
        else:
            os.makedirs(save_dir)
            torchvision.utils.save_image(pred, f"{save_dir}/pred_{step}.png")


    return total_test_loss / (step + 1), num_corract / num_pixels * 100

