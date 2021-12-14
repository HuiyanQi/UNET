import torch
import os
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from model import UNET
from dataset import CarvanaDataset
from utils import train_one_epoch, evaluate, parse




args = parse()
def main():
    train_transform = transforms.Compose([transforms.Resize((args.image_height,args.image_width)),
                                transforms.RandomRotation(degrees=35),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])])
    mask_transform = transforms.Compose([transforms.Resize((args.image_height, args.image_width)),
                        transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.Resize((args.image_height, args.image_width)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])])


    train_dataset = CarvanaDataset(image_dir=args.train_image_dir, mask_dir = args.train_mask_dir, image_transform=train_transform, mask_transform=mask_transform)
    val_dataset = CarvanaDataset(image_dir=args.val_image_dir, mask_dir = args.val_mask_dir, image_transform=val_transform, mask_transform=mask_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)
    model = UNET(in_channels=3,out_channels=1).cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    best_acc = -1
    for epoch in range (args.epochs):
        train_loss = train_one_epoch(epoch=epoch,data_loader=train_loader, model=model, optimizer=optimizer, loss_fn=loss_fn)
        val_loss, val_acc = evaluate(epoch=epoch, model=model,data_loader=val_loader, loss_fn=loss_fn)
        if val_acc > best_acc:
            best_acc = val_acc
            if os.path.exists(args.model_weights_dir):
                torch.save(model.state_dict(),"./weights/model-{}.pth".format(epoch))
            else:
                os.makedirs(args.model_weights_dir)
                torch.save(model.state_dict(),"./weights/model-{}.pth".format(epoch))

if __name__ == "__main__":
    main()
