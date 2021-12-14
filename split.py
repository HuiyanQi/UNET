import os
import shutil




ROOT_PATH = '/workspace/qhy/UNET/data/'
# ROOT_PATH = 'D:/qhy/UNET/data/'
images_path = ROOT_PATH + 'images/'
masks_path = ROOT_PATH + 'masks/'
train_images_dir = ROOT_PATH + 'train_images/'
train_masks_dir = ROOT_PATH + 'train_masks/'
val_images_dir = ROOT_PATH + 'val_images/'
val_masks_dir = ROOT_PATH + 'val_masks/'

images = os.listdir(images_path)
masks = os.listdir(masks_path)
images.sort()
masks.sort()
train_percent = 0.8
val_percent = 0.2

num = len(images)
for i in range(num):
    if i <= train_percent * num:
        if os.path.isdir(train_images_dir):
            shutil.copy(images_path + images[i],train_images_dir + images[i])
        else:
            os.makedirs(train_images_dir)
            shutil.copy(images_path + images[i],train_images_dir + images[i])
        if os.path.isdir(train_masks_dir):
            shutil.copy(masks_path + masks[i],train_masks_dir + masks[i])
        else:
            os.makedirs(train_masks_dir)
            shutil.copy(masks_path + masks[i],train_masks_dir + masks[i])
    else:
        if os.path.isdir(val_images_dir):
            shutil.copy(images_path + images[i],val_images_dir + images[i])
        else:
            os.makedirs(val_images_dir)
            shutil.copy(images_path + images[i],val_images_dir + images[i])
        if os.path.isdir(val_masks_dir):
            shutil.copy(masks_path + masks[i],val_masks_dir + masks[i])
        else:
            os.makedirs(val_masks_dir)
            shutil.copy(masks_path + masks[i],val_masks_dir + masks[i])


