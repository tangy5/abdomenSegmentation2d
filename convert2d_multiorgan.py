
import torch.utils.data as data
import os
import random
import glob
from PIL import Image
import numpy as np
import nibabel as nb

image_dir = ''
label_dir = ''

image2d_dir=''
label2d_dir = ''

count = 0
for img in os.listdir(image_dir):
    count += 1
    image_idx = int(img.split('.nii.gz')[0].split('img')[1])
    image_path = os.path.join(image_dir, img)
    # if os.path.isfile(image_path) and os.path.isfile(seg_file):
    imgnb = nb.load(image_path)
    imgnp = np.array(imgnb.dataobj)

    label_path = os.path.join(label_dir, img)
    labelnb = nb.load(label_path)
    labelnp = np.array(labelnb.dataobj)
    # convert 17,18 to 14
    idx = np.where(labelnp == 17)
    labelnp[idx] = 14
    idx = np.where(labelnp == 18)
    labelnp[idx] = 14
    # idx = np.where(labelnp == 16)
    # labelnp[idx] = 0
    # idx = np.where(labelnp == 17)
    # labelnp[idx] = 0
    # idx = np.where(labelnp == 18)
    # labelnp[idx] = 0

    idx = np.where(imgnp < -125)
    imgnp[idx] = -125
    idx = np.where(imgnp > 275)
    imgnp[idx] = 275

    # imgnb_new = nb.Nifti1Image(imgnp, imgnb.affine)
    # img_newfile = os.path.join(output_dir, img)
    # nb.save(imgnb_new, img_newfile)


    z_range = imgnp.shape[2]
    
    if not os.path.isdir(image2d_dir):
        os.makedirs(image2d_dir)
    if not os.path.isdir(label2d_dir):
        os.makedirs(label2d_dir)

    for i in range(z_range):
        slice2dnp = (imgnp[:,:,i] - imgnp[:,:,i].min()) * 255.0 / (imgnp[:,:,i].max() - imgnp[:,:,i].min())
        slice2d = Image.fromarray(slice2dnp.astype(np.uint8)).rotate(90)
        slice2d = slice2d.convert('RGB')
                        
        #save label png
        label2dnp = labelnp[:,:,i]
        label2d = Image.fromarray(label2dnp.astype(np.uint8)).rotate(90)
        label2d = label2d.convert('L')
                        
        # split to train and valid
        if image_idx <= 80 and image_idx > 60:
          train_out = os.path.join(image2d_dir, 'val')
          label_out = os.path.join(label2d_dir, 'val')
        else:
          train_out = os.path.join(image2d_dir, 'train')
          label_out = os.path.join(label2d_dir, 'train')   

        image2d_file = os.path.join(train_out, '{}_{}.png'.format(img, i))
        slice2d.save(image2d_file)
        label2d_file = os.path.join(label_out, '{}_{}.png'.format(img, i))
        label2d.save(label2d_file)
    print('[{}] -- {} processed'.format(count, img))





# single image

count = 0
for img in os.listdir(image_dir):
    count += 1
    # image_idx = int(img.split('.nii.gz')[0].split('img')[1])
    image_path = os.path.join(image_dir, img)
    # if os.path.isfile(image_path) and os.path.isfile(seg_file):
    imgnb = nb.load(image_path)
    imgnp = np.array(imgnb.dataobj)


    idx = np.where(imgnp < -125)
    imgnp[idx] = -125
    idx = np.where(imgnp > 275)
    imgnp[idx] = 275




    z_range = imgnp.shape[2]
    
    if not os.path.isdir(image2d_dir):
        os.makedirs(image2d_dir)


    for i in range(z_range):
        slice2dnp = (imgnp[:,:,i] - imgnp[:,:,i].min()) * 255.0 / (imgnp[:,:,i].max() - imgnp[:,:,i].min())
        slice2d = Image.fromarray(slice2dnp.astype(np.uint8)).rotate(90)
        slice2d = slice2d.convert('RGB')
                        

                        
        # split to train and valid
        train_out = os.path.join(image2d_dir)
        # train_out = os.path.join(image2d_dir, 'train')

        image2d_file = os.path.join(train_out, '{}_{}.png'.format(img, i))
        slice2d.save(image2d_file)
    print('[{}] -- {} processed'.format(count, img))