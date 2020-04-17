import torch.utils.data as data
import os
import random
import glob
from PIL import Image
import numpy as np
import nibabel as nb

image_dir = ''
gt_dir = ''

label2d_dir = ''

output_dir = ''

result2d = ''

for i in range(10, 51):
  label2d_dir = os.path.join(result2d, 'val_{}'.format(i))
  output_dir = os.path.join(result2d, 'result_{}'.format(i))
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  count = 0
  imagename2file = {}
  for img in os.listdir(label2d_dir):
    count += 1
    image_name = img.split('_')[0]
    image_idx = int(img.split('_')[1].split('.png')[0])
    image_path = os.path.join(label2d_dir, img)
    if image_name not in imagename2file:
      imagename2file[image_name] = {}
    imagename2file[image_name][image_idx] = image_path

  count = 0
  for item in imagename2file:
    image_file = os.path.join(image_dir, item)
    imgnb = nb.load(image_file)
    label_np = np.zeros((imgnb.shape[0], imgnb.shape[1], imgnb.shape[2]))
    for i in range(label_np.shape[2]):
      labelslice_file = imagename2file[item][i]
      labelslice_image = Image.open(labelslice_file).rotate(-90)
      labelslice_np = np.array(labelslice_image)
      label_np[:,:,i] = labelslice_np
    label_nb = nb.Nifti1Image(label_np, imgnb.affine)
    label_file = os.path.join(output_dir, item)
    nb.save(label_nb, label_file)
    count += 1
    print('[{}] converted {}'.format(count, item))





def dice(nparray, gtarray):
  return np.sum(nparray[gtarray==1])*2.0 / (np.sum(nparray) + np.sum(gtarray)+0.000000001) 






for i in range(10, 51):
  count = 0
  output_dir = os.path.join(result2d, 'result_{}'.format(i))
  dice_file = os.path.join(output_dir, 'dice_result_{}.txt'.format(i))
  dice_wr = open(dice_file, 'w')
  average_dice = 0
  average_count = 0

  organ_dice_list = [0] * 13
  organ_dice_count = [0] * 13
  for img in os.listdir(output_dir):
      if img.endswith('.nii.gz'):
        count += 1
        label_path = os.path.join(output_dir, img)
        # if os.path.isfile(image_path) and os.path.isfile(seg_file):
        labelnb = nb.load(label_path)
        labelnp = np.array(labelnb.dataobj)

        gt_path = os.path.join(gt_dir, img)
        gtnb = nb.load(gt_path)
        gtnp = np.array(gtnb.dataobj)
        DSC_list = []
        for i in range(1, 14):
          idx = np.where(labelnp == i)
          organ_np = np.zeros((labelnp.shape[0], labelnp.shape[1], labelnp.shape[2]))
          organ_np[idx] = 1

          idx = np.where(gtnp == i)
          gt_organ = np.zeros((gtnp.shape[0], gtnp.shape[1], gtnp.shape[2]))
          gt_organ[idx] = 1

          organ_DSC = dice(organ_np, gt_organ)
          DSC_list.append(organ_DSC)
          average_dice += organ_DSC

          organ_dice_list[i-1] += organ_DSC
          if organ_DSC > 0.01:
            average_count += 1
            organ_dice_count[i-1] += 1
        for j, item in enumerate(DSC_list):
          if j == len(DSC_list)-1:
            dice_wr.write(str(item) + '\n')
          else:
            dice_wr.write(str(item) + ' ')
        print('[{}] -- {} processed'.format(count, img))

  average_organ_dice = []
  for i in range(13):
    average_organ_dice.append(organ_dice_list[i]/organ_dice_count[i])
    dice_wr.write(str(organ_dice_list[i]/organ_dice_count[i]) + ' ')
  dice_wr.write(str(average_dice/average_count))

  print(average_dice/average_count)
  print(average_organ_dice)  
  dice_wr.close()


#single



if not os.path.isdir(output_dir):
  os.makedirs(output_dir)
count = 0
imagename2file = {}
for img in os.listdir(label2d_dir):
  count += 1
  image_name = img.split('.nii.gz')[0] + '.nii.gz'
  image_idx = int(img.split('.nii.gz_')[1].split('.png')[0])
  image_path = os.path.join(label2d_dir, img)
  if image_name not in imagename2file:
    imagename2file[image_name] = {}
  imagename2file[image_name][image_idx] = image_path

count = 0
for item in imagename2file:
  image_file = os.path.join(image_dir, item)
  imgnb = nb.load(image_file)
  label_np = np.zeros((imgnb.shape[0], imgnb.shape[1], imgnb.shape[2]))
  for i in range(label_np.shape[2]):
    labelslice_file = imagename2file[item][i]
    labelslice_image = Image.open(labelslice_file).rotate(-90)
    labelslice_np = np.array(labelslice_image)
    label_np[:,:,i] = labelslice_np
  label_nb = nb.Nifti1Image(label_np, imgnb.affine)
  label_file = os.path.join(output_dir, item)
  nb.save(label_nb, label_file)
  count += 1
  print('[{}] converted {}'.format(count, item))


def dice(nparray, gtarray):
  return np.sum(nparray[gtarray==1])*2.0 / (np.sum(nparray) + np.sum(gtarray)+0.000000001) 

count = 0
# output_dir = os.path.join(result2d, 'result_{}'.format(i))
dice_file = os.path.join(output_dir, 'dice_result_{}.txt'.format(i))
dice_wr = open(dice_file, 'w')
average_dice = 0
average_count = 0

organ_dice_list = [0] * 13
organ_dice_count = [0] * 13
for img in os.listdir(output_dir):
    if img.endswith('.nii.gz'):
      count += 1
      label_path = os.path.join(output_dir, img)
      # if os.path.isfile(image_path) and os.path.isfile(seg_file):
      labelnb = nb.load(label_path)
      labelnp = np.array(labelnb.dataobj)

      gt_path = os.path.join(gt_dir, img)
      gtnb = nb.load(gt_path)
      gtnp = np.array(gtnb.dataobj)
      DSC_list = []
      for i in range(1, 14):
        idx = np.where(labelnp == i)
        organ_np = np.zeros((labelnp.shape[0], labelnp.shape[1], labelnp.shape[2]))
        organ_np[idx] = 1

        idx = np.where(gtnp == i)
        gt_organ = np.zeros((gtnp.shape[0], gtnp.shape[1], gtnp.shape[2]))
        gt_organ[idx] = 1

        organ_DSC = dice(organ_np, gt_organ)
        DSC_list.append(organ_DSC)
        average_dice += organ_DSC

        organ_dice_list[i-1] += organ_DSC
        if organ_DSC > 0.01:
          average_count += 1
          organ_dice_count[i-1] += 1
      for j, item in enumerate(DSC_list):
        if j == len(DSC_list)-1:
          dice_wr.write(str(item) + '\n')
        else:
          dice_wr.write(str(item) + ' ')
      print('[{}] -- {} processed'.format(count, img))

average_organ_dice = []
for i in range(13):
  average_organ_dice.append(organ_dice_list[i]/organ_dice_count[i])
  dice_wr.write(str(organ_dice_list[i]/organ_dice_count[i]) + ' ')
dice_wr.write(str(average_dice/average_count))

print(average_dice/average_count)
print(average_organ_dice)  
dice_wr.close()