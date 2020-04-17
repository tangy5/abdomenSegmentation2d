
import torch.utils.data as data
import os
import random
import glob
from PIL import Image
import numpy as np
import nibabel as nb

label_dir = ''
gt_dir = ''


def dice(nparray, gtarray):
  return np.sum(nparray[gtarray==1])*2.0 / (np.sum(nparray) + np.sum(gtarray)+0.000000001) 

count = 0
dice_file = os.path.join(output_dir, 'dice_result_24.txt')
dice_wr = open(dice_file, 'w')
average_dice = 0
average_count = 0

organ_dice_list = [0] * 13
organ_dice_count = [0] * 13
for img in os.listdir(label_dir):
    if img.endswith('.nii.gz'):
      count += 1
      label_path = os.path.join(label_dir, img)
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
dice_wr.close()


average_organ_dice = []
for i in range(13):
  average_organ_dice.append(organ_dice_list[i]/organ_dice_count[i])



print(average_dice/average_count)
print(average_organ_dice)