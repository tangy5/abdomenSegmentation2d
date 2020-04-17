import os
import matplotlib.pyplot as plt
import numpy as np


# put all txt file in a folder
txt_dir = ''

datascore_file_list = []
for txt_file in os.listdir(txt_dir):
    if txt_file.endswith('.txt'):
        datascore_file_list.append(txt_file)

file2dicelist = {}
for txt_file in datascore_file_list:
    txt_file_path = os.path.join(txt_dir, txt_file)
    with open(txt_file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    dice_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
    for row_idx, row in enumerate(content):
        if row_idx < 20:
            image_name = row.split(' ')[0]
            for idx in range(12):
                dice_list[idx].append(float(row.split(' ')[idx]))
    file2dicelist[txt_file] = dice_list

# calculatge mean DSC and set result in order
txt2meanDSC = {}
for txt_name in file2dicelist:
    result_list = file2dicelist[txt_name]
    result_mean = np.mean(np.array(result_list))
    txt2meanDSC[txt_name] = result_mean
txt2meanDSC = {k: v for k, v in sorted(txt2meanDSC.items(), key=lambda item: item[1])}


# combine list group (12 groups)
groups = []
for i in range(12):
    g = []
    for txt_name in txt2meanDSC:
        g.append(file2dicelist[txt_name][i])
    groups.append(g)

fig1, ax1 = plt.subplots()
green_diamond = dict(markerfacecolor='g', marker='D')
# ax1.set_title('Boxplot Test Dice of Random Patch (20)')
# ax1.set_xlabel('Organs')
# ax1.set_ylabel('Dice')
position = [i+1 for i in range(len(datascore_file_list))]
for i in range(len(groups)):
    bplot = ax1.boxplot(groups[i], positions=position,flierprops=green_diamond, patch_artist=True,widths=[0.6]*len(datascore_file_list))
    position = [j+len(datascore_file_list) for j in position]

    colors = ['#C9B498','#d3aea6', '#a2a2a3','#a0ccb3']
    # colors = ['#d3aea6', '#a2a2a3','#a0ccb3']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

ax1.set_ylim([-0.1, 1.05])
xticks_position = []
for p in range(12):
    po = p *  math.floor(len(datascore_file_list)) + math.floor(len(datascore_file_list) / 2)
    xticks_position.append(po)
plt.xticks(xticks_position, ['spleen', 'kigney(R)', 'kidney(L)', 'gallbladder', 'esophagus', 'liver',\
                'stomach', 'aorta', 'IVC', 'P&S veins', 'pancreas', 'Ad Gland'])
plt.show()















# save list to txt
output_dir = ''
save_result_txt = os.path.join(output_dir, 'base_result_dice_list_new')
wr_file = open(save_result_txt,'w')
for case_idx in range(20):
    for organ_idx in range(12):
        organ_dice = dice_list2[organ_idx][case_idx]
        if organ_idx == 11:
            wr_file.write(str(organ_dice) + '\n')
        else:
            wr_file.write(str(organ_dice) + ' ')
wr_file.close()


## overlay single slices

count = 0

# for con_dir in os.listdir(contrast_dir):
#     image_dir = os.path.join(contrast_dir, con_dir)
#     seg_dir = os.path.join(seg_contrast_dir, con_dir)
image2d_dir = ''
result_slice_dir = ''

for img in os.listdir(result_slice_dir):
    count += 1
    seg_file = os.path.join(result_slice_dir, img)
    image_path = os.path.join(image2d_dir, img)

    slice2d = Image.open(image_path)                                             
    overlayslice_image = Image.open(seg_file).convert('RGB')

    overlay = Image.blend(slice2d, overlayslice_image, 0.4)
    overlay_file = os.path.join(output_dir, img)
    overlay.save(overlay_file)
    print('[{}] -- processed'.format(count))