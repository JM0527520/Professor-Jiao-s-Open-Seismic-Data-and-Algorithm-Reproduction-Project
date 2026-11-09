import numpy as np
from os.path import splitext
from os import listdir
from functions import *

subset = 'train'

seis_path = f"/mnt/HD_18T_pt1/chen/Datasets/FaultSeg3D/{({'train':'train', 'test' :'validation'}[subset])}/seis"
fault_path = f"/mnt/HD_18T_pt1/chen/Datasets/FaultSeg3D/{({'train':'train', 'test' :'validation'}[subset])}/fault"

ids = [splitext(file)[0] for file in listdir(seis_path) if not file.startswith('.')]
ids.sort(key=int)

if subset=='train': ids = ids[:int(0.1*len(ids))]# take only 10 percent

t_start = time.time()
# "axis 0 ")
image_list1, mask_list1 = [], []
# "axis 1 ")
image_list2, mask_list2 = [], []

for idx in (ids):
    # print("{}/{}.dat".format(fault_path,idx))
    # iterate current cube
    label_volume = np.fromfile("{}/{}.dat".format(fault_path,idx),dtype=np.single).reshape(128,128,128)
    image_volume = np.fromfile("{}/{}.dat".format(seis_path,idx),dtype=np.single).reshape(128,128,128)
    assert image_volume.shape==label_volume.shape
    # print(idx,  image_volume.shape, )
    # iterate all hundered slcies
    for ithslice in (range(len(image_volume))):
        # axix 0
        image_list1.append(image_volume[ithslice,:,:].squeeze().transpose())
        mask_list1.append(label_volume[ithslice,:,:].squeeze().transpose())
        image_list2.append(image_volume[:,ithslice,:].squeeze().transpose())
        mask_list2.append(label_volume[:,ithslice,:].squeeze().transpose())


assert len(image_list1) == len(mask_list1) == len(image_list2) == len(mask_list2)
print(len(image_list1) , len(mask_list1))
print(image_list1[0].shape)
# i_cnt += len(X_list)
print("read {} images and labels in {} sec".format(len(image_list1), time.time()-t_start))    

save_path = os.path.join('/mnt/HD_18T_pt1/chen/Datasets/FaultSeg3D/sliced', subset)

os.makedirs(save_path,exist_ok=True)
for imageNo, (mask1, mask2) in enumerate(zip(mask_list1, mask_list2)):
    np.save(os.path.join(save_path,"{}_{}".format(imageNo,'axis_0')),mask1)   
    np.save(os.path.join(save_path,"{}_{}".format(imageNo,'axis_1')),mask2)   