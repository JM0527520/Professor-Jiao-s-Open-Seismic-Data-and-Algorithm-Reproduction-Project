import numpy as np
import time
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, generate_binary_structure
from functions import *
from tqdm import tqdm


im_height = 1537
im_width = 3174
splitsize = 96
stepsize = 48 #overlap half
overlapsize = splitsize-stepsize


horizontal_splits_number = int(np.ceil((im_width)/stepsize))
print(horizontal_splits_number)
width_after_pad = stepsize*horizontal_splits_number+2*overlapsize
print(width_after_pad)
left_pad = int((width_after_pad-im_width)/2)
right_pad = width_after_pad-im_width-left_pad
print(left_pad,right_pad)

vertical_splits_number = int(np.ceil((im_height)/stepsize))
print(vertical_splits_number)
height_after_pad = stepsize*vertical_splits_number+2*overlapsize
print(height_after_pad)
top_pad = int((height_after_pad-im_height)/2)
bottom_pad = height_after_pad-im_height-top_pad
print(top_pad,bottom_pad)


horizontal_splits_number = horizontal_splits_number+1
print(horizontal_splits_number)
vertical_splits_number = vertical_splits_number+1
print(vertical_splits_number)

halfoverlapsize = int(overlapsize/2)
print(halfoverlapsize)

seis_path = '/mnt/data-tmp/chen/Datasets/Thebe/test/seis/'
fault_path = '/mnt/data-tmp/chen/Datasets/Thebe/test/fault/'
t_start = time.time()

# sectionNumber = 141
# if isval:
#     sectionNumber = 40
# for i in range(0,sectionNumber,1): #len(seis)
image_list, mask_list = [], []
i_cnt = 0
# Iterate 4 image_volumes
for i_vol in range(1, 2):
    # image_list, mask_list = [], []
    file_path = seis_path + "seistest{}.npy".format(i_vol)

    image_volume = np.load(file_path)
    label_volume = np.load(fault_path + 'faulttest{}.npy'.format(i_vol))
    assert image_volume.shape==label_volume.shape
    print(file_path,  image_volume.shape, )
    # iterate all hundered slcies
    for i_sect in tqdm(range(len(image_volume))):
        img = image_volume[i_sect].T
        lbl = label_volume[i_sect].T

        # Apply skeletonization
        skeleton = skeletonize(lbl)
        
        # Apply dilation
        s = generate_binary_structure(2, 1)
        dilated = binary_dilation(skeleton, structure=s, iterations=1)
        
        # Save the result as a .npy file
        np.save(f'/mnt/data-tmp/chen/Datasets/Thebe/test/fault/morphed/vol_{i_vol}_slice_{i_sect}.npy', dilated)
        # 
        image_list.extend(split_Image(img,True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
        mask_list.extend(split_Image(dilated,True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
    
    assert len(image_list) == len(mask_list)
    print(len(image_list), len(mask_list))
    print(image_list[0].shape)
    i_cnt += len(image_list)
    print("read {} images and labels in {} sec".format(i_cnt, time.time()-t_start))

            
# X = np.asarray(image_list)
# print(len(mask_list),) ### 231200
# exit()
# print("read images in {} sec".format(time.time()-t_start))
assert len(image_list)==len(mask_list)

def save_to_mmap(image_list, full_shape, dtype, filename):
    # Create a memory-mapped array in write mode.
    # This file will store the full dataset without loading it all into RAM.
    mmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=full_shape)
    # 
    # Fill the memmap with each image (stacking them along the first axis).
    for i, img in enumerate(image_list):
        mmap_array[i] = img
    
    # Flush changes to ensure they are written to disk.
    mmap_array.flush()

# Example settings: each image is (height, width, channels)
image_shape = image_list[0].shape
num_images = len(image_list)  # For demonstration; in your case, this might be 300,000
# Define the full shape for the stacked dataset.
# This will be (num_images, height, width, channels)
full_shape = (num_images,) + image_shape
# Specify a filename for the memory-mapped file.
save_to_mmap(image_list, full_shape, np.float64, filename = f'/mnt/data-tmp/chen/Datasets/Thebe/test/seis/memmap/images_memmap{i_vol}.dat')

save_to_mmap(mask_list, full_shape, np.bool_, filename = f'/mnt/data-tmp/chen/Datasets/Thebe/test/fault/memmap/masks_memmap{i_vol}.dat')

print(full_shape)
print(full_shape)
print(full_shape)
