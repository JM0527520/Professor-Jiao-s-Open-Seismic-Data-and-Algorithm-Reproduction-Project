import argparse
import torch
import torch.nn as nn
from torchvision import models
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from pytorchtools import EarlyStopping
from os.path import splitext
from os import listdir
import time
from functions import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1 is visible to this code
seed = 1
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

torch.cuda.manual_seed(seed)
def _init_fn(worker_id):
    np.random.seed(int(seed))

def get_model(modelName, isPretrain):
    if modelName == "deeplabv3_resnet101":
        if isPretrain == "cocoft":
            print("use pretrain cocoft")
            model = models.segmentation.deeplabv3_resnet101(pretrained=True,aux_loss=False)
            pretrain_model_fpath = "coco pretrain"     
        elif isPretrain == "bsdsft":
            print("use pretrain bsdsft")
            model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=False, pretrained_backbone=False)
            pretrain_model_fpath = "deeplabv3_resnet101_bsds_3_10_0.001_DN_100_20_crop96_b32_trainaugvalnohvflip"
        elif isPretrain == "faultSegft":
            print("use pretrain faultSegft")
            model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=False, pretrained_backbone=False)
            pretrain_model_fpath = "deeplabv3_resnet101_faultSeg_3_10_0.001_DN_100_20_crop96_b32_trainaugvalnohflip"
        elif isPretrain == "none":
            model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=False, pretrained_backbone=False)
        elif isPretrain == "imagenetft":
            model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=False, pretrained_backbone=True)
            pretrain_model_fpath = "imagenet pretrain"   
    elif modelName == "unet":
        from model_zoo.UNET import Unet
        model = Unet()
        print("use model Unet")
        if isPretrain == "bsdsft":
            pretrain_model_fpath = "unet_bsds_3_10_0.01_DN"
        elif isPretrain == "faultSegft":
            pretrain_model_fpath = "unet_faultSeg_3_10_0.01_DN_v2"
    elif modelName == "deeplab":
        from model_zoo.DEEPLAB.deeplab import DeepLab
        model = DeepLab(backbone='mobilenet', num_classes=1, output_stride=16)
        print("use model DeepLab")
        if isPretrain == "bsdsft":
            pretrain_model_fpath = "deeplab_bsds_3_10_0.01_DN"
        elif isPretrain == "faultSegft":
            pretrain_model_fpath = "deeplab_faultSeg_3_10_0.01_DN_v2"
    elif modelName == "hed":
        from model_zoo.HED import HED
        model = HED()
        print("use model HED")
        if isPretrain == "bsdsft":
            pretrain_model_fpath = "hed_bsds_3_10_0.001_DN"
        elif isPretrain == "faultSegft":
            pretrain_model_fpath = "hed_faultSeg_3_10_0.001_DN_v2"
    elif modelName == "rcf":
        from model_zoo.RCF import RCF
        model = RCF()
        print("use model RCF")
        if isPretrain == "bsdsft":
            pretrain_model_fpath = "rcf_bsds_3_10_0.001_DN"
        elif isPretrain == "faultSegft":
            pretrain_model_fpath = "rcf_faultSeg_3_10_0.001_DN_v2"
    else:
        print("please select a valid model")
    if isPretrain != "none":
        print("pretrain weights are: " + pretrain_model_fpath)


    # In[2]:


    # print(model)
    return model

def random_partition(my_list, ratio=0.9):
    # Shuffle the list randomly
    random.shuffle(my_list)
    
    # Calculate the index for the split
    split_index = int(len(my_list) * ratio)
    
    # Split the list into two sublists
    sublist1 = my_list[:split_index]
    sublist2 = my_list[split_index:]
    
    return sublist1, sublist2

def get_dataloaders(targetData, data_path, label_source, batch_size, isDatasetNorm, isSourceDatasetNorm, isClip, isExtremOutlier, isPretrain, modelName):
    if targetData == 'thebe':
        seis_path = os.path.join(data_path,'seis/memmap/images_memmap.dat')
        fault_path = os.path.join(data_path,'fault/memmap/masks_memmap.dat')
        full_shape = (41320, 96, 96)
       
        images_memmap = np.memmap(seis_path, dtype=np.float64, mode='r', shape=full_shape)

        masks_memmap = np.memmap(fault_path, dtype=np.bool_, mode='r', shape=full_shape)

        # split train/val
        train_list, val_list = random_partition(list(range(len(masks_memmap))) , ratio=0.9)


        faults_dataset_train = thebeDataset(imgs=images_memmap, masks=masks_memmap, ids=train_list, isDatasetNorm=isDatasetNorm, isSourceDatasetNorm=isSourceDatasetNorm, isPretrain=isPretrain, modelName=modelName)
        faults_dataset_val = thebeDataset(imgs=images_memmap, masks=masks_memmap, ids=val_list, isDatasetNorm=isDatasetNorm, isSourceDatasetNorm=isSourceDatasetNorm, isPretrain=isPretrain, modelName=modelName)
        train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
                                               batch_size=batch_size, 
                                               shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
                                               batch_size=batch_size, 
                                               shuffle=False)  
    
    # elif targetData == 'bsds':    
    #     faults_dataset_train = bsdsDataset(imgs_dir = "{}/images/train".format(data_path), masks_dir= "{}/GT_np/train".format(data_path), aug = True)
    #     faults_dataset_val = bsdsDataset(imgs_dir = "{}/images/val".format(data_path), masks_dir= "{}/GT_np/val".format(data_path), aug = False)
    #     train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
    #                                            batch_size=batch_size, 
    #                                            shuffle=True)
    #     val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
    #                                            batch_size=batch_size, 
    #                                            shuffle=False)
    elif targetData == 'faultSeg':
        faults_dataset_full = faultSegDataset(imgs_dir = "{}/train/seis".format(data_path), masks_dir= "{}/train/fault".format(data_path), aug = True, isDatasetNorm=isDatasetNorm, isClip=isClip, isExtremOutlier=isExtremOutlier, modelName=modelName)

        # Define split sizes
        train_size = int(0.9 * len(faults_dataset_full))
        test_size = len(faults_dataset_full) - train_size
        
        # Split the dataset
        faults_dataset_train, faults_dataset_val = torch.utils.data.random_split(
            faults_dataset_full,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(seed)  # Seed for reproducibility
        )            
        train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
                                                   batch_size=batch_size, 
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
                                                   batch_size=batch_size, 
                                                   shuffle=False)
    elif targetData == 'f3':
        # data_path = '/data/CRACKS'
        seis_path = os.path.join(data_path,'image_volume.npy')
        image_volume = np.load(seis_path)
        image_volume = np.swapaxes(image_volume, 1, 2)
        image_volume = image_volume[30:-30, :, :]
        if label_source  == 'novice15':
            label_volume=np.load(os.path.join(data_path,'novices/novice15_npy_morphed.npy'))
        elif label_source == 'practitioner3':
            label_volume=np.load(os.path.join(data_path,'practitioners/practitioner3_npy_morphed.npy'))
        elif label_source == 'expert':
            label_volume=np.load(os.path.join(data_path,'experts/ATHOQ52EXZNQ9_npy_morphed.npy'))
        else:
            raise
        label_volume = label_volume[30:-30, :, :]
        assert len(label_volume)==len(image_volume)

        im_height = image_volume.shape[-2:][0]#1537
        im_width = image_volume.shape[-2:][1] #3174
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

        # split train/val
        train_list, val_list = random_partition(list(range(len(image_volume))), ratio=0.9)
        train_patches, val_patches = [], []
        train_labels, val_labels = [], []
        for i in train_list:
            # print("extracting train {}.npy".format(i)) 
            img = image_volume[i]
            lbl = label_volume[i]
            train_patches.extend(split_Image(img,True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
            train_labels.extend(split_Image(lbl,True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
        
        print('read train images:',  len(train_patches))

        for i in val_list:
            # print("extracting validation {}.npy".format(i)) 
            img = image_volume[i]
            lbl = label_volume[i]
            val_patches.extend(split_Image(img,True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
            val_labels.extend(split_Image(lbl,True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
        print('read validation images:',  len(val_patches))
        faults_dataset_train = f3Dataset(train_patches, train_labels , aug = True, isDatasetNorm=isDatasetNorm, isClip=isClip, isExtremOutlier=isExtremOutlier, modelName=modelName)
        faults_dataset_val = f3Dataset(val_patches, val_labels , aug = False, isDatasetNorm=isDatasetNorm, isClip=isClip, isExtremOutlier=isExtremOutlier, modelName=modelName)

        train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train, 
                                                   batch_size=batch_size, 
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val, 
                                                   batch_size=batch_size, 
                                                   shuffle=False)
    else:
        raise        
    return train_loader, val_loader

class thebeDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, ids, isDatasetNorm, isSourceDatasetNorm, isPretrain, modelName):
        self.imgs = imgs
        self.masks = masks
        self.ids = ids
        self.isDatasetNorm, self.isSourceDatasetNorm = isDatasetNorm, isSourceDatasetNorm
        self.isPretrain = isPretrain
        self.modelName = modelName


    def __len__(self):
        return len(self.ids)
    
    def transform(self, img, mask):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        
        # if isinlinenorm == False:          
            # scale to (0,1) dataset wise, using dataset min, max
    #         if isClip:
    #             if isExtremOutlier:  
    # #                 print("in clip eo")
    #                 img = np.clip(img, -0.1924, 0.1939) # to change
    #                 img = (img+0.1924)/(0.1939+0.1924)
    #             else:
    # #                 print("in clip")
    #                 img = np.clip(img, -0.1096, 0.1111) # to change
    #                 img = (img+0.1096)/(0.1096+0.1111)
            # else:
    #             print("in noclip")
        img = (img+8.3939)/(9.4332+8.3939)
            # normalize based on dataset mean std
        if self.isDatasetNorm:
    #             if isClip:
    #                 if isExtremOutlier:
    # #                     print("in clip eo dn")
    #                     img = TF.normalize(img, [0.4986,], [0.2230,])
    #                 else:
    # #                     print("in clip dn")
    #                     img = TF.normalize(img, [0.4985,], [0.2855,]) # to change
                # else:
    #                 print("in noclip dn")
            img = TF.normalize(img, [0.4708,], [0.0079,])
        
#         print(img.shape)
        if self.modelName == 'resnet101' or self.modelName == "deeplabv3_resnet101":
            imgrgb = torch.zeros(3,96,96).type(torch.cuda.FloatTensor)
            imgrgb[0] = img
            imgrgb[1] = img
            imgrgb[2] = img
        else:
            imgrgb = img
            
        if self.isSourceDatasetNorm:
            # if isPretrain == "cocoft" or isPretrain == "imagenetft":
            #     imgrgb = TF.normalize(imgrgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            if self.isPretrain == "faultSegft":
                if self.modelName == 'resnet101' or self.modelName == "deeplabv3_resnet101":
                    imgrgb = TF.normalize(imgrgb, [0.4915, 0.4915, 0.4915], [0.0655, 0.0655, 0.0655])
#                     imgrgb = TF.normalize(imgrgb, [0.5005, 0.5005, 0.5005], [0.1205, 0.1205, 0.1205]) # clipEO
                else:
                    imgrgb = TF.normalize(imgrgb, [0.4915,], [0.0655,])
#                     imgrgb = TF.normalize(imgrgb, [0.5005,], [0.1205,])
            elif 'f3' in self.isPretrain:
                imgrgb = TF.normalize(imgrgb, [0.5005611460825908,], [0.1051461597291201,])
            # elif isPretrain == "bsdsft":
            #     imgrgb = TF.normalize(imgrgb, [0.4316, 0.4357, 0.3637], [0.2521, 0.2360, 0.2446])
#                 if modelName == 'resnet101' or modelName == "deeplabv3_resnet101":
#                     imgrgb = TF.normalize(imgrgb, [0.4256, 0.4256, 0.4256], [0.2327, 0.2327, 0.2327])
#                 else:
#                     imgrgb = TF.normalize(imgrgb, [0.4256,], [0.2327,])  
            else:
                print("please enter a valid source dataset set")
        
        return imgrgb, mask.float()
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask = self.masks[idx]
        img = self.imgs[idx]
        
#         print(img.shape)

        img, mask = self.transform(img, mask)
    

        return (img, mask)


class f3Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, aug, isDatasetNorm, isClip, isExtremOutlier, modelName):
        self.imgs = imgs
        self.masks = masks
        self.ids = list(range(len(imgs)))
        self.aug = aug
        self.isClip, self.isExtremOutlier = isClip, isExtremOutlier
        self.isDatasetNorm = isDatasetNorm
        self.modelName = modelName

    def __len__(self):
        return len(self.ids)
    
    def transform(self, img, mask, aug):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        
        # # random crop
        # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(96, 96))
        # img = img[:,i:i+h,j:j+w]
        # mask = mask[:,i:i+h,j:j+w]

        if aug:
#             print("aug")
            if random.random() > 0.5:
                img = torch.flip(img,(2,)) 
                mask = torch.flip(mask,(2,))
#             if random.random() > 0.5:
#                 img = torch.flip(img,(1,)) 
#                 mask = torch.flip(mask,(1,))
#         else:
#             print("noaug")
        # scale to (0,1) dataset wise, using dataset min, max
#         if self.isClip:
#             if self.isExtremOutlier:  
# #                 print("in clip eo")
#                 img = np.clip(img, -4.3624, 4.3543) # to change
#                 img = (img+4.3624)/(4.3543+4.3624)
#             else:
# #                 print("in clip")
#                 img = np.clip(img, -2.4945, 2.4864) # to change
#                 img = (img+2.4945)/(2.4864+2.4945)
#         else:
# #             print("in noclip")
#             img = (img+7.892)/(8.168+7.892)
        img = (img+1.0)/(1.0 +1.0)
        # normalize based on dataset mean std
        if self.isDatasetNorm:
#             if self.isClip:
#                 if self.isExtremOutlier:
# #                     print("in clip eo dn")
#                     img = TF.normalize(img, [0.5005,], [0.1205,])
#                 else:
# #                     print("in clip dn")
#                     img = TF.normalize(img, [0.5008,], [0.2028,]) # to change
#             else:
# #                 print("in noclip dn")
#                 img = TF.normalize(img, [0.4915,], [0.0655,])
            img = TF.normalize(img, [0.5005611460825908,], [0.1051461597291201,])
    
        if self.modelName == 'resnet101' or self.modelName == "deeplabv3_resnet101":
            imgrgb = torch.zeros(3,96,96).type(torch.cuda.FloatTensor)
            imgrgb[0] = img
            imgrgb[1] = img
            imgrgb[2] = img
        else:
            imgrgb = img
            
        return imgrgb, mask
        
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask = self.masks[idx]
        img = self.imgs[idx]
        augment = self.aug
        # print(img.shape, mask.shape)
        # print(np.min(img), np.max(img), np.unique(mask))

        # axis0or1 = np.random.randint(2,size=1)
        # ithslice = np.random.randint(128,size=1)
        
#         if axis0or1 == 0:
# #             print("axis 0 ")
#             mask = mask[ithslice,:,:].squeeze().transpose()
#             img = img[ithslice,:,:].squeeze().transpose()
#         else:
# #             print("axis 1 ")
#             mask = mask[:,ithslice,:].squeeze().transpose()
#             img = img[:,ithslice,:].squeeze().transpose()

        
        img, mask = self.transform(img, mask, augment)

#         assert img.size() == mask.size(),             f'Image and mask {idx} should be the same size, but are {img.size()} and {mask.size()}'
        
        return (img, mask)


class faultSegDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, masks_dir, aug, isDatasetNorm, isClip, isExtremOutlier, modelName):
        self.images_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        self.aug = aug
        self.isClip, self.isExtremOutlier = isClip, isExtremOutlier
        self.isDatasetNorm = isDatasetNorm
        self.modelName = modelName
        
    def __len__(self):
        return len(self.ids)
    
    def transform(self, img, mask, aug):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        
        # random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(96, 96))
        img = img[:,i:i+h,j:j+w]
        mask = mask[:,i:i+h,j:j+w]
        
        if aug:
#             print("aug")
            if random.random() > 0.5:
                img = torch.flip(img,(2,)) 
                mask = torch.flip(mask,(2,))
#             if random.random() > 0.5:
#                 img = torch.flip(img,(1,)) 
#                 mask = torch.flip(mask,(1,))
#         else:
#             print("noaug")
        # scale to (0,1) dataset wise, using dataset min, max
        if self.isClip:
            if self.isExtremOutlier:  
#                 print("in clip eo")
                img = np.clip(img, -4.3624, 4.3543) # to change
                img = (img+4.3624)/(4.3543+4.3624)
            else:
#                 print("in clip")
                img = np.clip(img, -2.4945, 2.4864) # to change
                img = (img+2.4945)/(2.4864+2.4945)
        else:
#             print("in noclip")
            img = (img+7.892)/(8.168+7.892)
        # normalize based on dataset mean std
        if self.isDatasetNorm:
            if self.isClip:
                if self.isExtremOutlier:
#                     print("in clip eo dn")
                    img = TF.normalize(img, [0.5005,], [0.1205,])
                else:
#                     print("in clip dn")
                    img = TF.normalize(img, [0.5008,], [0.2028,]) # to change
            else:
#                 print("in noclip dn")
                img = TF.normalize(img, [0.4915,], [0.0655,])
    
        if self.modelName == 'resnet101' or self.modelName == "deeplabv3_resnet101":
            imgrgb = torch.zeros(3,96,96).type(torch.cuda.FloatTensor)
            imgrgb[0] = img
            imgrgb[1] = img
            imgrgb[2] = img
        else:
            imgrgb = img
            
        return imgrgb, mask
        
    
    def __getitem__(self, i):
        idx = self.ids[i]
        mask = np.fromfile("{}/{}.dat".format(self.masks_dir,idx),dtype=np.single).reshape(128,128,128)
        img = np.fromfile("{}/{}.dat".format(self.images_dir,idx),dtype=np.single).reshape(128,128,128)
        augment = self.aug
        
        axis0or1 = np.random.randint(2,size=1)
        ithslice = np.random.randint(128,size=1)
        
        if axis0or1 == 0:
#             print("axis 0 ")
            mask = mask[ithslice,:,:].squeeze().transpose()
            img = img[ithslice,:,:].squeeze().transpose()
        else:
#             print("axis 1 ")
            mask = mask[:,ithslice,:].squeeze().transpose()
            img = img[:,ithslice,:].squeeze().transpose()

        
        img, mask = self.transform(img, mask, augment)

#         assert img.size() == mask.size(),             f'Image and mask {idx} should be the same size, but are {img.size()} and {mask.size()}'
        
        return (img, mask)


# In[6]:

# pytorch version of the loss:
def dice_loss_with_empty_handling(y_true, y_pred, epsilon=1e-6):
    """
    Dice loss with handling for samples with no positive pixels.
    Args:
        y_true: Tensor of ground truth labels (binary: 0 or 1).
        y_pred: Tensor of predicted probabilities (requires_grad=True).
        epsilon: Small constant for numerical stability.
    Returns:
        Dice loss value.
    """
    y_true = y_true.float()
    y_pred = y_pred.float()
    # Flatten tensors
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    # Compute intersection and denominator
    intersection = torch.sum(y_true_flat * y_pred_flat)
    denominator = torch.sum(y_true_flat) + torch.sum(y_pred_flat)
    # Ensure denominator does not cause division by zero
    dice_coeff = (2. * intersection + epsilon) / (denominator + epsilon)
    # Ensure loss remains part of computational graph
    dice_loss_value = 1.0 - dice_coeff
    return dice_loss_value  # This remains differentiable


def main():
    parser = argparse.ArgumentParser(description='pre-training')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument("--modelName", default='', type=str)
    parser.add_argument('--isPretrain', default="none", type=str)
    parser.add_argument('--lr', default= 1e-3, type=float)
    parser.add_argument('--lr_patience', default = 3, type=int)
    parser.add_argument('--patience', default = 10, type=int)
    parser.add_argument('--epoches', default = 100, type=int)
    parser.add_argument('--targetData', default='faultSeg', type=str)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--batch_size', default = 32, type=int)
    parser.add_argument('--isinlinenorm', default=False, type=bool) 
    parser.add_argument('--isClip', default  = False, type=bool)
    parser.add_argument('--isExtremOutlier', default  = False, type=bool)
    parser.add_argument('--train_iterations', default = 100, type=int)
    parser.add_argument('--best_iou_threshold', default=0.5,type=float)
    parser.add_argument('--isportion', default = "none", type=str) #["none", "10per", "1per", "1per12", "1pic"]
    parser.add_argument('--val_iterations', default = 20, type=int)
    parser.add_argument('--label_source', default = '', type=str)
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    modelName = args.modelName #'deeplabv3_resnet101' #["deeplabv3_resnet101","resnet101", "unet", "deeplab", "hed", "rcf"]
    isPretrain = args.isPretrain
    lr = args.lr
    lr_patience = args.lr_patience
    patience = args.patience
    epoches = args.epoches
    targetData = args.targetData
    data_folder = args.data_folder
    batch_size = args.batch_size
    isClip = args.isClip
    isExtremOutlier = args.isExtremOutlier
    isinlinenorm=args.isinlinenorm
    train_iterations = args.train_iterations
    val_iterations = args.val_iterations
    best_iou_threshold = args.best_iou_threshold
    isportion = args.isportion
    label_source = args.label_source
    if isinlinenorm:
        isDatasetNorm = False
        isSourceDatasetNorm = False
    elif isPretrain == "none":  
        isDatasetNorm = True
        isSourceDatasetNorm = False
    else:
        isDatasetNorm = False
        isSourceDatasetNorm = True

    best_model_fpath = modelName + "_{}_{}_{}_{}".format(targetData,lr_patience,patience,lr)
    if targetData != 'bsds':
        if isClip:
            if isExtremOutlier:
                best_model_fpath = best_model_fpath + "_clipEO"
            else:
                best_model_fpath = best_model_fpath + "_clip"
    if isDatasetNorm:
        best_model_fpath = best_model_fpath + "_DN"
    if isPretrain == "cocoft":
        best_model_fpath = best_model_fpath + "_cocoft"
        if isfeature_extract:
            best_model_fpath = best_model_fpath + "_fe"
    elif isPretrain == "imagenetft":
        best_model_fpath = best_model_fpath + "_imagenetft"
    elif isPretrain == "faultSegft":
        best_model_fpath = best_model_fpath + "_faultSegft"
    elif isPretrain == "bsdsft":
        best_model_fpath = best_model_fpath + "_bsdsft"
    if isPretrain != "none" and isSourceDatasetNorm:
        best_model_fpath = best_model_fpath + "_SDN"

    if isinlinenorm:
        best_model_fpath = best_model_fpath + "_ilnorm"
    if isportion != "none":
        best_model_fpath = best_model_fpath + "_" + isportion
        
    best_model_root = f'/mnt/HD_18T_pt1/chen/FaultBenchmark/pretrain/{targetData}/{label_source}'
    os.makedirs(best_model_root, exist_ok=True)
    best_model_fpath = os.path.join(best_model_root, best_model_fpath + "_{}_{}_crop96_b{}_new".format(train_iterations, val_iterations, batch_size)) #_trainaugvalnohvflip
    print(best_model_fpath)
    
    model = get_model(modelName, isPretrain)
    
    if isPretrain != "none" and isPretrain != "cocoft" and isPretrain != "imagenetft":
        print("load pretrained weights")
        model.load_state_dict(torch.load(pretrain_model_fpath, map_location="cuda:1"))
    # Send the model to GPU
    model = model.double()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=lr_patience, verbose=True)

    t_start_train = time.time()
    bceloss = dice_loss_with_empty_handling#nn.BCELoss()
    bcelogloss = nn.BCEWithLogitsLoss()
    celoss = nn.CrossEntropyLoss()
    mean_train_losses = []
    mean_val_losses = []
    mean_train_accuracies = []
    mean_val_accuracies = []
    t_start = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0)

    if targetData == 'thebe':
        data_path = data_folder # "{}/processedThebe".format(data_folder)
    elif targetData == 'bsds':
        # data_folder = "/data/anyu/BSDS500/data"
        data_path = "{}".format(data_folder)
    elif targetData == 'faultSeg':
        # data_folder = args.data_folder
        data_path = "{}".format(data_folder)
    elif targetData == 'f3':
        data_path = "{}".format(data_folder)
    train_loader, val_loader =   get_dataloaders(targetData, data_path, label_source, batch_size, isDatasetNorm, isSourceDatasetNorm, isClip, isExtremOutlier, isPretrain, modelName)
    time1 = time.time()
    # Train
    for epoch in range(epoches):                  
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        labelled_val_accuracies = []

        model.train()
        train_iter = iter(train_loader)
        for ith_iter in range (train_iterations):
            if targetData == 'faultSeg' or targetData == "bsds" or isportion == "1per": 
                train_iter = iter(train_loader)
            images, masks = next(train_iter)

            images = Variable(images.double().to(device=device))
            masks = Variable(masks.double().to(device=device))

            outputs = model(images)

            loss = torch.zeros(1).to(device)
            y_preds = outputs
            if modelName == "unet" or modelName == "deeplab":
                loss = bceloss(outputs, masks) 
    #             loss = cross_entropy_loss_HED(outputs, masks) 
            elif modelName == "hed":
                for o in range(5):
                    loss = loss + cross_entropy_loss_HED(outputs[o], masks)
                loss = loss + bceloss(outputs[-1],masks)
                y_preds = outputs[-1]
            elif modelName == "rcf":
                for o in outputs:
                    loss = loss + cross_entropy_loss_RCF(o, masks)
                y_preds = outputs[-1]
            elif modelName == "resnet101" or modelName == "deeplabv3_resnet101":
                classifier_outputs = outputs["out"] 
                loss = bcelogloss(classifier_outputs, masks) 
    #             classifier_outputs, aux_outputs = outputs["out"], outputs["aux"]      
    #             loss = bcelogloss(classifier_outputs, masks) + 0.4*bcelogloss(aux_outputs, masks)
                y_preds = torch.sigmoid(classifier_outputs)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.data)
            predicted_mask = y_preds > best_iou_threshold
            train_acc = iou_pytorch(predicted_mask.squeeze(1).byte(), masks.squeeze(1).byte())
            train_accuracies.append(train_acc.mean()) 
            if ith_iter%10 == 0:
                print("Epoch {}:[{}/{}]".format(epoch,ith_iter,train_iterations), "Acc: {:.4f}".format(train_acc.mean().item()), "Loss: {:.4f}".format(loss.item()))

        model.eval()
        val_iter = iter(val_loader)
        for ith_iter in range (val_iterations):
            if targetData == 'faultSeg' or targetData == "bsds" or isportion == "1per": 
                val_iter = iter(val_loader)
            images, masks = next(val_iter)
            images = Variable(images.double().to(device=device))
            masks = Variable(masks.double().to(device=device))

            outputs = model(images)

            loss = torch.zeros(1).to(device)
            y_preds = outputs
            if modelName == "unet" or modelName == "deeplab":
                loss = bceloss(outputs, masks) 
    #             loss = cross_entropy_loss_HED(outputs, masks) 
            elif modelName == "hed":
                for o in range(5):
                    loss = loss + cross_entropy_loss_HED(outputs[o], masks)
                loss = loss + bceloss(outputs[-1],masks)
                y_preds = outputs[-1]
            elif modelName == "rcf":
                for o in outputs:
                    loss = loss + cross_entropy_loss_RCF(o, masks)
                y_preds = outputs[-1]
            elif modelName == "resnet101" or modelName == "deeplabv3_resnet101":
                classifier_outputs = outputs["out"] 
                loss = bcelogloss(classifier_outputs, masks) 
                y_preds = torch.sigmoid(classifier_outputs)


            val_losses.append(loss.data)
            predicted_mask = y_preds > best_iou_threshold
            val_acc = iou_pytorch(predicted_mask.byte(), masks.squeeze(1).byte())
            val_accuracies.append(val_acc.mean())


        mean_train_losses.append(torch.mean(torch.stack(train_losses)))
        mean_val_losses.append(torch.mean(torch.stack(val_losses)))
        mean_train_accuracies.append(torch.mean(torch.stack(train_accuracies)))
        mean_val_accuracies.append(torch.mean(torch.stack(val_accuracies)))

        scheduler.step(torch.mean(torch.stack(val_losses)))    
        early_stopping(torch.mean(torch.stack(val_losses)), model, best_model_fpath)



        if early_stopping.early_stop:
            print("Early stopping")
            break


        for param_group in optimizer.param_groups:
            learningRate = param_group['lr']


        # Print Epoch results
        t_end = time.time()

        print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Time: {}. LR: {}'
              .format(epoch+1, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(val_losses)), torch.mean(torch.stack(train_accuracies)), torch.mean(torch.stack(val_accuracies)), t_end-t_start, learningRate))

        t_start = time.time()

    total_training_time = time.time()-t_start_train
    print("total training time: {:.4f}".format(total_training_time/3600))


    # In[ ]:


    mean_train_losses = np.asarray(torch.stack(mean_train_losses).cpu())
    mean_val_losses = np.asarray(torch.stack(mean_val_losses).cpu())
    mean_train_accuracies = np.asarray(torch.stack(mean_train_accuracies).cpu())
    mean_val_accuracies = np.asarray(torch.stack(mean_val_accuracies).cpu())

    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    train_loss_series = pd.Series(mean_train_losses)
    val_loss_series = pd.Series(mean_val_losses)
    train_loss_series.plot(label="train_loss")
    val_loss_series.plot(label="validation_loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    train_acc_series = pd.Series(mean_train_accuracies)
    val_acc_series = pd.Series(mean_val_accuracies)
    train_acc_series.plot(label="train_acc")
    val_acc_series.plot(label="validation_acc")
    plt.legend()
    plt.title('{}_loss_acc'.format(best_model_fpath))
    plt.savefig('{}_loss_acc.png'.format(best_model_fpath))

    totaltime = time.time()-time1
    print("total cost {} hours".format(totaltime/3600))


if __name__ == '__main__':
    main()
