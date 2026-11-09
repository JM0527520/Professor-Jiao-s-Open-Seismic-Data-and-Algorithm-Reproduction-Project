### prediction
from functions import *
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
# import cmapy
import argparse
import numpy as np
from os.path import splitext
from os import listdir


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1 is visible to this code


# in order to get reproducable results
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

parser = argparse.ArgumentParser(description='prediction')
parser.add_argument('--subset', default='test', type=str)
parser.add_argument('--modelname', default='', type=str)
parser.add_argument('--label_source', default='', type=str)
parser.add_argument('--target_data', default='', type=str)
parser.add_argument('--save_root', default='', type=str)
args = parser.parse_args()

subset = args.subset

isval = False

# seis_path = ''
# # fault_path = ''
# if isval:
#     seis_path = ''
# #     fault_path = ''
# #print(seis_path)


# Make input is IL, Z, XL order.

im_height = 128
im_width = 128
splitsize = 96
stepsize = 48 #overlap half
overlapsize = splitsize-stepsize


horizontal_splits_number = int(np.ceil((im_width)/stepsize))
#print(horizontal_splits_number)
width_after_pad = stepsize*horizontal_splits_number+2*overlapsize
#print(width_after_pad)
left_pad = int((width_after_pad-im_width)/2)
right_pad = width_after_pad-im_width-left_pad
#print(left_pad,right_pad)

vertical_splits_number = int(np.ceil((im_height)/stepsize))
#print(vertical_splits_number)
height_after_pad = stepsize*vertical_splits_number+2*overlapsize
#print(height_after_pad)
top_pad = int((height_after_pad-im_height)/2)
bottom_pad = height_after_pad-im_height-top_pad
#print(top_pad,bottom_pad)


horizontal_splits_number = horizontal_splits_number+1
#print(horizontal_splits_number)
vertical_splits_number = vertical_splits_number+1
#print(vertical_splits_number)

halfoverlapsize = int(overlapsize/2)
#print(halfoverlapsize)

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
# i_cnt = 0
# Iterate xx volumes
for idx in (ids):
    # #print("{}/{}.dat".format(fault_path,idx))
    # iterate current cube
    label_volume = np.fromfile("{}/{}.dat".format(fault_path,idx),dtype=np.single).reshape(128,128,128)
    image_volume = np.fromfile("{}/{}.dat".format(seis_path,idx),dtype=np.single).reshape(128,128,128)
    assert image_volume.shape==label_volume.shape
    # #print(idx,  image_volume.shape, )
    # iterate all hundered slcies
    for ithslice in (range(len(image_volume))):
        image_list1.extend(split_Image(image_volume[ithslice,:,:].squeeze().transpose(),True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
        mask_list1.extend(split_Image(label_volume[ithslice,:,:].squeeze().transpose(),True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
        image_list2.extend(split_Image(image_volume[:,ithslice,:].squeeze().transpose(),True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
        mask_list2.extend(split_Image(label_volume[:,ithslice,:].squeeze().transpose(),True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))


    
assert len(image_list1) == len(mask_list1) == len(image_list2) == len(mask_list2)
#print(len(image_list1) , len(mask_list1))
#print(image_list1[0].shape)
# i_cnt += len(X_list)
#print("read {} images and labels in {} sec".format(len(image_list1), time.time()-t_start))

# #print(len(X_list))
# #print(X_list[0].shape)
# #print("read images in {} sec".format(time.time()-t_start))
# X = np.asarray(X_list)
# #print(X.shape)
# #print("read images in {} sec".format(time.time()-t_start))

# X = X.astype(np.float32)
# # X = np.expand_dims(X,1)
# #print(X.shape)

label_source = args.label_source #''#'novice15'
target_data = args.target_data #'thebe'

best_model_root = os.path.join(args.save_root, target_data, label_source)

modelname = args.modelname #'deeplab_faultSeg_3_10_0.001_DN_100_20_crop96_b32_new' #'deeplab_f3_3_10_0.001_DN_100_20_crop96_b32_new' #'rcf_f3_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new' #'hed_f3_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new' #'unet_f3_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new' #'deeplab_f3_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new' #
if "deeplabv3_resnet101" in modelname: 
    from torchvision import models
    if "cocoft" in modelname:
        #print("coco")
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=21,aux_loss=True, pretrained_backbone=False)
        num_ftrs = model.classifier[4].in_channels
        model.classifier[4] = nn.Conv2d(num_ftrs, 1, kernel_size=(1, 1), stride=(1, 1))
#         #print(model)
    else:
        #print("other")
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=False, pretrained_backbone=False)
#     model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,aux_loss=True)
    #print("use torchvision deeplabv3_resnet101")
elif "resnet101" in modelname: 
    from torchvision import models
    model = models.segmentation.fcn_resnet101(pretrained=False, num_classes=1,aux_loss=True)
    #print("use torchvision resnet101")   
elif "unet" in modelname: 
    from model_zoo.UNET import Unet
    model = Unet()
    #print("use model Unet")
elif "deeplab" in modelname: 
    from model_zoo.DEEPLAB.deeplab import DeepLab
    model = DeepLab(backbone='mobilenet', num_classes=1, output_stride=16)
    #print("use model DeepLab")
elif "hed" in modelname: 
    from model_zoo.HED import HED
    model = HED()
    #print("use model HED") 
elif "rcf" in modelname: 
    from model_zoo.RCF import RCF
    model = RCF()
    #print("use model RCF")
else:
    raise#print("please enter a valid model")
    
best_model_fpath = os.path.join(best_model_root, modelname) #+ ".model"
model.load_state_dict(torch.load(best_model_fpath, map_location="cuda:1")) 
device = torch.device("cuda:1" if torch.cuda.is_available() else "CPU")
#print('best_model_fpath:', best_model_fpath)
# import pdb;pdb.set_trace()
if 'hed' in modelname or "rcf" in modelname: 
    model = model.double()
# Send the model to GPU
model = model.to(device)
save_path = os.path.join(best_model_root, 'overlap_clean_predictions/{}_smooth/faultseg/{}'.format(modelname, subset))
if isval:
    save_path = save_path + "_val"
#print(save_path)


class faultSegDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, modelName, isDatasetNorm=True):
        # self.images_dir = imgs_dir
        # self.masks_dir = masks_dir
        # self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        # self.aug = aug
        self.isDatasetNorm = isDatasetNorm
        self.ids = list(range(len(imgs)))
        self.imgs = imgs
        self.modelName = modelName

    def __len__(self):
        return len(self.ids)
    
    def transform(self, img):
        # to tensor
        img = TF.to_tensor(img)
        # mask = TF.to_tensor(mask)

        # scale to (0,1) dataset wise, using dataset min, max
        if "_clip" in self.modelName:
            raise
        #             #print("in noclip")
        else:
            img = (img+7.892)/(8.168+7.892)
        # normalize based on dataset mean std
        if "_DN" in self.modelName:
            # #print("in noclip dn")
            img = TF.normalize(img, [0.4915,], [0.0655,])
    
        if self.modelName == 'resnet101' or self.modelName == "deeplabv3_resnet101":
            imgrgb = torch.zeros(3,96,96).type(torch.cuda.FloatTensor)
            imgrgb[0] = img
            imgrgb[1] = img
            imgrgb[2] = img
        else:
            imgrgb = img
            
        if "_SDN" in self.modelName:
            if "cocoft" in self.modelName:
                imgrgb = TF.normalize(imgrgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            elif "imagenet" in self.modelName:
                imgrgb = TF.normalize(imgrgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            elif "faultSegft" in self.modelName:
                if "resnet101" in self.modelName: 
                    imgrgb = TF.normalize(imgrgb, [0.5005, 0.5005, 0.5005], [0.1205, 0.1205, 0.1205])
                else:
                    imgrgb = TF.normalize(imgrgb, [0.4915,], [0.0655,])
            elif "f3ft" in self.modelName:
                if "resnet101" in self.modelName: 
                    imgrgb = TF.normalize(imgrgb, [0.4256, 0.4256, 0.4256], [0.2327, 0.2327, 0.2327])
                else:
                    imgrgb = TF.normalize(imgrgb, [0.5005611460825908,], [0.1051461597291201,])  
            elif 'thebeft' in self.modelName:
                imgrgb = TF.normalize(imgrgb, [0.4708,], [0.0079,])                    
            else:
                raise#print("please enter a valid source dataset set")
                                    
        return imgrgb
        
    
    def __getitem__(self, i):
        idx = self.ids[i]

        img = self.imgs[idx]
        img = self.transform(img)

#         assert img.size() == mask.size(),             f'Image and mask {idx} should be the same size, but are {img.size()} and {mask.size()}'
        
        return img


# data_path = '/mnt/HD_18T_pt1/chen/Datasets/FaultSeg3D'    
faults_dataset_test1  = faultSegDataset(image_list1, modelname)
faults_dataset_test2  = faultSegDataset(image_list2, modelname)
batch_size = 32

test_loader1 = torch.utils.data.DataLoader(dataset=faults_dataset_test1, 
                                           batch_size=batch_size, 
                                           shuffle=False)    

test_loader2 = torch.utils.data.DataLoader(dataset=faults_dataset_test2, 
                                           batch_size=batch_size, 
                                           shuffle=False)    


    
def saveResults(save_path, axis, test_loader):     
    os.makedirs(save_path, exist_ok=True)

    test_predictions = []
    imageNo = -1 #+len(image_volume)-30
    model.eval()
    for images in test_loader:
#         images = Variable(images.double().to(device=device))
        images = Variable(images.to(device=device, dtype=torch.float))
        if 'hed' in modelname or "rcf" in modelname:
            images = images.double()
        outputs = model(images)
        y_preds = outputs
        if "hed" in modelname: 
            y_preds = outputs[-1]
        elif "rcf" in modelname: 
            y_preds = outputs[-1]
        elif "resnet101" in modelname: 
            classfer_outputs = outputs["out"]
            y_preds = torch.sigmoid(classfer_outputs)
#             outputs = model(images)["out"]
#             y_preds = torch.sigmoid(outputs)
                

        test_predictions.extend(y_preds.detach().cpu())
        if len(test_predictions)>=vertical_splits_number*horizontal_splits_number:
            imageNo = imageNo +1
            tosave = torch.stack(test_predictions).detach().cpu().numpy()[0:vertical_splits_number*horizontal_splits_number]
            test_predictions = test_predictions[vertical_splits_number*horizontal_splits_number:]

            WINDOW_SPLINE_2D = window_2D(window_size=splitsize, power=2)
            tosave = np.moveaxis(tosave,-3,-1)
            tosave = np.array([patch * WINDOW_SPLINE_2D for patch in tosave])
            tosave = tosave.reshape((vertical_splits_number, horizontal_splits_number, splitsize,splitsize,1))
            recover_Y_test_pred = recover_Image2(tosave, (im_height,im_width,1), left_pad,right_pad,top_pad,bottom_pad,overlapsize)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path,"{}_{}".format(imageNo,axis)),np.squeeze(recover_Y_test_pred))   
            #print(imageNo)
                        

#print("save")
t1 = time.time()
saveResults(save_path, 'axis_0', test_loader1)
saveResults(save_path, 'axis_1', test_loader2)
t2 = time.time()
#print('save in {} sec'.format(t2-t1))

#print(save_path)

# a = np.load(save_path + "/0.npy")
# #print(a.dtype, a.max(), a.min())


# heatmap_img = cv2.applyColorMap((a*255).astype(np.uint8), cmapy.cmap('jet_r'))
# plt.figure(figsize=(20,16))
# plt.imshow(heatmap_img)
# # plt.colorbar(shrink=0.5)
# plt.axis('off')
# plt.savefig(save_path + "/0.png", dpi=300)
