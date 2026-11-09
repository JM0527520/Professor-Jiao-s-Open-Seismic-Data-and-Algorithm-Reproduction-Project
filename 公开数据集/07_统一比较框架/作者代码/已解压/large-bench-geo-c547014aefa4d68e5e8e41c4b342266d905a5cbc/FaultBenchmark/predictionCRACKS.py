### prediction
from functions import *
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
# import cmapy
import argparse
import numpy as np

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
seis_path = '/mnt/HD_18T_pt1/chen/Datasets/CRACKS/image_volume.npy'
# fault_path = ''
if isval:
    seis_path = ''
#     fault_path = ''
#print(seis_path)

image_volume = np.load(seis_path)
# Make input is IL, Z, XL order.
im_height = image_volume.shape[-2:][1]#1537
im_width = image_volume.shape[-2:][0] #3174
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


t_start = time.time()
X_list = []
sectionNumber = 30
# if isval:
#     sectionNumber = 40

for i in {"train": range(sectionNumber, 400-sectionNumber), "test1": range(0,sectionNumber,1), "test2": range(-sectionNumber, 0)}[subset]: #len(seis)
# for i in :
    # #print("{}.npy".format(i)) 
    img = image_volume[i].T
    X_list.extend(split_Image(img,True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
    
#print(len(X_list))
#print(X_list[0].shape)
#print("read images in {} sec".format(time.time()-t_start))
X = np.asarray(X_list)
#print(X.shape)
#print("read images in {} sec".format(time.time()-t_start))

X = X.astype(np.float32)
# X = np.expand_dims(X,1)
#print(X.shape)

label_source = args.label_source #'novice15'
target_data = args.target_data #'f3'
best_model_root = os.path.join(args.save_root, target_data, label_source) #f'/mnt/HD_18T_pt1/chen/Datasets/chen/FaultBenchmark/pretrain/{target_data}/{label_source}'

modelname = args.modelname #'deeplab_f3_3_10_0.001_DN_100_20_crop96_b32_new' #'rcf_f3_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new' #'hed_f3_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new' #'unet_f3_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new' #'deeplab_f3_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new' #
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
save_path = os.path.join(best_model_root, 'overlap_clean_predictions/{}_smooth/cracks/{}'.format(modelname, subset))
if isval:
    save_path = save_path + "_val"
#print(save_path)


class f3Dataset_faultSegPretrain(torch.utils.data.Dataset):
    # def __init__(self, imgs_dir, masks_dir, aug, isDatasetNorm, isClip, isExtremOutlier, modelName):
    def __init__(self, imgs, modelName):
        # self.images_dir = imgs_dir
        # self.masks_dir = masks_dir
        self.ids = list(range(len(imgs)))
        self.imgs = imgs
        self.modelName = modelName
        
    def __len__(self):
        return len(self.ids)
    
    def transform(self, img):
        # scale to (0,1) dataset wise, using dataset min, max
        if "_clip" in self.modelName:
            raise
#             #print("in noclip")
        else:
            img = (img+1.0)/(1.0 +1.0)

        img = TF.to_tensor(img) #
        # normalize based on dataset mean std
        if "_DN" in self.modelName:
#                 #print("in noclip dn")
            img = TF.normalize(img, [0.5005611460825908,], [0.1051461597291201,])
    
        if 'resnet101' in self.modelName  or "deeplabv3_resnet101" in self.modelName:
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
    
    
faults_dataset_test = f3Dataset_faultSegPretrain(X, modelname)

batch_size = 32

test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test, 
                                           batch_size=batch_size, 
                                           shuffle=False)    



    
def saveResults(save_path, test_loader):     
    os.makedirs(save_path, exist_ok=True)

    test_predictions = []
    imageNo = {"test1": -1, "test2": -1+len(image_volume)-sectionNumber, "train": -1+sectionNumber}[subset] #
    imageNo+=1
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
            np.save(os.path.join(save_path,f"section_{(imageNo):03}"),np.squeeze(recover_Y_test_pred))   
            #print(imageNo)
                        

#print("save")
t1 = time.time()
saveResults(save_path, test_loader)
t2 = time.time()
#print('save in {} sec'.format(t2-t1))

import os
import shutil

def copy_files_by_type(src_folder, dst_folder, file_extension):
    os.makedirs(dst_folder, exist_ok=True)

    for file_name in os.listdir(src_folder):
        if file_name.endswith(file_extension):
            src_file = os.path.join(src_folder, file_name)
            dst_file = os.path.join(dst_folder, file_name)
            shutil.copy2(src_file, dst_file)
            # #print(f"Copied: {src_file} -> {dst_file}")

# Example: Copy only .txt files
if 'test' in subset:
    copy_files_by_type(save_path, save_path.replace(subset, 'test'), ".npy")