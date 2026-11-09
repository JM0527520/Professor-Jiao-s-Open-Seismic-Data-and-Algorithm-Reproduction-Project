from functions import *
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
# import cmapy
import argparse

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
seis_path = f'/mnt/HD_18T_pt1/chen/Datasets/Thebe/{subset}/seis/memmap/images_memmap1.dat'
i_vol = 1
# fault_path = ''
if isval:
    seis_path = None 
#     fault_path = ''
#print(seis_path)


im_height = 1537
im_width = 3174
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
# X_list = []
# sectionNumber = 141
# if isval:
#     sectionNumber = 40
# for i in range(0,sectionNumber,1): #len(seis)
#     file_path = seis_path + "{}.npy".format(i)
#     img = np.load(file_path)
#     X_list.extend(split_Image(img,True,top_pad,bottom_pad,left_pad,right_pad,splitsize,stepsize,vertical_splits_number,horizontal_splits_number))
full_shape = (231200, 96, 96)
X_list = np.memmap(seis_path, dtype=np.float64, mode='r', shape=full_shape)
#print(len(X_list))
#print(X_list[0].shape)
#print("read images in {} sec".format(time.time()-t_start))
#print(X_list[1000].max(), X_list[1000].min(),)
#print("read images in {} sec".format(time.time()-t_start))

if subset=='train': X_list = X_list[:len(X_list)//10]

label_source = args.label_source #''#'novice15'
target_data = args.target_data #'thebe'

best_model_root = os.path.join(args.save_root, target_data, label_source)
# best_model_root = os.path.join(f'/mnt/data-tmp/chen/FaultBenchmark/finetune/{targetData}', f'{label_source}')
 # best_model_root = f'/mnt/HD_18T_pt1/chen/FaultBenchmark/pretrain/{targetData}/{label_source}'

modelname = args.modelname #'unet_thebe_3_10_0.001_f3ft_novice15_SDN_100_20_crop96_b32_new' #'unet_thebe_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new' #'deeplab_thebe_3_10_0.001_f3ft_novice15_SDN_100_20_crop96_b32_new' #'deeplab_thebe_3_10_0.001_faultSegft_SDN_100_20_crop96_b32_new'
# import pdb;pdb.set_trace()
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
    
best_model_fpath = os.path.join(best_model_root,  modelname) #+ ".model"
model.load_state_dict(torch.load(best_model_fpath, map_location="cuda:1")) 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#print('best_model_fpath:', best_model_fpath)
if 'hed' in modelname or "rcf" in modelname: 
    model = model.double()
# Send the model to GPU
model = model.to(device)
save_path = os.path.join(best_model_root, 'overlap_clean_predictions/{}_smooth/thebe/{}'.format(modelname, subset))
if isval:
    save_path = save_path + "_val"
#print(save_path)


import torch.utils.data
class faultsDataset(torch.utils.data.Dataset):

    def __init__(self,preprocessed_images, name):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """

        self.images = preprocessed_images
        self.name = name


    def __len__(self):
        return len(self.images)
    
    def transform(self, img):
        # to tensor
        img = TF.to_tensor(img)  
        
        # scale to (0,1) dataset wise, using dataset min, max
        if "_clip" in self.name:
            if "_clipEO" in self.name: 
#                 #print("in clip eo")
                img = np.clip(img, -0.1924, 0.1939) # to change
                img = (img+0.1924)/(0.1939+0.1924)
            else:
#                 #print("in clip")
                img = np.clip(img, -0.1111, 0.1096) # to change
                img = (img+0.1111)/(0.1096+0.1111)
        else:
            img = (img+8.3939)/(9.4332+8.3939)
        
        # normalize based on dataset mean std
        if "_DN" in self.name:
            if "_clip" in self.name:
                if "_clipEO" in self.name:  
#                     #print("clip eo dn")
                    img = TF.normalize(img, [0.4986,], [0.2230,])
                else:
#                     #print("clip dn")
                    img = TF.normalize(img, [0.4985,], [0.2855,]) # to change
            else:
#                 #print("dn")
                img = TF.normalize(img, [0.4708,], [0.0079,])
        
#         #print(img.shape)
        if "resnet101" in self.name: 
            imgrgb = torch.zeros(3,96,96).type(torch.cuda.FloatTensor)
            imgrgb[0] = img
            imgrgb[1] = img
            imgrgb[2] = img
        else:
            imgrgb = img
            
        if "_SDN" in self.name:
            if "cocoft" in self.name:
                imgrgb = TF.normalize(imgrgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            elif "imagenet" in self.name:
                imgrgb = TF.normalize(imgrgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            elif "faultSegft" in self.name:
                if "resnet101" in self.name: 
                    imgrgb = TF.normalize(imgrgb, [0.5005, 0.5005, 0.5005], [0.1205, 0.1205, 0.1205])
                else:
                    imgrgb = TF.normalize(imgrgb, [0.4915,], [0.0655,])
            elif "f3ft" in self.name:
                if "resnet101" in self.name: 
                    imgrgb = TF.normalize(imgrgb, [0.4256, 0.4256, 0.4256], [0.2327, 0.2327, 0.2327])
                else:
                    imgrgb = TF.normalize(imgrgb, [0.5005611460825908,], [0.1051461597291201,])  
            elif 'thebeft' in self.name:
                imgrgb = TF.normalize(imgrgb, [0.4708,], [0.0079,])                                        
            else:
                raise#print("please enter a valid source dataset set")
            
        return imgrgb
    

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        return image


faults_dataset_test = faultsDataset(X_list, modelname)

batch_size = 32

test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test, 
                                           batch_size=batch_size, 
                                           shuffle=False)


# UPPER_BOUND = 800
# LOWER_BOUND = 1300
# #print("UPPER_BOUND",UPPER_BOUND)
# #print("LOWER_BOUND",LOWER_BOUND)
    
def saveResults(save_path, test_loader):     
    os.makedirs(save_path, exist_ok=True)

    test_predictions = []
    imageNo = -1
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
            np.save(os.path.join(save_path,"vol_{}_slice_{}".format(i_vol,imageNo)),np.squeeze(recover_Y_test_pred)) #[UPPER_BOUND:LOWER_BOUND,:])   
            #print(imageNo)
                        

#print("save")
t1 = time.time()
saveResults(save_path, test_loader)
t2 = time.time()
#print('save in {} sec'.format(t2-t1))

# (231200, 96, 96)
# (231200, 96, 96)
# (231200, 96, 96)
