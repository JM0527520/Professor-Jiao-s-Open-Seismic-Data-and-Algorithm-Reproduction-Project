import torch.distributed as dist

from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, SegformerConfig
from torchvision.models.segmentation import deeplabv3_resnet50
from skimage.metrics import hausdorff_distance
from medpy.metric.binary import dc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
torch.cuda.empty_cache()


import segmentation_models_pytorch as smp
from torchvision import transforms

import os
from PIL import Image
import numpy as np
from torch.nn import functional as F
import random
from utils import yaml_config_hook

from datasets import ThebeDataset, CRACKSDataset, SYNTHDataset


stats = {'thebe': [3.985043, -4.957859],
         'cv_model': [1.0, -1.0],
         'synth': [8.168264,  -7.8922825]}
class SegformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SegformerModel, self).__init__()
        # Load pretrained model
        config = SegformerConfig(num_labels=num_classes)
        self.model = SegformerForSemanticSegmentation(config)
        self.model.decode_head.classifier = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

        layers_to_unfreeze = [
            "encoder.block.3",
            "encoder.layer_norm",
            "decode_head"
        ]

        for name, param in self.model.named_parameters():
            param.requires_grad = True


    def forward(self, x):
        return self.model(x).logits

def main(args):
    model = torch.load(args.model_path)
    model.to(args.device)

    files = os.listdir(args.data_path + 'images/')

    rot_keys = ['cv_model', 'synth']
    rotate = False
    for w in rot_keys:
        if w in args.data_path:
            rotate = True

    for key in stats.keys():
        if key in args.data_path:
            data_max, data_min = stats[key]


    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),  # Unet only
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:1, :, :])
    ])

    for f in tqdm(files):
        image = np.load(os.path.join(args.data_path, 'images' ,f))
        label = np.load(os.path.join(args.data_path, 'labels' ,f))

        image = (image - data_min) / (data_max - data_min)

        if rotate:
            image, label = image.T, label.T

        image, label = image_transform(image).to(args.device).float(), mask_transform(label).to(args.device)

        outputs = model(image[None,...])
        #outputs = torch.sigmoid(outputs)
        outputs = torch.softmax(outputs, dim=1)[:, 1, :, :] #segformer
        outputs = F.interpolate(outputs[None,...], size=label.shape[-2:], mode="bilinear", align_corners=False)
        outputs = outputs.cpu().detach().numpy().squeeze()


        if rotate:
            outputs = outputs.T

        np.save(os.path.join(args.save_path,  f), outputs.astype(np.float16))











if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train-Thebe")
    config = yaml_config_hook("config/inference.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()


    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    main(args)