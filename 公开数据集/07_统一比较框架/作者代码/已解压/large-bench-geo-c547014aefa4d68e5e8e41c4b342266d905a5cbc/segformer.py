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


import wandb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

import segmentation_models_pytorch as smp
from torchvision import transforms

import os
from PIL import Image
import numpy as np
from torch.nn import functional as F
import random
from utils import yaml_config_hook

from datasets import ThebeDataset, CRACKSDataset, SYNTHDataset


def log_segmentation_results(images, masks, preds, num_samples=4):
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    log_images = []

    for i in range(min(num_samples, images.shape[0])):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].imshow(images[i, 0], cmap="seismic")  # Original Seismic Image
        ax[0].set_title("Seismic Image")

        # ax[1].imshow(masks[i], cmap="gray")  # Ground Truth
        ax[1].imshow(masks[i].squeeze(0), cmap="gray")
        ax[1].set_title("Ground Truth")

        # ax[2].imshow(preds[i], cmap="gray")  # Model Prediction
        ax[2].imshow(preds[i], cmap="gray")
        ax[2].set_title("Model Prediction")

        plt.tight_layout()

        # Log figure as wandb image
        log_images.append(wandb.Image(fig))
        plt.close(fig)

    wandb.log({"Segmentation Samples": log_images})


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

def generalized_dice_loss_fp_penalized(y_true, y_pred, alpha=2.0, epsilon=1e-6):
    """
    Generalized Dice Loss with False Positive penalty.

    Args:
        y_true: Tensor of ground truth labels (binary: 0 or 1).
        y_pred: Tensor of predicted probabilities.
        alpha: Scaling factor for penalizing false positives.
        epsilon: Small constant for numerical stability.

    Returns:
        Dice loss value with weighted FP penalty.
    """
    y_true = y_true.float()
    y_pred = torch.sigmoid(y_pred)  # Convert logits to probabilities

    # Compute per-class weights (inverse frequency)
    foreground_pixels = torch.sum(y_true, dim=(1, 2, 3)) + epsilon  # Avoid div-by-zero
    background_pixels = torch.sum(1 - y_true, dim=(1, 2, 3)) + epsilon
    w_foreground = 1.0 / (foreground_pixels ** 2)  # Higher weight for sparse foreground
    w_background = 1.0 / (background_pixels ** 2)  # Lower weight for background

    # Compute intersection and denominator
    intersection = torch.sum(w_foreground * torch.sum(y_true * y_pred, dim=(1, 2, 3)))

    # Denominator includes false positives (weighted by alpha)
    false_positives = torch.sum(w_background * torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3)))
    denominator = torch.sum(w_foreground * torch.sum(y_true + y_pred, dim=(1, 2, 3))) + alpha * false_positives

    dice_coeff = (2. * intersection + epsilon) / (denominator + epsilon)

    return 1.0 - dice_coeff  # Loss is 1 - Dice


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

def main(args):

    wandb.init(project="segformer-" + args.mode + "-" + args.dataset)
    model = SegformerModel(num_classes=args.num_classes).to(args.device)
    if args.mode == 'finetune':
        model = torch.load(f'{args.save_path}/segformer_best.pth')

        # Freeze the first half of the encoder layers
        for idx, layer in enumerate(model.model.segformer.encoder.block):
            if idx < len(model.model.segformer.encoder.block) // 2:  # Freeze early layers
                for param in layer.parameters():
                    param.requires_grad = False

        # Ensure decoder (segmentation head) is trainable
        for param in model.model.decode_head.parameters():
            param.requires_grad = True

        # Verify trainable vs. frozen parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

        print(f"Trainable Parameters: {trainable_params}")
        print(f"Frozen Parameters: {frozen_params}")

    criterion = dice_loss_with_empty_handling  # nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:1, :, :])
    ])

    if args.dataset == 'thebe':
        train_dataset = ThebeDataset(args.root_dir, mode='train', image_transform=image_transform, mask_transform=mask_transform)
        val_dataset = ThebeDataset(args.root_dir, mode='val', image_transform=image_transform, mask_transform=mask_transform)
    elif args.dataset == 'cracks-novice':
        train_dataset = CRACKSDataset(args.root_dir, mode='train', annotator='novice', image_transform=image_transform, mask_transform=mask_transform)
        val_dataset = CRACKSDataset(args.root_dir, mode='val', annotator='novice', image_transform=image_transform, mask_transform=mask_transform)
    elif args.dataset == 'cracks-practitioner':
        train_dataset = CRACKSDataset(args.root_dir, mode='train', annotator='prac', image_transform=image_transform, mask_transform=mask_transform)
        val_dataset = CRACKSDataset(args.root_dir, mode='val', annotator='prac', image_transform=image_transform, mask_transform=mask_transform)
    elif args.dataset == 'cracks-expert':
        train_dataset = CRACKSDataset(args.root_dir, mode='train', annotator='expert', image_transform=image_transform, mask_transform=mask_transform)
        val_dataset = CRACKSDataset(args.root_dir, mode='val', annotator='expert', image_transform=image_transform, mask_transform=mask_transform)
    elif args.dataset == 'synth':
        train_dataset = SYNTHDataset(args.root_dir, mode='train', image_transform=image_transform, mask_transform=mask_transform, dim=(args.dim, args.dim, args.dim))
        val_dataset = SYNTHDataset(args.root_dir, mode='val', image_transform=image_transform, mask_transform=mask_transform, dim=(args.dim, args.dim, args.dim))


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    counter = 0
    min_loss = np.inf

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}', leave=False):
            images, masks = batch
             #images, masks = preprocess(images, masks)
            images, masks = images.to(args.device), masks.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            #outputs = torch.sigmoid(outputs)
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)[:, 1, :, :]
             # masks = masks.unsqueeze(1)
             # inv_masks= 1-masks
             # masks=torch.cat((inv_masks,masks),1)

            loss = criterion(masks,outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {epoch_loss:.4f}')
        model.eval()
        running_loss = 0.0

        all_images, all_masks, all_preds = [], [], []
        for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}', leave=False):
            images, masks = batch

            images, masks = images.to(args.device), masks.to(args.device)

            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            outputs = nn.functional.interpolate(
                outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )[:, 1, :, :]

            #outputs = torch.sigmoid(outputs)


            loss = criterion(masks, outputs)

            running_loss += loss.item() * images.size(0)

            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_preds.append((outputs > 0.5).cpu().float())  # Binarize predictions

        valid_loss = running_loss / len(val_loader.dataset)
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Val Loss: {valid_loss:.4f}')

        wandb.log({
            "Train Loss": epoch_loss,
            "Validation Loss": valid_loss,
            "Learning Rate": optimizer.param_groups[0]["lr"]
        })

        all_images = torch.cat(all_images, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_preds = torch.cat(all_preds, dim=0)



        log_segmentation_results(all_images, all_masks, all_preds)
        scheduler.step(valid_loss)

        if np.round(valid_loss, 6) < np.round(min_loss, 6):
            counter = 0
            min_loss = valid_loss
            if args.mode == 'pretrain':
                torch.save(model, f'{args.save_path}/segformer_best.pth')
                print("model saved at", str(min_loss) + f'{args.save_path}/segformer_best.pth')
            elif args.mode == 'finetune':
                torch.save(model, f'{args.save_path}/segformer_best_finetuned_on_{args.dataset}.pth')
                print("model saved at", str(min_loss) + f'{args.save_path}/segformer_best_finetuned_on_{args.dataset}.pth')
        else:
            counter += 1
        if counter > 10:
            print("model saved at ", str(min_loss) + ', early stopping...')

            break



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train-Thebe")
    config = yaml_config_hook("config/segformer.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()


    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    main(args)