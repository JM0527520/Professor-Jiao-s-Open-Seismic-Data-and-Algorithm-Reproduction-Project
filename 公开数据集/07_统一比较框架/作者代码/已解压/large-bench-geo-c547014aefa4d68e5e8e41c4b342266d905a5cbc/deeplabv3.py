import torch.distributed as dist


from torchvision.models.segmentation import deeplabv3_resnet50
from skimage.metrics import hausdorff_distance
from medpy.metric.binary import dc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.cuda.empty_cache()
from torch.utils.data import Dataset

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
        ax[2].imshow(preds[i].squeeze(0), cmap="gray")
        ax[2].set_title("Model Prediction")

        plt.tight_layout()

        # Log figure as wandb image
        log_images.append(wandb.Image(fig))
        plt.close(fig)

    wandb.log({"Segmentation Samples": log_images})


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

    wandb.init(project="deeplab-" + args.mode + "-" + args.dataset)
    model = deeplabv3_resnet50(pretrained=False, num_classes=1)
    if args.mode == 'finetune':
        # state_dict = torch.load(f'{args.save_path}/deeplab_best.pth')
        # model.load_state_dict(state_dict)
        model = torch.load(f'{args.save_path}/deeplab_best.pth')
        for param in model.backbone.parameters():
            param.requires_grad = False


    model.to(args.device)
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
            outputs = model(images)["out"]
            #outputs = torch.softmax(outputs, dim=1)[:, 1, :, :]
            outputs = torch.sigmoid(outputs)
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
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

            outputs = model(images)["out"]
            # outputs = torch.softmax(outputs, dim=1)[:, 1, :, :]
            outputs = torch.sigmoid(outputs)
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)

            loss = criterion(masks,outputs)

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
                torch.save(model, f'{args.save_path}/deeplab_best.pth')
                print("model saved at", str(min_loss) + f'{args.save_path}/deeplab_best.pth')
            elif args.mode == 'finetune':
                torch.save(model, f'{args.save_path}/deeplab_best_finetuned_on_{args.dataset}.pth')
                print("model saved at", str(min_loss) + f'{args.save_path}/deeplab_best_finetuned_on_{args.dataset}.pth')
        else:
            counter += 1
        if counter > 10:
            print("model saved at", str(min_loss) + ', early stopping...')

            break
        print('Training complete.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train-Thebe")
    config = yaml_config_hook("config/deeplab.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()


    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    main(args)