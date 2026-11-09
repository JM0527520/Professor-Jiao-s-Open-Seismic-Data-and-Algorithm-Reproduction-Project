# Seismic Fault Segmentation Benchmark

This repository provides a unified benchmark for evaluating deep learning models on the task of seismic fault segmentation, as described in [A Large-scale Benchmark on Geological Fault Delineation Models: Domain Shift, Training Dynamics, Generalizability, Evaluation and Inferential Behavior](https://arxiv.org/abs/2505.08585). It supports multiple architectures (UNet, UNet++, DeepLabV3+, SegFormer) and datasets (real, synthetic, and crowdsourced fault annotations). The benchmark includes modules for training, fine-tuning, inference, and evaluation using standard and task-specific metrics.

## 📦 Download Datasets

You can download the datasets used on this benchamrk in these links:
[CRACKS](https://zenodo.org/records/13926822)
[Thebe](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YBYGBK)
[FaultSeg3D](https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8)

## 🔧 Installation

Make sure the following Python packages are installed:
```
pip install -r requirements.txt
```

## 🚀 Usage

**Train a model:**
Each model script is controlled via a YAML config file. Example:
```
unet.py --config config/unet.yaml
```
You can customize training mode, dataset used and model hyperparameters either by using the config file or overriding the corresponding flag using the command line. Training logs are printed and optionally visualized via Weights & Biases.

**Run inference:**
In order to run inference using a saved model on a folder of .npy images run:
```
python inference.py --config config/inference.yaml
```

**Evaluate predictions:**
To evaluate a models predictions on a set of images, run:
```
python metrics.py \
  --gt_folder path/to/train_ground_truth \
  --pred_folder path/to/train_predictions \
  --eval_gt_folder path/to/eval_ground_truth \
  --eval_pred_folder path/to/eval_predictions \
  --log_file_path results.log
```





