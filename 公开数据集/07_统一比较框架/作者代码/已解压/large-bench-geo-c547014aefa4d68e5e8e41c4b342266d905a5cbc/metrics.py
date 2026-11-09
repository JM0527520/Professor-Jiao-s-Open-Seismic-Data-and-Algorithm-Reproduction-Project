import os
import glob
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, generate_binary_structure
from sklearn.neighbors import NearestNeighbors
import argparse
from skimage.metrics import hausdorff_distance

def compute_bcd_2d(label, prediction):
    label_points = np.argwhere(label == 1)
    prediction_points = np.argwhere(prediction == 1)

    if len(label_points) == 0 and len(prediction_points) == 0:
        return 0
    if len(label_points) == 0 or len(prediction_points) == 0:
        return np.sqrt(label.shape[0]**2 + label.shape[1]**2)

    nbrs_label = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(prediction_points)
    dists_label_to_pred, _ = nbrs_label.kneighbors(label_points)

    nbrs_pred = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(label_points)
    dists_pred_to_label, _ = nbrs_pred.kneighbors(prediction_points)

    min_dist_label_to_pred = np.mean(dists_label_to_pred)
    min_dist_pred_to_label = np.mean(dists_pred_to_label)

    if np.isinf(min_dist_label_to_pred + min_dist_pred_to_label) or np.isnan(min_dist_label_to_pred + min_dist_pred_to_label):
        return np.sqrt(label.shape[0]**2 + label.shape[1]**2)

    return min_dist_label_to_pred + min_dist_pred_to_label


def Dice(mask1, mask2):
    # Convert masks to boolean masks based on non-zero RGB values

    mask1[mask1 != 0] = 1
    mask2[mask2 != 0] = 1

    if mask1.sum() == mask2.sum() == 0:
        return 1.
    # Calculate the intersection of color coincidences
    intersection = np.logical_and(mask1, mask2)

    # Calculate the Dice coefficient
    dice = (2 * np.sum(intersection)) / (np.sum(mask1) + np.sum(mask2))
    return dice

def hausdorff(mask1, mask2):
    d = hausdorff_distance(mask1.astype('bool'), mask2.astype('bool'),method='modified')
    if np.isinf(d) or np.isnan(d):
        return np.sqrt(mask1.shape[0] ** 2 + mask2.shape[1] ** 2)
    return d

def get_metrics_for_threshold(pred_prob, gt_image, threshold):
    # Threshold the prediction using the best threshold
    pred_binary = (pred_prob > threshold).astype(np.uint8)

    # Morphological processing: skeletonization and dilation
    skeleton = skeletonize(pred_binary)
    structure = generate_binary_structure(2, 1)
    dilated = binary_dilation(skeleton, structure=structure, iterations=1)

    # Compute the Bidirectional Chamfer Distance (BCD)
    metric_bcd = compute_bcd_2d(gt_image, dilated)

    metric_dice = Dice(gt_image, dilated)

    metric_hausdorff = hausdorff(gt_image, dilated)

    return metric_bcd, metric_dice, metric_hausdorff


def get_metrics_from_path(gt_folder, pred_folder, eval_gt_folder, eval_pred_folder):


    # List all ground truth files (sorted)
    gt_files = sorted(glob.glob(os.path.join(gt_folder, '*.npy')))

    # Build a dictionary for all prediction files keyed by their base filename
    pred_files_all = glob.glob(os.path.join(pred_folder, '*.npy'))
    pred_dict = {os.path.basename(f): f for f in pred_files_all}

    # Create a list of file pairs (ground truth, prediction)
    file_pairs = []
    for gt_file in gt_files:
        base = os.path.basename(gt_file)
        if base in pred_dict:
            file_pairs.append((gt_file, pred_dict[base]))
        else:
            print(f"Warning: No corresponding prediction file found for {gt_file}")

    print("Total file pairs for threshold selection:", len(file_pairs))

    if len(file_pairs) == 0:
        return None

    # Loop over thresholds (from 0.01 to 0.99) and compute the metric
    metrics_bcd = []
    metrics_dice = []
    metrics_hd = []
    for th in tqdm(range(1, 10), desc="Processing thresholds"):
        metric_total_bcd = 0.0
        metric_total_dice = 0.0
        metric_total_hd = 0.0
        # Iterate over each pair of ground truth and prediction files
        for gt_file, pred_file in tqdm(file_pairs, desc=f"Threshold {th/10:.2f}", leave=False, total=len(file_pairs)):
            # Load the numpy arrays
            gt_image = np.load(gt_file)       # Binary ground truth image (0 or 1)
            pred_prob = np.load(pred_file)      # Probability map (already sigmoid-activated)

            # Apply threshold to prediction probabilities
            pred_binary = (pred_prob > (th / 10)).astype(np.uint8)

            # Skeletonize the thresholded prediction
            skeleton = skeletonize(pred_binary)
            # Dilate the skeleton to account for small misalignments
            structure = generate_binary_structure(2, 1)
            dilated = binary_dilation(skeleton, structure=structure, iterations=1)

            # Accumulate the metric using the Bidirectional Chamfer Distance
            metric_total_bcd += compute_bcd_2d(gt_image, dilated)
            metric_total_hd += hausdorff(gt_image, dilated)
            metric_total_dice += Dice(gt_image, dilated)

        # Average metric for the current threshold
        metrics_bcd.append(metric_total_bcd / len(file_pairs))
        metrics_dice.append(metric_total_dice / len(file_pairs))
        metrics_hd.append(metric_total_hd / len(file_pairs))

    # Convert the metrics list to a numpy array and determine the best threshold
    metrics_bcd = np.asarray(metrics_bcd)
    metrics_dice = np.asarray(metrics_dice)
    metrics_hd = np.asarray(metrics_hd)
    best_threshold_bcd = list(range(1, 10))[np.argmin(metrics_bcd)] / 10
    best_threshold_dice = list(range(1, 10))[np.argmax(metrics_dice)] / 10
    best_threshold_hd = list(range(1, 10))[np.argmin(metrics_hd)] / 10

    min_metric_bcd = np.min(metrics_bcd)
    max_metric_dice = np.max(metrics_dice)
    min_metric_hd = np.min(metrics_hd)

    print("Min Metric BCD:", min_metric_bcd)
    print(f"Best BCD Threshold: {best_threshold_bcd}")

    print("Max Metric Dice:", max_metric_dice)
    print(f"Best Dice Threshold: {best_threshold_dice}")

    print("Min Metric HD:", min_metric_hd)
    print(f"Best HD Threshold: {best_threshold_hd}")




    # Build a dictionary for evaluation ground truth files keyed by filename for fast lookup
    eval_gt_files = glob.glob(os.path.join(eval_gt_folder, '*.npy'))
    eval_gt_dict = {os.path.basename(f): f for f in eval_gt_files}

    # List evaluation prediction files and pair with corresponding ground truth files
    eval_pred_files = glob.glob(os.path.join(eval_pred_folder, '*.npy'))
    eval_file_pairs = []
    for pred_file in eval_pred_files:
        base = os.path.basename(pred_file)
        if base in eval_gt_dict:
            eval_file_pairs.append((eval_gt_dict[base], pred_file))
        else:
            print(f"Warning: No ground truth file for {pred_file}")

    print("Total evaluation file pairs:", len(eval_file_pairs))

    # Process each evaluation image using the best threshold
    total_metric_bcd_on_bcd = []
    total_metric_dice_on_bcd = []
    total_metric_hausdorff_on_bcd = []

    total_metric_bcd_on_dice = []
    total_metric_dice_on_dice = []
    total_metric_hausdorff_on_dice = []

    total_metric_bcd_on_hd = []
    total_metric_dice_on_hd = []
    total_metric_hausdorff_on_hd = []
    results = {}  # To store per-image metrics

    for gt_file, pred_file in tqdm(eval_file_pairs, desc="Evaluating predictions"):
        # Load ground truth (binary) and prediction (probability) images
        gt_image = np.load(gt_file)
        pred_prob = np.load(pred_file)

        # Threshold the prediction using the best threshold
        metric_bcd, metric_dice, metric_hausdorff = get_metrics_for_threshold(pred_prob, gt_image, best_threshold_bcd)

        total_metric_bcd_on_bcd.append(metric_bcd)
        total_metric_dice_on_bcd.append(metric_dice)
        total_metric_hausdorff_on_bcd.append(metric_hausdorff)

        metric_bcd, metric_dice, metric_hausdorff = get_metrics_for_threshold(pred_prob, gt_image, best_threshold_dice)

        total_metric_bcd_on_dice.append(metric_bcd)
        total_metric_dice_on_dice.append(metric_dice)
        total_metric_hausdorff_on_dice.append(metric_hausdorff)

        metric_bcd, metric_dice, metric_hausdorff = get_metrics_for_threshold(pred_prob, gt_image, best_threshold_hd)

        total_metric_bcd_on_hd.append(metric_bcd)
        total_metric_dice_on_hd.append(metric_dice)
        total_metric_hausdorff_on_hd.append(metric_hausdorff)

        # Store the metric with the image's filename for later inspection
        # results['BCD-' + os.path.basename(pred_file)] = metric_bcd
        # results['Dice-' + os.path.basename(pred_file)] = metric_dice
        # results['Hausdorff-' + os.path.basename(pred_file)] = metric_hausdorff

    # Compute the average metric over all evaluated images
    if len(eval_file_pairs) > 0:
        avg_metric_bcd_on_bcd = np.mean(np.array(total_metric_bcd_on_bcd))
        avg_metric_dice_on_bcd = np.mean(np.array(total_metric_dice_on_bcd))
        avg_metric_hausdorff_on_bcd = np.mean(np.array(total_metric_hausdorff_on_bcd))

        std_metric_bcd_on_bcd = np.std(np.array(total_metric_bcd_on_bcd))
        std_metric_dice_on_bcd = np.std(np.array(total_metric_dice_on_bcd))
        std_metric_hausdorff_on_bcd = np.std(np.array(total_metric_hausdorff_on_bcd))

        avg_metric_bcd_on_dice = np.mean(np.array(total_metric_bcd_on_dice))
        avg_metric_dice_on_dice = np.mean(np.array(total_metric_dice_on_dice))
        avg_metric_hausdorff_on_dice = np.mean(np.array(total_metric_hausdorff_on_dice))

        std_metric_bcd_on_dice = np.std(np.array(total_metric_bcd_on_dice))
        std_metric_dice_on_dice = np.std(np.array(total_metric_dice_on_dice))
        std_metric_hausdorff_on_dice = np.std(np.array(total_metric_hausdorff_on_dice))

        avg_metric_bcd_on_hd = np.mean(np.array(total_metric_bcd_on_hd))
        avg_metric_dice_on_hd = np.mean(np.array(total_metric_dice_on_hd))
        avg_metric_hausdorff_on_hd = np.mean(np.array(total_metric_hausdorff_on_hd))

        std_metric_bcd_on_hd = np.std(np.array(total_metric_bcd_on_hd))
        std_metric_dice_on_hd = np.std(np.array(total_metric_dice_on_hd))
        std_metric_hausdoff_on_hd = np.std(np.array(total_metric_hausdorff_on_hd))

    else:
        return None

    # with open(args.log_file_path, "a") as log_file:
    #
    #     print("Evaluation Results:")
    #     log_file.write(f"Evaluation Results at {args.eval_pred_folder}\n")
    #
    #     print(f"Number of evaluated images: {len(eval_file_pairs)}")
    #     log_file.write(f"Number of evaluated images: {len(eval_file_pairs)}\n")
    #
    #     log_file.write(f"Best Threshold: {best_threshold_bcd}\n")
    #
    #     print(f"Average BCD_2D: {avg_metric_bcd}")
    #     log_file.write(f"Average BCD_2D: {avg_metric_bcd}\n")
    #
    #     print(f"Average Dice: {avg_metric_dice}")
    #     log_file.write(f"Average Dice: {avg_metric_dice}\n")
    #
    #     print(f"Average Hausdorff: {avg_metric_hausdorff}")
    #     log_file.write(f"Average Hausdorff: {avg_metric_hausdorff}\n")

    res = {'dice' : [best_threshold_dice, avg_metric_bcd_on_dice, avg_metric_dice_on_dice, avg_metric_hausdorff_on_dice, std_metric_bcd_on_dice, std_metric_dice_on_dice, std_metric_hausdorff_on_dice],
           'bcd': [best_threshold_bcd, avg_metric_bcd_on_bcd, avg_metric_dice_on_bcd, avg_metric_hausdorff_on_bcd, std_metric_bcd_on_bcd, std_metric_dice_on_bcd, std_metric_hausdorff_on_bcd],
           'hd': [best_threshold_hd, avg_metric_bcd_on_hd, avg_metric_dice_on_hd, avg_metric_hausdorff_on_hd, std_metric_bcd_on_hd, std_metric_dice_on_hd, std_metric_hausdoff_on_hd]}

    return res

    # Optionally, print per-image results
    # for fname, m in results.items():
    #     print(f"{fname}: BCD_2D = {m}")
def get_stds_given_threshold(eval_gt_folder, eval_pred_folder, threshold):
    # Build a dictionary for evaluation ground truth files keyed by filename for fast lookup
    eval_gt_files = glob.glob(os.path.join(eval_gt_folder, '*.npy'))
    eval_gt_dict = {os.path.basename(f): f for f in eval_gt_files}

    # List evaluation prediction files and pair with corresponding ground truth files
    eval_pred_files = glob.glob(os.path.join(eval_pred_folder, '*.npy'))
    eval_file_pairs = []
    for pred_file in eval_pred_files:
        base = os.path.basename(pred_file)
        if base in eval_gt_dict:
            eval_file_pairs.append((eval_gt_dict[base], pred_file))
        else:
            print(f"Warning: No ground truth file for {pred_file}")

    print("Total evaluation file pairs:", len(eval_file_pairs))

    metrics_bcd = []
    metrics_dice = []
    metrics_hd = []

    for gt_file, pred_file in tqdm(eval_file_pairs, desc="Evaluating predictions"):
        # Load ground truth (binary) and prediction (probability) images
        gt_image = np.load(gt_file)
        pred_prob = np.load(pred_file)

        # Threshold the prediction using the best threshold
        metric_bcd, metric_dice, metric_hausdorff = get_metrics_for_threshold(pred_prob, gt_image, threshold)

        metrics_bcd.append(metric_bcd)
        metrics_dice.append(metric_dice)
        metrics_hd.append(metric_hausdorff)

    # Compute the average metric over all evaluated images
    metrics_bcd = np.array(metrics_bcd)
    metrics_dice = np.array(metrics_dice)
    metrics_hd = np.array(metrics_hd)

    return np.mean(metrics_bcd), np.mean(metrics_dice), np.mean(metrics_hd), np.std(metrics_bcd), np.std(metrics_dice), np.std(metrics_hd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run Bidirectional Chamfer Distance (BCD_2D) evaluation experiment."
    )
    parser.add_argument(
        "--gt_folder", type=str, required=True,
        help="Folder with binary ground truth .npy files (for threshold selection)"
    )
    parser.add_argument(
        "--pred_folder", type=str, required=True,
        help="Folder with prediction probability .npy files (sigmoid activated, for threshold selection)"
    )
    parser.add_argument(
        "--eval_gt_folder", type=str, required=True,
        help="Folder with evaluation ground truth .npy files"
    )
    parser.add_argument(
        "--eval_pred_folder", type=str, required=True,
        help="Folder with evaluation prediction .npy files (sigmoid activated)"
    )
    parser.add_argument(
        "--log_file_path", type=str, required=True,
        help="File to store prediction metrics)"
    )


    args = parser.parse_args()
    get_metrics_from_path(args.gt_folder, args.pred_folder, args.eval_gt_folder, args.eval_pred_folder)