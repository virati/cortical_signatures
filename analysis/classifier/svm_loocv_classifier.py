#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM Classifier with Leave-One-Patient-Out Cross-Validation

This script implements leave-one-patient-out cross-validation for SVM classification
and plots coefficient distributions across patients.

Based on online_classifier.py but modified for proper LOOCV.

Usage:
    python svm_loocv_classifier.py [--data_path PATH] [--output_dir DIR] [--regularization l1|l2]

Arguments:
    --data_path: Path to the preprocessed EEG pickle file (default: see below)
    --output_dir: Directory to save output plots (default: /tmp)
    --regularization: Regularization type for SVM (default: l1)

The script expects a pickle file with the following structure:
    {
        'States': list of patients, each with list of conditions containing feature matrices
        'Labels': list of patients, each with list of conditions containing labels
        'Times': list of patients, each with list of conditions containing timestamps
    }
"""

import sklearn
from sklearn import svm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.decomposition import PCA
import scipy.stats as stats
import os
import argparse

sns.set_context('paper')
sns.set(font_scale=2)
sns.set_style('white')

#%% Parse command line arguments
parser = argparse.ArgumentParser(description='SVM LOOCV Classifier')
parser.add_argument('--data_path', type=str,
                    default='/home/virati/Dropbox/projects/Research/MDD-DBS/Data/streaming_EEG.pickle',
                    help='Path to preprocessed EEG pickle file')
parser.add_argument('--output_dir', type=str, default='/tmp',
                    help='Directory to save output plots')
parser.add_argument('--regularization', type=str, default='l1', choices=['l1', 'l2'],
                    help='Regularization type for SVM')
parser.add_argument('--do_null', action='store_true',
                    help='Perform null testing by shuffling labels')

args = parser.parse_args()

regularization = args.regularization
do_null = args.do_null
output_dir = args.output_dir

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

#%% Load the preprocessed EEG data
print("="*60)
print("SVM Leave-One-Patient-Out Cross-Validation")
print("="*60)
print(f"\nConfiguration:")
print(f"  Data path: {args.data_path}")
print(f"  Output directory: {output_dir}")
print(f"  Regularization: {regularization}")
print(f"  Null testing: {do_null}")
print()

print("Loading preprocessed EEG data...")
try:
    with open(args.data_path, 'rb') as f:
        inFile = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Data file not found at {args.data_path}")
    print("\nTo generate mock data for testing, run:")
    print("  python test_svm_loocv.py")
    print("\nThen run this script with:")
    print("  python svm_loocv_classifier.py --data_path /tmp/mock_streaming_EEG.pickle")
    exit(1)

rec = inFile['States']  # List of patients, each with list of conditions
lab = inFile['Labels']
times = inFile['Times']

# Determine number of patients
n_patients = len(rec)
print(f"Number of patients: {n_patients}")

#%% Prepare data with patient labels
# Concatenate data but keep track of patient IDs
all_data = []
all_labels = []
patient_ids = []

label_map = {0: 'OFF', 2: 'OnTON', 1: 'OffTON'}

for pt_idx in range(n_patients):
    # Concatenate conditions for this patient (OnT and OffT)
    pt_data = np.concatenate([rec[pt_idx][cdt] for cdt in range(len(rec[pt_idx]))])
    pt_labels = np.concatenate([lab[pt_idx][cdt] for cdt in range(len(lab[pt_idx]))])

    all_data.append(pt_data)
    all_labels.append(pt_labels)
    patient_ids.append(np.full(len(pt_data), pt_idx))

    print(f"Patient {pt_idx}: {len(pt_data)} samples")

# Concatenate all data
X_all = np.vstack(all_data)
y_all_numeric = np.concatenate(all_labels)
patient_ids = np.concatenate(patient_ids)

# Convert numeric labels to string labels
y_all = np.array([label_map[item] for item in y_all_numeric])

print(f"\nTotal samples: {len(X_all)}")
print(f"Feature dimensions: {X_all.shape}")

#%% Leave-One-Patient-Out Cross-Validation
print("\n" + "="*60)
print("Starting Leave-One-Patient-Out Cross-Validation")
print("="*60)

# Storage for results
loocv_accuracies = []
loocv_models = []
loocv_predictions = []
loocv_true_labels = []
loocv_test_patient_ids = []

# Store coefficients from each fold
all_fold_coefficients = []  # Will store (n_folds, n_classes, n_channels, n_bands)

for test_patient in range(n_patients):
    print(f"\nFold {test_patient + 1}/{n_patients}: Testing on patient {test_patient}")

    # Split data into train and test
    train_mask = patient_ids != test_patient
    test_mask = patient_ids == test_patient

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")

    # Train SVM
    clf = svm.LinearSVC(penalty=regularization, dual=False, max_iter=10000)

    # Optional: null testing
    if do_null:
        y_train = shuffle(y_train)

    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = clf.score(X_test, y_test)
    loocv_accuracies.append(accuracy)

    print(f"  Accuracy: {accuracy:.3f}")

    # Store results
    loocv_models.append(clf)
    loocv_predictions.append(y_pred)
    loocv_true_labels.append(y_test)
    loocv_test_patient_ids.append(test_patient)

    # Extract and store coefficients (reshape to n_classes x n_channels x n_bands)
    coeffs = clf.coef_.reshape(3, 257, -1, order='F')
    all_fold_coefficients.append(coeffs)

# Convert to numpy array for easier manipulation
all_fold_coefficients = np.array(all_fold_coefficients)  # Shape: (n_patients, 3, 257, n_bands)

print("\n" + "="*60)
print("Cross-Validation Results")
print("="*60)
print(f"Mean accuracy: {np.mean(loocv_accuracies):.3f} ± {np.std(loocv_accuracies):.3f}")
print(f"Individual accuracies: {[f'{acc:.3f}' for acc in loocv_accuracies]}")

#%% Plot overall results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot accuracies per fold
axes[0].bar(range(n_patients), loocv_accuracies)
axes[0].axhline(y=np.mean(loocv_accuracies), color='r', linestyle='--', label='Mean')
axes[0].set_xlabel('Test Patient')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Leave-One-Patient-Out Accuracy')
axes[0].set_ylim([0, 1])
axes[0].legend()

# Confusion matrix (aggregate across all folds)
all_preds = np.concatenate(loocv_predictions)
all_true = np.concatenate(loocv_true_labels)

# Convert string labels to numeric for confusion matrix
label_to_num = {'OFF': 0, 'OffTON': 1, 'OnTON': 2}
all_preds_num = np.array([label_to_num[p] for p in all_preds])
all_true_num = np.array([label_to_num[t] for t in all_true])

conf_mat = confusion_matrix(all_true_num, all_preds_num)
im = axes[1].imshow(conf_mat, cmap='Blues')
axes[1].set_xticks([0, 1, 2])
axes[1].set_yticks([0, 1, 2])
axes[1].set_xticklabels(['OFF', 'OffTON', 'OnTON'])
axes[1].set_yticklabels(['OFF', 'OffTON', 'OnTON'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Aggregate Confusion Matrix')

# Add text annotations
for i in range(3):
    for j in range(3):
        text = axes[1].text(j, i, conf_mat[i, j],
                           ha="center", va="center", color="black" if conf_mat[i, j] < conf_mat.max()/2 else "white")

plt.colorbar(im, ax=axes[1])
plt.tight_layout()
output_file = os.path.join(output_dir, 'loocv_overall_results.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved overall results to {output_file}")

#%% Plot coefficient distributions across patients
print("\n" + "="*60)
print("Plotting Coefficient Distributions Across Patients")
print("="*60)

n_classes = 3
n_channels = 257
n_bands = all_fold_coefficients.shape[-1]

class_names = ['OFF', 'OffTON', 'OnTON']
band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Create figure for coefficient distributions for each class and band
for class_idx in range(n_classes):
    fig, axes = plt.subplots(1, n_bands, figsize=(20, 4))
    fig.suptitle(f'Coefficient Distributions for Class: {class_names[class_idx]}', fontsize=16)

    for band_idx in range(n_bands):
        # Get coefficients for this class and band across all patients and channels
        # Shape: (n_patients, n_channels)
        coeffs_class_band = all_fold_coefficients[:, class_idx, :, band_idx]

        # Plot distribution for each channel
        # We'll create a violin plot showing distribution across patients for each channel
        # For visualization purposes, we'll show statistics rather than all 257 channels

        # Flatten to get all coefficient values for this class/band combo
        all_coeffs = coeffs_class_band.flatten()

        ax = axes[band_idx] if n_bands > 1 else axes
        ax.hist(all_coeffs, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{band_names[band_idx]}')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # Add statistics
        mean_val = np.mean(all_coeffs)
        std_val = np.std(all_coeffs)
        ax.text(0.05, 0.95, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'coefficient_dist_class_{class_idx}_{class_names[class_idx]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved coefficient distribution for {class_names[class_idx]} to {output_file}")

#%% Plot coefficient distributions per channel (across patients)
print("\nPlotting per-channel coefficient distributions...")

# For each class, create a heatmap showing mean and std of coefficients across patients
for class_idx in range(n_classes):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    fig.suptitle(f'Channel Coefficients Across Patients: {class_names[class_idx]}', fontsize=16)

    # Calculate mean and std across patients for each channel and band
    coeffs_class = all_fold_coefficients[:, class_idx, :, :]  # (n_patients, n_channels, n_bands)

    mean_coeffs = np.mean(coeffs_class, axis=0)  # (n_channels, n_bands)
    std_coeffs = np.std(coeffs_class, axis=0)    # (n_channels, n_bands)

    # Plot mean
    im1 = axes[0].imshow(mean_coeffs.T, aspect='auto', cmap='RdBu_r',
                         extent=[0, n_channels, 0, n_bands])
    axes[0].set_xlabel('Channel')
    axes[0].set_ylabel('Frequency Band')
    axes[0].set_title('Mean Coefficient Across Patients')
    axes[0].set_yticks(np.arange(n_bands) + 0.5)
    axes[0].set_yticklabels(band_names)
    plt.colorbar(im1, ax=axes[0])

    # Plot std
    im2 = axes[1].imshow(std_coeffs.T, aspect='auto', cmap='viridis',
                         extent=[0, n_channels, 0, n_bands])
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Frequency Band')
    axes[1].set_title('Std Dev of Coefficient Across Patients')
    axes[1].set_yticks(np.arange(n_bands) + 0.5)
    axes[1].set_yticklabels(band_names)
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'channel_coeffs_heatmap_class_{class_idx}_{class_names[class_idx]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved channel coefficient heatmap for {class_names[class_idx]} to {output_file}")

#%% Create box plots for specific channels of interest
print("\nCreating box plots for channels with highest variability...")

# Find channels with highest std dev across patients (averaged across bands and classes)
avg_std_per_channel = np.mean(np.std(all_fold_coefficients, axis=0), axis=(0, 2))  # Average across classes and bands
top_channels = np.argsort(avg_std_per_channel)[-10:][::-1]  # Top 10 channels

print(f"Top 10 most variable channels: {top_channels}")

# Create box plots for these channels
for class_idx in range(n_classes):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Top 10 Variable Channels - {class_names[class_idx]} (Alpha Band)', fontsize=16)
    axes = axes.flatten()

    alpha_band_idx = 2  # Alpha is the 3rd band (index 2)

    for idx, channel in enumerate(top_channels):
        # Get coefficients for this channel across all patients
        channel_coeffs = all_fold_coefficients[:, class_idx, channel, alpha_band_idx]

        axes[idx].boxplot(channel_coeffs)
        axes[idx].set_title(f'Channel {channel}')
        axes[idx].set_ylabel('Coefficient Value')
        axes[idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[idx].set_xticklabels([''])

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'top_channels_boxplot_class_{class_idx}_{class_names[class_idx]}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved top channels boxplot for {class_names[class_idx]} to {output_file}")

#%% Summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

for class_idx in range(n_classes):
    print(f"\n{class_names[class_idx]}:")
    for band_idx in range(n_bands):
        coeffs = all_fold_coefficients[:, class_idx, :, band_idx]
        print(f"  {band_names[band_idx]:8s}: mean={np.mean(coeffs):7.4f}, std={np.std(coeffs):7.4f}, "
              f"min={np.min(coeffs):7.4f}, max={np.max(coeffs):7.4f}")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
print(f"\nGenerated plots in {output_dir}:")
print("  - loocv_overall_results.png")
print("  - coefficient_dist_class_*.png (3 files)")
print("  - channel_coeffs_heatmap_class_*.png (3 files)")
print("  - top_channels_boxplot_class_*.png (3 files)")
