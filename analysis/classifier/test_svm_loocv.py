#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for SVM LOOCV with mock data
This generates synthetic data to test the leave-one-patient-out cross-validation implementation
"""

import numpy as np
import pickle
import os

# Create mock data that mimics the structure of the real EEG data
np.random.seed(42)

n_patients = 4
n_conditions = 2  # OnT and OffT
n_channels = 257
n_bands = 5
samples_per_condition = 100  # Number of segments per condition

print("Generating mock EEG data...")

rec = []  # States
lab = []  # Labels
times = []

for pt in range(n_patients):
    pt_rec = []
    pt_lab = []
    pt_times = []

    for cond in range(n_conditions):
        # Generate feature matrix (samples x (channels * bands))
        n_samples = samples_per_condition + np.random.randint(-10, 10)
        features = np.random.randn(n_samples, n_channels * n_bands) * 0.5

        # Add some patient-specific and condition-specific patterns
        # This simulates real differences between patients and conditions
        patient_offset = pt * 0.3
        condition_offset = cond * 0.5

        features += patient_offset
        features[:, :50] += condition_offset  # First 50 features differ by condition

        # Generate labels: mix of OFF (0), OffTON (1), and OnTON (2)
        # Make the distribution depend on condition
        if cond == 0:  # OnT condition - more OnTON
            labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.2, 0.6])
        else:  # OffT condition - more OffTON
            labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.6, 0.2])

        # Generate timestamps
        timestamps = np.arange(n_samples)

        pt_rec.append(features)
        pt_lab.append(labels)
        pt_times.append(timestamps)

    rec.append(pt_rec)
    lab.append(pt_lab)
    times.append(pt_times)

print(f"Generated data for {n_patients} patients")
for pt in range(n_patients):
    total_samples = sum([len(pt_lab) for pt_lab in lab[pt]])
    print(f"  Patient {pt}: {total_samples} total samples")

# Save to pickle file
output_dir = '/tmp'
output_file = os.path.join(output_dir, 'mock_streaming_EEG.pickle')

with open(output_file, 'wb') as f:
    pickle.dump({'States': rec, 'Labels': lab, 'Times': times}, f)

print(f"\nSaved mock data to {output_file}")
print("\nNow run the LOOCV classifier with this mock data by modifying the data path in svm_loocv_classifier.py")
