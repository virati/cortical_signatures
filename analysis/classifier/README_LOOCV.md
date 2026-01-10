# SVM Leave-One-Patient-Out Cross-Validation

This directory contains an implementation of SVM classification with leave-one-patient-out cross-validation (LOOCV) for EEG data analysis.

## Overview

The implementation trains SVM classifiers using a rigorous leave-one-patient-out cross-validation strategy, where:
1. For each patient in the dataset:
   - Train SVM on all other patients
   - Test on the held-out patient
2. Collect coefficients from each fold (patient)
3. Plot coefficient distributions across patients

This approach addresses the TODO comment in `online_classifier.py:109` which noted that proper cross-validation was needed.

## Files

### Main Scripts

- **`svm_loocv_classifier.py`**: Main implementation of leave-one-patient-out cross-validation
  - Loads preprocessed EEG data
  - Performs LOOCV across patients
  - Collects SVM coefficients from each fold
  - Generates comprehensive visualizations

- **`test_svm_loocv.py`**: Generates mock data for testing
  - Creates synthetic EEG-like data with realistic structure
  - Useful for testing the pipeline without access to real data

### Legacy Script

- **`online_classifier.py`**: Original classifier (uses simple train-test split)

## Usage

### Step 1: Prepare Data

First, run the preprocessing script to generate the data file:

```bash
python form_streaming_dEEG_struct.py
```

This creates a pickle file with the structure:
```python
{
    'States': list of patients, each containing list of conditions with feature matrices
    'Labels': list of patients, each containing list of conditions with labels
    'Times': list of patients, each containing list of conditions with timestamps
}
```

### Step 2: Run LOOCV Classifier

```bash
python svm_loocv_classifier.py [OPTIONS]
```

#### Options:

- `--data_path PATH`: Path to preprocessed EEG pickle file (default: `/home/virati/Dropbox/projects/Research/MDD-DBS/Data/streaming_EEG.pickle`)
- `--output_dir DIR`: Directory to save output plots (default: `/tmp`)
- `--regularization {l1,l2}`: SVM regularization type (default: `l1`)
- `--do_null`: Perform null testing by shuffling labels

#### Example:

```bash
# With real data
python svm_loocv_classifier.py --output_dir ./results --regularization l1

# With mock data for testing
python test_svm_loocv.py  # Generate mock data first
python svm_loocv_classifier.py --data_path /tmp/mock_streaming_EEG.pickle --output_dir ./test_results
```

## Output

The script generates the following visualizations:

### 1. Overall Results (`loocv_overall_results.png`)
- Bar plot of accuracy for each patient fold
- Aggregate confusion matrix across all folds

### 2. Coefficient Distributions (`coefficient_dist_class_*.png`)
For each class (OFF, OffTON, OnTON):
- Histograms showing distribution of coefficients across all channels and patients
- Separate histogram for each frequency band (Delta, Theta, Alpha, Beta, Gamma)
- Statistics (mean, std) for each distribution

### 3. Channel Coefficient Heatmaps (`channel_coeffs_heatmap_class_*.png`)
For each class:
- Heatmap of mean coefficients across patients (channels × frequency bands)
- Heatmap of std dev of coefficients across patients

### 4. Top Variable Channels (`top_channels_boxplot_class_*.png`)
For each class:
- Box plots showing coefficient distributions for the 10 most variable channels
- Focuses on Alpha band coefficients

## Key Features

### Proper Cross-Validation
- **Leave-One-Patient-Out**: Each patient is used as test set exactly once
- **No data leakage**: Train and test sets are completely separate at patient level
- **Reproducible**: Clear patient assignment for each fold

### Comprehensive Coefficient Analysis
- **Per-fold coefficients**: Collects coefficients from each trained model
- **Multi-dimensional analysis**:
  - By class (OFF, OffTON, OnTON)
  - By channel (257 EEG channels)
  - By frequency band (Delta, Theta, Alpha, Beta, Gamma)
- **Statistical summaries**: Mean, std, min, max across patients

### Visualizations
- **Distribution plots**: Show how coefficients vary across patients
- **Heatmaps**: Reveal spatial and spectral patterns
- **Box plots**: Highlight channels with high inter-patient variability

## Data Structure

### Input Data Format

The preprocessing pipeline expects:
- **Patients**: 905, 906, 907, 908 (configurable)
- **Conditions**: OnT (On-Target), OffT (Off-Target)
- **Channels**: 257 (GSN-HydroCel electrode array)
- **Frequency Bands**: 5 (Delta, Theta, Alpha, Beta, Gamma)
- **Labels**:
  - 0 = OFF (no stimulation)
  - 1 = OffTON (Off-Target stimulation ON)
  - 2 = OnTON (On-Target stimulation ON)

### Feature Matrix Shape

- Raw: `(n_samples, 257 * 5)` = `(n_samples, 1285)`
- Reshaped: `(n_samples, 257 channels, 5 bands)`

## Implementation Details

### SVM Configuration
- **Model**: `sklearn.svm.LinearSVC`
- **Regularization**: L1 or L2 (configurable)
- **Parameters**: `dual=False`, `max_iter=10000`
- **Multi-class**: One-vs-rest strategy (3 classes)

### Coefficient Extraction
```python
# Extract coefficients from trained SVM
coeffs = clf.coef_.reshape(3, 257, -1, order='F')
# Shape: (n_classes, n_channels, n_bands)
```

### Cross-Validation Loop
```python
for test_patient in range(n_patients):
    # Split by patient ID
    train_mask = patient_ids != test_patient
    test_mask = patient_ids == test_patient

    # Train on N-1 patients
    clf.fit(X_train, y_train)

    # Test on held-out patient
    accuracy = clf.score(X_test, y_test)

    # Store coefficients
    coeffs = clf.coef_.reshape(3, 257, -1, order='F')
    all_fold_coefficients.append(coeffs)
```

## Comparison with Original Implementation

### Original (`online_classifier.py`)
- Uses simple train-test split (33% test)
- Performs 10 iterations with random splits
- No guarantee of patient separation
- Selects "best" model based on CV accuracy
- Plots coefficients from single model

### New (`svm_loocv_classifier.py`)
- Leave-one-patient-out cross-validation
- Each patient used as test set exactly once
- Strict patient-level separation
- Analyzes coefficients from ALL folds
- Shows coefficient distributions across patients

## Advantages of LOOCV Approach

1. **Generalization**: Tests model's ability to generalize to new patients
2. **No data leakage**: Training and test data completely independent at patient level
3. **Comprehensive**: Every patient serves as test set
4. **Reproducible**: Deterministic fold assignment
5. **Coefficient stability**: Reveals which features are consistent across patients

## Requirements

- Python 3.6+
- scikit-learn
- numpy
- matplotlib
- seaborn
- scipy
- pickle

## Notes

- The script expects data preprocessed by `form_streaming_dEEG_struct.py`
- DBSpace library required for preprocessing (not needed for main classifier)
- Coefficient order uses Fortran ordering ('F') to match original implementation
- All plots use seaborn styling for publication-quality figures

## Future Enhancements

Potential improvements:
- [ ] Add nested cross-validation for hyperparameter tuning
- [ ] Implement permutation testing for statistical significance
- [ ] Add 3D scalp visualizations using EEG_Viz
- [ ] Export coefficients to CSV for external analysis
- [ ] Add support for LFP features
- [ ] Implement feature selection based on coefficient stability
