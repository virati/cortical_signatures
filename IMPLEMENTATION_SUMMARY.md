# Electrode Location Estimation - Implementation Summary

## Overview
This implementation adds mass-centroid and geometric centroid calculation capabilities to estimate electrode locations from voltage-specific tractography masks in Deep Brain Stimulation (DBS) studies.

## Files Created

### 1. Core Module: `analysis/eeg_support/mask_analysis.py`
**Purpose**: Provides centroid calculation functions for electrode location estimation

**Key Functions**:
- `calculate_mass_centroid(mask_img, threshold=0.0)` - Weighted centroid calculation
- `calculate_geometric_centroid(mask_img, threshold=0.5)` - Unweighted geometric center
- `calculate_centroids_for_voltages(voltage_masks, method='mass', **kwargs)` - Batch processing
- `combine_voltage_masks(voltage_masks, operation='mean')` - Mask aggregation

**Features**:
- Returns both voxel and world (mm) coordinates
- Supports thresholding for noise reduction
- Works with NIfTI images via nibabel/nilearn
- Handles dictionaries and lists of masks
- Comprehensive error handling

### 2. Example Script: `analysis/eeg_support/example_centroid_calculation.py`
**Purpose**: Demonstrates all functionality with concrete examples

**Includes**:
- Single mask centroid calculation
- Multiple voltage analysis
- Mask combination and stability analysis
- Integration pattern with real DTI data

### 3. Test Suite: `analysis/eeg_support/test_mask_analysis.py`
**Purpose**: Validates all functions with synthetic data

**Coverage**:
- Mass-centroid accuracy with Gaussian blobs
- Geometric centroid accuracy with binary masks
- Multi-voltage processing
- Mask combination operations
- Edge case handling (empty masks, single voxels)

**Status**: All 5 tests passing ✓

### 4. Documentation: `analysis/eeg_support/README_mask_analysis.md`
**Purpose**: Complete API documentation and usage guide

**Contents**:
- Function signatures and parameters
- Usage examples for each function
- Integration guide with existing code
- Dependencies and requirements

## Technical Details

### Centroid Calculation Methods

**Mass-Centroid (Center of Mass)**:
- Formula: `centroid = Σ(position × intensity) / Σ(intensity)`
- Use case: Probabilistic tractography maps where intensity represents connection strength
- Weights voxels by their intensity values
- Best for continuous-valued activation volumes

**Geometric Centroid**:
- Formula: `centroid = mean(positions of all voxels > threshold)`
- Use case: Binary masks or regions of interest
- Treats all voxels equally
- Best for well-defined anatomical regions

### Coordinate Systems
Both methods return:
1. **Voxel coordinates**: (i, j, k) indices in the 3D array
2. **World coordinates**: (x, y, z) position in mm (MNI space or patient space)

Conversion uses the NIfTI affine transformation matrix.

### Integration with Existing Code
The module integrates seamlessly with:
- `analysis/eeg_support/Targeting_DTI.py` - Loads OnT/OffT tractography
- `analysis/DOs/DO_DTI.py` - Processes DO-specific masks
- `nilearn.image.math_img()` - For mask operations
- `dbspace.control.DTI.Etrode_map` - For electrode contact mapping

## Usage Pattern

```python
from nilearn import image
from analysis.eeg_support import mask_analysis

# Load voltage-specific masks
voltage_masks = {
    2: image.load_img('DBS906.L2.2V.bin.nii.gz'),
    3: image.load_img('DBS906.L2.3V.bin.nii.gz'),
    4: image.load_img('DBS906.L2.4V.bin.nii.gz'),
    5: image.load_img('DBS906.L2.5V.bin.nii.gz'),
    6: image.load_img('DBS906.L2.6V.bin.nii.gz'),
}

# Calculate centroids for each voltage
centroids = mask_analysis.calculate_centroids_for_voltages(
    voltage_masks, 
    method='mass', 
    threshold=0.01
)

# Display results
for voltage, (voxel, world, mass) in centroids.items():
    print(f"{voltage}V: Location = {world} mm, Mass = {mass:.2f}")

# Combine masks and get overall estimate
combined = mask_analysis.combine_voltage_masks(voltage_masks, operation='mean')
final_voxel, final_world, final_mass = mask_analysis.calculate_mass_centroid(combined)
print(f"\nEstimated electrode location: {final_world} mm")
```

## Quality Assurance

✓ **Tests**: All tests passing (5/5)
✓ **Code Review**: Addressed all feedback (path handling, imports)
✓ **Security**: CodeQL scan - 0 vulnerabilities found
✓ **Documentation**: Comprehensive README and inline docstrings
✓ **Integration**: Compatible with existing codebase patterns

## Dependencies
All required packages are in `requirements.txt`:
- numpy (array operations)
- nibabel (NIfTI file handling)
- nilearn (neuroimaging utilities)
- scipy (indirect via nilearn)

## Future Enhancements (Optional)
- Add uncertainty quantification (confidence ellipsoids)
- Support for multiple electrode contacts simultaneously
- Visualization functions for centroid locations
- Statistical comparison between OnT/OffT centroids
- Integration with patient-specific anatomy

## References
This implementation supports the analysis described in:
- Targeting_DTI.py: OnTarget vs OffTarget tractography analysis
- DO_DTI.py: Dynamic Oscillation spatial analysis

## Author
@virati

## Date
February 2026
