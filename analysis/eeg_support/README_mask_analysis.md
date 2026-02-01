# Mask Analysis Module

This module provides functions to estimate electrode locations from voltage-specific tractography masks using centroid calculations.

## Overview

When performing Deep Brain Stimulation (DBS), accurately determining the electrode location is critical for understanding the therapeutic effects. This module provides tools to calculate:

1. **Mass-Centroid**: The center of mass of the tractography mask, weighted by voxel intensities
2. **Geometric Centroid**: The geometric center of the binary mask region

## Functions

### `calculate_mass_centroid(mask_img, threshold=0.0)`

Calculates the mass-centroid (center of mass) of a mask image.

**Parameters:**
- `mask_img`: A 3D NIfTI image representing the tractography mask
- `threshold`: Minimum intensity value to consider (default: 0.0)

**Returns:**
- `centroid_voxel`: Voxel coordinates (i, j, k)
- `centroid_world`: World coordinates in mm (x, y, z)
- `mass`: Total mass (sum of intensities)

**Example:**
```python
from analysis.eeg_support import mask_analysis
from nilearn import image

mask = image.load_img('tractography_mask.nii.gz')
voxel, world, mass = mask_analysis.calculate_mass_centroid(mask, threshold=0.1)
print(f"Electrode location: {world} mm")
```

### `calculate_geometric_centroid(mask_img, threshold=0.5)`

Calculates the geometric centroid of a binary mask.

**Parameters:**
- `mask_img`: A 3D NIfTI image
- `threshold`: Threshold for binarization (default: 0.5)

**Returns:**
- `centroid_voxel`: Voxel coordinates (i, j, k)
- `centroid_world`: World coordinates in mm (x, y, z)
- `n_voxels`: Number of voxels in the mask

**Example:**
```python
voxel, world, n_voxels = mask_analysis.calculate_geometric_centroid(mask, threshold=0.5)
print(f"Geometric center: {world} mm ({n_voxels} voxels)")
```

### `calculate_centroids_for_voltages(voltage_masks, method='mass', **kwargs)`

Calculates centroids for multiple voltage-specific masks.

**Parameters:**
- `voltage_masks`: Dictionary mapping voltages to mask images, or list of masks
- `method`: Either 'mass' or 'geometric'
- `**kwargs`: Additional arguments passed to centroid function

**Returns:**
- Dictionary or list with centroids for each voltage

**Example:**
```python
voltage_masks = {
    2: image.load_img('mask_2V.nii.gz'),
    3: image.load_img('mask_3V.nii.gz'),
    4: image.load_img('mask_4V.nii.gz'),
}

centroids = mask_analysis.calculate_centroids_for_voltages(voltage_masks, method='mass')
for v, (voxel, world, mass) in centroids.items():
    print(f"{v}V: {world} mm (mass={mass:.2f})")
```

### `combine_voltage_masks(voltage_masks, operation='mean')`

Combines multiple voltage masks into a single mask.

**Parameters:**
- `voltage_masks`: Dictionary or list of mask images
- `operation`: 'mean', 'sum', 'max', or 'min'

**Returns:**
- Combined NIfTI image

**Example:**
```python
combined = mask_analysis.combine_voltage_masks(voltage_masks, operation='mean')
voxel, world, mass = mask_analysis.calculate_mass_centroid(combined)
print(f"Overall electrode location: {world} mm")
```

## Usage Examples

See `example_centroid_calculation.py` for comprehensive usage examples including:
1. Single mask centroid calculation
2. Multiple voltage analysis
3. Combined mask analysis
4. Integration with real tractography data

Run the examples:
```bash
cd /path/to/cortical_signatures
python analysis/eeg_support/example_centroid_calculation.py
```

## Testing

Run the test suite:
```bash
cd /path/to/cortical_signatures
python analysis/eeg_support/test_mask_analysis.py
```

## Integration with Existing Code

This module integrates seamlessly with existing DTI analysis code in:
- `analysis/eeg_support/Targeting_DTI.py`
- `analysis/DOs/DO_DTI.py`

Example integration:
```python
import dbspace.control.DTI as DTI
from nilearn import image
from analysis.eeg_support import mask_analysis

# Load voltage-specific masks
voltage_masks = {}
for voltage in range(2, 7):
    dti_file = f'DBS906.L2.{voltage}V.bin.nii.gz'
    voltage_masks[voltage] = image.smooth_img(dti_file, fwhm=1)

# Calculate electrode location
centroids = mask_analysis.calculate_centroids_for_voltages(
    voltage_masks, 
    method='mass', 
    threshold=0.01
)

# Analyze stability across voltages
locations = [world for _, world, _ in centroids.values()]
mean_location = np.mean(locations, axis=0)
print(f"Estimated electrode location: {mean_location}")
```

## Dependencies

- numpy
- nibabel
- nilearn
- scipy (indirect via nilearn)

All dependencies are listed in the repository's `requirements.txt`.

## Author

@virati
