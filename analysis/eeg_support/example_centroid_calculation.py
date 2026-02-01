#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Calculate Electrode Location using Centroid Methods

This script demonstrates how to use the mask_analysis module to estimate
electrode locations from voltage-specific tractography masks.

@author: virati
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image, plotting
from analysis.eeg_support import mask_analysis
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')

# Try to import dbspace utilities if available
try:
    from dbspace.utils.structures import nestdict
    import dbspace.control.DTI as DTI
    DBSPACE_AVAILABLE = True
except ImportError:
    print("Warning: dbspace not available. Using simplified example.")
    DBSPACE_AVAILABLE = False
    nestdict = dict


def example_single_mask():
    """
    Example 1: Calculate centroids for a single voltage mask.
    """
    print("=" * 70)
    print("Example 1: Single Voltage Mask Centroid Calculation")
    print("=" * 70)
    
    # In a real scenario, you would load an actual tractography mask
    # mask_img = image.load_img('/path/to/DBS906.L2.3V.bin.nii.gz')
    
    # For this example, create a synthetic mask with a Gaussian blob
    shape = (91, 109, 91)  # Standard MNI space dimensions
    data = np.zeros(shape)
    
    # Create a Gaussian blob centered at (45, 54, 45)
    center = np.array([45, 54, 45])
    sigma = 5
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                pos = np.array([i, j, k])
                dist = np.linalg.norm(pos - center)
                data[i, j, k] = np.exp(-(dist**2) / (2 * sigma**2))
    
    # Create a NIfTI image
    affine = np.diag([2, 2, 2, 1])  # 2mm isotropic voxels
    mask_img = nib.Nifti1Image(data, affine)
    
    # Calculate mass-centroid
    print("\nMass-Centroid Calculation:")
    voxel_coords, world_coords, mass = mask_analysis.calculate_mass_centroid(
        mask_img, threshold=0.1
    )
    print(f"  Voxel coordinates: ({voxel_coords[0]:.2f}, {voxel_coords[1]:.2f}, {voxel_coords[2]:.2f})")
    print(f"  World coordinates: ({world_coords[0]:.2f}, {world_coords[1]:.2f}, {world_coords[2]:.2f}) mm")
    print(f"  Total mass: {mass:.4f}")
    
    # Calculate geometric centroid
    print("\nGeometric Centroid Calculation:")
    voxel_coords, world_coords, n_voxels = mask_analysis.calculate_geometric_centroid(
        mask_img, threshold=0.1
    )
    print(f"  Voxel coordinates: ({voxel_coords[0]:.2f}, {voxel_coords[1]:.2f}, {voxel_coords[2]:.2f})")
    print(f"  World coordinates: ({world_coords[0]:.2f}, {world_coords[1]:.2f}, {world_coords[2]:.2f}) mm")
    print(f"  Number of voxels: {n_voxels}")
    
    return mask_img


def example_multiple_voltages():
    """
    Example 2: Calculate centroids across multiple voltages.
    """
    print("\n" + "=" * 70)
    print("Example 2: Multiple Voltage Masks - Tracking Electrode Location")
    print("=" * 70)
    
    # Create synthetic masks for different voltages
    # Each voltage creates a slightly different activation pattern
    voltage_masks = {}
    shape = (91, 109, 91)
    affine = np.diag([2, 2, 2, 1])
    
    base_center = np.array([45, 54, 45])
    
    for voltage in [2, 3, 4, 5, 6]:
        data = np.zeros(shape)
        # Higher voltage = larger activation radius
        sigma = 3 + voltage * 0.5
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    pos = np.array([i, j, k])
                    dist = np.linalg.norm(pos - base_center)
                    data[i, j, k] = voltage * np.exp(-(dist**2) / (2 * sigma**2))
        
        voltage_masks[voltage] = nib.Nifti1Image(data, affine)
    
    # Calculate mass-centroids for all voltages
    print("\nMass-Centroids Across Voltages:")
    centroids_mass = mask_analysis.calculate_centroids_for_voltages(
        voltage_masks, method='mass', threshold=0.1
    )
    
    for voltage, (voxel, world, mass) in centroids_mass.items():
        print(f"  {voltage}V: World coords = ({world[0]:.2f}, {world[1]:.2f}, {world[2]:.2f}) mm, Mass = {mass:.2f}")
    
    # Calculate geometric centroids for all voltages
    print("\nGeometric Centroids Across Voltages:")
    centroids_geom = mask_analysis.calculate_centroids_for_voltages(
        voltage_masks, method='geometric', threshold=0.1
    )
    
    for voltage, (voxel, world, n_voxels) in centroids_geom.items():
        print(f"  {voltage}V: World coords = ({world[0]:.2f}, {world[1]:.2f}, {world[2]:.2f}) mm, N_voxels = {n_voxels}")
    
    # Analyze stability of electrode location estimate across voltages
    print("\nElectrode Location Stability Analysis:")
    world_coords_array = np.array([world for _, world, _ in centroids_mass.values()])
    mean_location = np.mean(world_coords_array, axis=0)
    std_location = np.std(world_coords_array, axis=0)
    
    print(f"  Mean location: ({mean_location[0]:.2f}, {mean_location[1]:.2f}, {mean_location[2]:.2f}) mm")
    print(f"  Std deviation: ({std_location[0]:.2f}, {std_location[1]:.2f}, {std_location[2]:.2f}) mm")
    print(f"  Max deviation: {np.max(std_location):.2f} mm")
    
    return voltage_masks, centroids_mass, centroids_geom


def example_combined_mask():
    """
    Example 3: Combine voltage masks and calculate overall centroid.
    """
    print("\n" + "=" * 70)
    print("Example 3: Combined Voltage Mask for Overall Electrode Location")
    print("=" * 70)
    
    # Create synthetic masks for different voltages
    voltage_masks = []
    shape = (91, 109, 91)
    affine = np.diag([2, 2, 2, 1])
    
    base_center = np.array([45, 54, 45])
    
    for voltage in [2, 3, 4, 5, 6]:
        data = np.zeros(shape)
        sigma = 3 + voltage * 0.5
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    pos = np.array([i, j, k])
                    dist = np.linalg.norm(pos - base_center)
                    data[i, j, k] = voltage * np.exp(-(dist**2) / (2 * sigma**2))
        
        voltage_masks.append(nib.Nifti1Image(data, affine))
    
    # Combine masks using mean
    print("\nCombining masks using mean operation...")
    combined_mask = mask_analysis.combine_voltage_masks(voltage_masks, operation='mean')
    
    # Calculate centroid of combined mask
    print("\nCentroid of Combined Mask:")
    voxel_coords, world_coords, mass = mask_analysis.calculate_mass_centroid(
        combined_mask, threshold=0.1
    )
    print(f"  Voxel coordinates: ({voxel_coords[0]:.2f}, {voxel_coords[1]:.2f}, {voxel_coords[2]:.2f})")
    print(f"  World coordinates: ({world_coords[0]:.2f}, {world_coords[1]:.2f}, {world_coords[2]:.2f}) mm")
    print(f"  Total mass: {mass:.4f}")
    
    return combined_mask


def example_with_real_data():
    """
    Example 4: Use with real tractography data (if available).
    
    This example demonstrates how to integrate with the existing DTI workflow.
    """
    print("\n" + "=" * 70)
    print("Example 4: Integration with Real Tractography Data")
    print("=" * 70)
    
    if not DBSPACE_AVAILABLE:
        print("  Skipping - requires dbspace and access to tractography files.")
        return None
    
    try:
        Etrode_map = DTI.Etrode_map
        
        # Example patient and condition
        pt = '906'
        condit = 'OnT'
        side = 'L'
        
        # Load masks for different voltages
        voltage_masks = {}
        vrange = range(2, 7)  # 2V to 6V
        
        for volt in vrange:
            cntct = Etrode_map[condit][pt][0] + 1  # 0 index for 'L'
            dti_file = (
                f'/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/'
                f'MDT_DBS_2_7V_Tractography/DBS{pt}.{side}{cntct}.{volt}V.bin.nii.gz'
            )
            
            try:
                voltage_masks[volt] = image.smooth_img(dti_file, fwhm=1)
                print(f"  Loaded {volt}V mask for patient {pt}")
            except FileNotFoundError:
                print(f"  Warning: File not found for {volt}V")
                continue
        
        if len(voltage_masks) == 0:
            print("  No mask files found.")
            return None
        
        # Calculate centroids
        print(f"\nElectrode Location Estimates for Patient {pt} ({condit}, {side}):")
        centroids = mask_analysis.calculate_centroids_for_voltages(
            voltage_masks, method='mass', threshold=0.01
        )
        
        for voltage, (voxel, world, mass) in centroids.items():
            print(f"  {voltage}V: ({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f}) mm")
        
        # Calculate combined estimate
        combined_mask = mask_analysis.combine_voltage_masks(voltage_masks, operation='mean')
        voxel, world, mass = mask_analysis.calculate_mass_centroid(combined_mask, threshold=0.01)
        
        print(f"\nCombined Electrode Location Estimate:")
        print(f"  World coordinates: ({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f}) mm")
        
        return voltage_masks, centroids, combined_mask
        
    except Exception as e:
        print(f"  Error processing real data: {e}")
        return None


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Electrode Location Estimation using Centroid Methods")
    print("=" * 70)
    
    # Run examples
    mask_img = example_single_mask()
    voltage_masks, centroids_mass, centroids_geom = example_multiple_voltages()
    combined_mask = example_combined_mask()
    
    # Try with real data if available
    real_data_results = example_with_real_data()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
