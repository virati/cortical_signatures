#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for mask_analysis module

Tests the centroid calculation functions with synthetic data.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import nibabel as nib
from nilearn import image
from analysis.eeg_support import mask_analysis


def test_mass_centroid():
    """Test mass-centroid calculation with a synthetic Gaussian blob."""
    print("Testing mass-centroid calculation...")
    
    # Create a simple mask with known centroid
    shape = (50, 50, 50)
    data = np.zeros(shape)
    
    # Create a Gaussian blob centered at (25, 25, 25)
    center = np.array([25.0, 25.0, 25.0])
    sigma = 3.0
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                pos = np.array([i, j, k])
                dist = np.linalg.norm(pos - center)
                data[i, j, k] = np.exp(-(dist**2) / (2 * sigma**2))
    
    # Create a NIfTI image with identity affine
    affine = np.eye(4)
    mask_img = nib.Nifti1Image(data, affine)
    
    # Calculate mass-centroid
    voxel_coords, world_coords, mass = mask_analysis.calculate_mass_centroid(mask_img, threshold=0.01)
    
    # Check that centroid is close to expected center
    assert np.allclose(voxel_coords, center, atol=0.5), f"Expected {center}, got {voxel_coords}"
    assert mass > 0, "Mass should be positive"
    
    print(f"  ✓ Mass-centroid: {voxel_coords} (expected ~{center})")
    print(f"  ✓ Mass: {mass:.4f}")
    return True


def test_geometric_centroid():
    """Test geometric centroid calculation with a synthetic binary mask."""
    print("\nTesting geometric centroid calculation...")
    
    # Create a simple binary mask
    shape = (50, 50, 50)
    data = np.zeros(shape)
    
    # Create a sphere of radius 5 centered at (25, 25, 25)
    center = np.array([25.0, 25.0, 25.0])
    radius = 5.0
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                pos = np.array([i, j, k])
                dist = np.linalg.norm(pos - center)
                if dist <= radius:
                    data[i, j, k] = 1.0
    
    # Create a NIfTI image with identity affine
    affine = np.eye(4)
    mask_img = nib.Nifti1Image(data, affine)
    
    # Calculate geometric centroid
    voxel_coords, world_coords, n_voxels = mask_analysis.calculate_geometric_centroid(mask_img, threshold=0.5)
    
    # Check that centroid is close to expected center
    assert np.allclose(voxel_coords, center, atol=0.5), f"Expected {center}, got {voxel_coords}"
    assert n_voxels > 0, "Should have positive number of voxels"
    
    print(f"  ✓ Geometric centroid: {voxel_coords} (expected ~{center})")
    print(f"  ✓ Number of voxels: {n_voxels}")
    return True


def test_multiple_voltages():
    """Test centroid calculation for multiple voltage masks."""
    print("\nTesting multiple voltage mask processing...")
    
    # Create synthetic masks for different voltages
    voltage_masks = {}
    shape = (50, 50, 50)
    affine = np.eye(4)
    center = np.array([25.0, 25.0, 25.0])
    
    for voltage in [2, 3, 4, 5]:
        data = np.zeros(shape)
        sigma = 2 + voltage * 0.3  # Increasing spread with voltage
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    pos = np.array([i, j, k])
                    dist = np.linalg.norm(pos - center)
                    data[i, j, k] = voltage * np.exp(-(dist**2) / (2 * sigma**2))
        
        voltage_masks[voltage] = nib.Nifti1Image(data, affine)
    
    # Calculate centroids for all voltages
    centroids = mask_analysis.calculate_centroids_for_voltages(
        voltage_masks, method='mass', threshold=0.01
    )
    
    assert len(centroids) == 4, "Should have centroids for 4 voltages"
    
    for voltage, (voxel, world, mass) in centroids.items():
        assert np.allclose(voxel, center, atol=1.0), f"Centroid for {voltage}V too far from center"
        assert mass > 0, f"Mass for {voltage}V should be positive"
        print(f"  ✓ {voltage}V: centroid={voxel}, mass={mass:.2f}")
    
    return True


def test_combine_masks():
    """Test combining multiple voltage masks."""
    print("\nTesting mask combination...")
    
    # Create synthetic masks
    voltage_masks = []
    shape = (50, 50, 50)
    affine = np.eye(4)
    center = np.array([25.0, 25.0, 25.0])
    
    for voltage in [2, 3, 4]:
        data = np.zeros(shape)
        sigma = 3
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    pos = np.array([i, j, k])
                    dist = np.linalg.norm(pos - center)
                    data[i, j, k] = voltage * np.exp(-(dist**2) / (2 * sigma**2))
        
        voltage_masks.append(nib.Nifti1Image(data, affine))
    
    # Combine masks
    combined = mask_analysis.combine_voltage_masks(voltage_masks, operation='mean')
    
    # Calculate centroid of combined mask
    voxel_coords, world_coords, mass = mask_analysis.calculate_mass_centroid(combined, threshold=0.01)
    
    assert np.allclose(voxel_coords, center, atol=0.5), "Combined centroid should be at center"
    assert mass > 0, "Combined mass should be positive"
    
    print(f"  ✓ Combined centroid: {voxel_coords}")
    print(f"  ✓ Combined mass: {mass:.4f}")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    # Test empty mask (should raise error)
    shape = (50, 50, 50)
    data = np.zeros(shape)
    affine = np.eye(4)
    empty_mask = nib.Nifti1Image(data, affine)
    
    try:
        mask_analysis.calculate_mass_centroid(empty_mask, threshold=0.1)
        print("  ✗ Should have raised error for empty mask")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly raised error for empty mask: {e}")
    
    # Test single-voxel mask
    data = np.zeros(shape)
    data[25, 25, 25] = 1.0
    single_voxel_mask = nib.Nifti1Image(data, affine)
    
    voxel_coords, world_coords, mass = mask_analysis.calculate_mass_centroid(single_voxel_mask, threshold=0.5)
    assert np.allclose(voxel_coords, [25, 25, 25]), "Single voxel centroid incorrect"
    print(f"  ✓ Single-voxel centroid: {voxel_coords}")
    
    return True


if __name__ == '__main__':
    print("=" * 70)
    print("Running mask_analysis test suite")
    print("=" * 70)
    
    tests = [
        test_mass_centroid,
        test_geometric_centroid,
        test_multiple_voltages,
        test_combine_masks,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
