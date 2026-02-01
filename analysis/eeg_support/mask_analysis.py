#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask Analysis Utilities for Electrode Location Estimation

This module provides functions to calculate centroids (mass-centroid and geometric centroid)
from voltage-specific tractography masks to estimate the exact location of electrodes.

@author: virati
"""

import numpy as np
import nibabel as nib
from nilearn import image


def calculate_mass_centroid(mask_img, threshold=0.0):
    """
    Calculate the mass-centroid (center of mass) of a mask image.
    
    The mass-centroid is the weighted average of voxel positions, where weights
    are the intensity values in the mask. This is useful for estimating electrode
    locations when the mask represents probabilistic tractography or activation volumes.
    
    Parameters
    ----------
    mask_img : Niimg-like object
        A 3D image (nibabel or nilearn image object) representing the mask.
        Can be a continuous-valued image (e.g., tractography probability map).
    threshold : float, optional
        Minimum intensity value to consider. Voxels below this threshold are 
        excluded from the calculation. Default is 0.0.
    
    Returns
    -------
    centroid_voxel : numpy.ndarray
        The (i, j, k) voxel coordinates of the mass-centroid.
    centroid_world : numpy.ndarray
        The (x, y, z) world coordinates of the mass-centroid in mm (using image affine).
    mass : float
        The total mass (sum of all intensity values above threshold).
    
    Examples
    --------
    >>> from nilearn import image
    >>> mask_img = image.load_img('tractography_mask.nii.gz')
    >>> voxel_coords, world_coords, mass = calculate_mass_centroid(mask_img)
    >>> print(f"Electrode location (voxel): {voxel_coords}")
    >>> print(f"Electrode location (mm): {world_coords}")
    """
    # Load the image data
    mask_data = image.get_data(mask_img)
    affine = mask_img.affine
    
    # Apply threshold
    mask_data = mask_data.copy()
    mask_data[mask_data < threshold] = 0
    
    # Get total mass
    mass = np.sum(mask_data)
    
    if mass == 0:
        raise ValueError("Total mass is zero. No voxels above threshold or empty mask.")
    
    # Create coordinate grids
    i_coords, j_coords, k_coords = np.meshgrid(
        np.arange(mask_data.shape[0]),
        np.arange(mask_data.shape[1]),
        np.arange(mask_data.shape[2]),
        indexing='ij'
    )
    
    # Calculate weighted centroid in voxel space
    centroid_i = np.sum(i_coords * mask_data) / mass
    centroid_j = np.sum(j_coords * mask_data) / mass
    centroid_k = np.sum(k_coords * mask_data) / mass
    
    centroid_voxel = np.array([centroid_i, centroid_j, centroid_k])
    
    # Convert to world coordinates using affine transformation
    centroid_world = nib.affines.apply_affine(affine, centroid_voxel)
    
    return centroid_voxel, centroid_world, mass


def calculate_geometric_centroid(mask_img, threshold=0.5):
    """
    Calculate the geometric centroid of a binary mask image.
    
    The geometric centroid is the mean position of all voxels in the mask,
    treating all voxels equally (unweighted). This is useful for binary masks
    where you want to find the center of the region of interest.
    
    Parameters
    ----------
    mask_img : Niimg-like object
        A 3D image (nibabel or nilearn image object) representing the mask.
    threshold : float, optional
        Threshold for binarizing the mask. Voxels with intensity >= threshold
        are considered part of the mask. Default is 0.5.
    
    Returns
    -------
    centroid_voxel : numpy.ndarray
        The (i, j, k) voxel coordinates of the geometric centroid.
    centroid_world : numpy.ndarray
        The (x, y, z) world coordinates of the geometric centroid in mm (using image affine).
    n_voxels : int
        The number of voxels in the binary mask.
    
    Examples
    --------
    >>> from nilearn import image
    >>> mask_img = image.load_img('electrode_mask.nii.gz')
    >>> voxel_coords, world_coords, n_voxels = calculate_geometric_centroid(mask_img)
    >>> print(f"Electrode location (voxel): {voxel_coords}")
    >>> print(f"Electrode location (mm): {world_coords}")
    >>> print(f"Region contains {n_voxels} voxels")
    """
    # Load the image data
    mask_data = image.get_data(mask_img)
    affine = mask_img.affine
    
    # Binarize the mask
    binary_mask = mask_data >= threshold
    
    # Count voxels
    n_voxels = np.sum(binary_mask)
    
    if n_voxels == 0:
        raise ValueError("No voxels above threshold. Empty mask.")
    
    # Get indices of all voxels in the mask
    i_indices, j_indices, k_indices = np.where(binary_mask)
    
    # Calculate mean position (geometric centroid)
    centroid_i = np.mean(i_indices)
    centroid_j = np.mean(j_indices)
    centroid_k = np.mean(k_indices)
    
    centroid_voxel = np.array([centroid_i, centroid_j, centroid_k])
    
    # Convert to world coordinates using affine transformation
    centroid_world = nib.affines.apply_affine(affine, centroid_voxel)
    
    return centroid_voxel, centroid_world, n_voxels


def calculate_centroids_for_voltages(voltage_masks, method='mass', **kwargs):
    """
    Calculate centroids for multiple voltage masks.
    
    This function processes a dictionary or list of voltage-specific masks and
    calculates the centroid for each, allowing analysis of how electrode location
    estimates vary across stimulation voltages.
    
    Parameters
    ----------
    voltage_masks : dict or list
        Dictionary mapping voltage values to mask images, or list of mask images.
        Each mask should be a Niimg-like object.
    method : str, optional
        Method for centroid calculation: 'mass' for mass-centroid or 
        'geometric' for geometric centroid. Default is 'mass'.
    **kwargs : dict
        Additional keyword arguments passed to the centroid calculation function
        (e.g., threshold).
    
    Returns
    -------
    centroids : dict or list
        Dictionary (or list) with the same structure as input, containing tuples
        of (centroid_voxel, centroid_world, metric) for each voltage.
    
    Examples
    --------
    >>> voltage_masks = {
    ...     2: image.load_img('mask_2V.nii.gz'),
    ...     3: image.load_img('mask_3V.nii.gz'),
    ...     4: image.load_img('mask_4V.nii.gz'),
    ... }
    >>> centroids = calculate_centroids_for_voltages(voltage_masks, method='mass')
    >>> for v, (vox, world, mass) in centroids.items():
    ...     print(f"{v}V: {world} mm (mass={mass:.2f})")
    """
    if method == 'mass':
        centroid_func = calculate_mass_centroid
    elif method == 'geometric':
        centroid_func = calculate_geometric_centroid
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mass' or 'geometric'.")
    
    # Process dictionary input
    if isinstance(voltage_masks, dict):
        centroids = {}
        for voltage, mask_img in voltage_masks.items():
            centroids[voltage] = centroid_func(mask_img, **kwargs)
        return centroids
    
    # Process list input
    elif isinstance(voltage_masks, (list, tuple)):
        centroids = []
        for mask_img in voltage_masks:
            centroids.append(centroid_func(mask_img, **kwargs))
        return centroids
    
    else:
        raise TypeError("voltage_masks must be a dict, list, or tuple.")


def combine_voltage_masks(voltage_masks, operation='mean'):
    """
    Combine multiple voltage masks into a single mask.
    
    This function aggregates voltage-specific masks to create a combined mask
    that represents the overall electrode location across all voltages.
    
    Parameters
    ----------
    voltage_masks : dict or list
        Dictionary mapping voltage values to mask images, or list of mask images.
        Each mask should be a Niimg-like object.
    operation : str, optional
        Operation to combine masks: 'mean' (average), 'sum' (addition), 
        'max' (maximum), or 'min' (minimum). Default is 'mean'.
    
    Returns
    -------
    combined_mask : Nifti1Image
        A single combined mask image.
    
    Examples
    --------
    >>> voltage_masks = [mask_2V, mask_3V, mask_4V]
    >>> combined = combine_voltage_masks(voltage_masks, operation='mean')
    >>> voxel, world, mass = calculate_mass_centroid(combined)
    """
    # Convert dict to list if needed
    if isinstance(voltage_masks, dict):
        mask_list = list(voltage_masks.values())
    else:
        mask_list = list(voltage_masks)
    
    if len(mask_list) == 0:
        raise ValueError("No masks provided.")
    
    # Create string for math_img operation
    img_refs = ','.join([f'img{i}' for i in range(len(mask_list))])
    img_dict = {f'img{i}': mask_list[i] for i in range(len(mask_list))}
    
    if operation == 'mean':
        formula = f"np.mean(np.array([{img_refs}]), axis=0)"
    elif operation == 'sum':
        formula = f"np.sum(np.array([{img_refs}]), axis=0)"
    elif operation == 'max':
        formula = f"np.max(np.array([{img_refs}]), axis=0)"
    elif operation == 'min':
        formula = f"np.min(np.array([{img_refs}]), axis=0)"
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'mean', 'sum', 'max', or 'min'.")
    
    combined_mask = image.math_img(formula, **img_dict)
    
    return combined_mask
