import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def validate_registration(fixed_ncct_path, transformed_centerline_path):
    """
    Validates the registration by checking the percentage of high-density 
    calcium voxels (HU > 130) within a 10mm radius of the transformed centerlines.
    """
    ncct_img = sitk.ReadImage(fixed_ncct_path, sitk.sitkFloat32)
    centerline_img = sitk.ReadImage(transformed_centerline_path, sitk.sitkUInt8)
    
    ncct_arr = sitk.GetArrayFromImage(ncct_img)
    centerline_arr = sitk.GetArrayFromImage(centerline_img)
    
    spacing = ncct_img.GetSpacing()  # (x, y, z) in mm
    
    # 1. Create a 10mm distance dilation mask around the centerline
    # Convert 10mm radius to voxel units for each dimension.
    # Note: sitk GetSpacing is (x, y, z) but numpy array is (z, y, x)
    voxel_radius = (
        int(round(10.0 / spacing[2])),
        int(round(10.0 / spacing[1])),
        int(round(10.0 / spacing[0]))
    )
    
    # Create the structuring element for 10mm dilation
    structuring_element = generate_ellipsoid(voxel_radius)
    dilated_centerline = ndimage.binary_dilation(centerline_arr > 0, structure=structuring_element)
    
    # 2. Identify calcium voxels (HU > 130) within the non-contrast GT 
    # (Assuming we evaluate against the entire non-contrast volume's calcifications)
    calcium_mask = ncct_arr > 130
    total_calcium_voxels = np.sum(calcium_mask)
    
    # 3. Calculate overlap
    captured_calcium = np.logical_and(calcium_mask, dilated_centerline)
    captured_calcium_count = np.sum(captured_calcium)
    
    if total_calcium_voxels == 0:
        print("No calcium found in the non-contrast scan.")
        percentage = 0.0
    else:
        percentage = (captured_calcium_count / total_calcium_voxels) * 100.0
        
    print(f"Total Calcium Voxels: {total_calcium_voxels}")
    print(f"Captured Calcium (<10mm): {captured_calcium_count}")
    print(f"Capture Percentage: {percentage:.2f}%")
    
    if percentage > 70.0:
        print("Target validation (>70%) achieved!")
    else:
        print("Target validation (>70%) NOT achieved.")
        
    # Generate Visual Overlay
    generate_visual_overlay(ncct_arr, dilated_centerline, captured_calcium)
    return percentage

def generate_ellipsoid(radius):
    """Generates an ellipsoid structuring element for dilation."""
    z, y, x = np.ogrid[-radius[0]:radius[0]+1, -radius[1]:radius[1]+1, -radius[2]:radius[2]+1]
    mask = ((x**2)/(radius[2]**2 if radius[2]>0 else 1) + 
            (y**2)/(radius[1]**2 if radius[1]>0 else 1) + 
            (z**2)/(radius[0]**2 if radius[0]>0 else 1)) <= 1
    return mask

def generate_visual_overlay(ncct_arr, dilated_centerline, captured_calcium):
    """Generates a matplotlib 2D slice overlay for validation."""
    # Find the slice with the maximum captured calcium to display
    z_counts = np.sum(captured_calcium, axis=(1, 2))
    if np.sum(z_counts) == 0:
        best_z = ncct_arr.shape[0] // 2
    else:
        best_z = np.argmax(z_counts)
        
    plt.figure(figsize=(10, 10))
    plt.imshow(ncct_arr[best_z], cmap='gray', vmin=-100, vmax=400)
    
    # Overlay the dilated centerline region in blue
    mask_slice = dilated_centerline[best_z]
    overlay_blue = np.zeros((*mask_slice.shape, 4))
    overlay_blue[mask_slice, :] = [0, 0, 1, 0.3]  # RGBA
    plt.imshow(overlay_blue)
    
    # Overlay the captured calcium in red
    calc_slice = captured_calcium[best_z]
    overlay_red = np.zeros((*calc_slice.shape, 4))
    overlay_red[calc_slice, :] = [1, 0, 0, 0.8]  # RGBA
    plt.imshow(overlay_red)
    
    plt.title(f'Axial Slice {best_z}: NCCT (Gray), 10mm Radius (Blue), Captured Calcium (Red)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('registration_overlay.png', dpi=300)
    print("Saved visual overlay to registration_overlay.png")

if __name__ == "__main__":
    print("To run: validate_registration('ncct.nii.gz', 'output/transformed_centerline.nii.gz')")
