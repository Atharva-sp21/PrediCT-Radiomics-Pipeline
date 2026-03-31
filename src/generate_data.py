import os
import numpy as np
import SimpleITK as sitk

def main():
    os.makedirs('data', exist_ok=True)
    for i in range(20):
        folder = os.path.join('data', f'patient_{i}')
        os.makedirs(folder, exist_ok=True)
        # CT image - 3D 20x20x20 random floats
        ct_arr = np.random.rand(20, 20, 20).astype(np.float32)
        ct_img = sitk.GetImageFromArray(ct_arr)
        
        # Mask image - 3D 20x20x20 uint8
        mask_arr = np.zeros((20, 20, 20), dtype=np.uint8)
        
        category = i % 3
        if category == 0:
            # Small: 3x3x3 = 27
            mask_arr[2:5, 2:5, 2:5] = 1
        elif category == 1:
            # Medium: 6x6x6 = 216
            mask_arr[2:8, 2:8, 2:8] = 1
        else:
            # Large: 12x12x12 = 1728
            mask_arr[2:14, 2:14, 2:14] = 1

        mask_img = sitk.GetImageFromArray(mask_arr)
        
        # Ensure direction and spacing are matching properly
        ct_img.SetOrigin((0.0, 0.0, 0.0))
        ct_img.SetSpacing((1.0, 1.0, 1.0))
        mask_img.SetOrigin((0.0, 0.0, 0.0))
        mask_img.SetSpacing((1.0, 1.0, 1.0))

        sitk.WriteImage(ct_img, os.path.join(folder, 'ct.nii.gz'))
        sitk.WriteImage(mask_img, os.path.join(folder, 'mask.nii.gz'))
    print("Dummy data generated.")

if __name__ == '__main__':
    main()
