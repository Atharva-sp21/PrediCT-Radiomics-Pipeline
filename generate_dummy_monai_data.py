import os
import numpy as np
import SimpleITK as sitk

def main():
    dirs_to_make = [
        "data/imagesTr", "data/labelsTr", 
        "data/imagesTs", "data/labelsTs"
    ]
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)
        
    np.random.seed(42)
    
    # Generate 10 Train pairs and 4 Test pairs
    for i in range(14):
        mode = "Tr" if i < 10 else "Ts"
        img_dir = f"data/images{mode}"
        lbl_dir = f"data/labels{mode}"
        
        # Create tiny 32x32x32 random volumes for speed
        vol_size = (32, 32, 32)
        ct_arr = np.random.rand(*vol_size).astype(np.float32)
        
        # Box mask for "heart"
        mask_arr = np.zeros(vol_size, dtype=np.uint8)
        # Random location
        cx = np.random.randint(5, 25)
        cy = np.random.randint(5, 25)
        cz = np.random.randint(5, 25)
        mask_arr[cx-4:cx+4, cy-4:cy+4, cz-4:cz+4] = 1
        
        ct_img = sitk.GetImageFromArray(ct_arr)
        mask_img = sitk.GetImageFromArray(mask_arr)
        
        sitk.WriteImage(ct_img, os.path.join(img_dir, f"scan_{i}.nii.gz"))
        sitk.WriteImage(mask_img, os.path.join(lbl_dir, f"scan_{i}.nii.gz"))
        
    print("Dummy MONAI data properly structured.")

if __name__ == "__main__":
    main()
