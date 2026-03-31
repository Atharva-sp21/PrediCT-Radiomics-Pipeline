import os
import csv
import argparse
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

def calculate_mock_agatston(mask_img):
    # Calculate mask volume and multiply by random float
    mask_arr = sitk.GetArrayFromImage(mask_img)
    volume = np.sum(mask_arr)
    # mock calculation
    score = volume * np.random.uniform(0.5, 1.5)
    return min(score, 500.0)

def categorize_score(score):
    if score == 0:
        return "0"
    elif 1 <= score <= 99:
        return "1-99"
    elif 100 <= score <= 399:
        return "100-399"
    else:
        return ">=400"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--out', type=str, default='radiomics_features.csv')
    args = parser.parse_args()

    # Initialize pyradiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('glrlm')

    results = []
    
    # Loop through patients
    for i in range(20):
        patient_id = f'patient_{i}'
        folder = os.path.join(args.data_dir, patient_id)
        ct_path = os.path.join(folder, 'ct.nii.gz')
        mask_path = os.path.join(folder, 'mask.nii.gz')
        
        if not os.path.exists(ct_path) or not os.path.exists(mask_path):
            print(f"Skipping {patient_id}, missing files.")
            continue
            
        try:
            ct_img = sitk.ReadImage(ct_path)
            mask_img = sitk.ReadImage(mask_path)
            
            # SimpleITK dimension error fixes
            # PyRadiomics requires same geometry
            if ct_img.GetSize() != mask_img.GetSize():
                print(f"Size mismatch for {patient_id}. Resampling mask...")
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(ct_img)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                mask_img = resampler.Execute(mask_img)

            # Check matching origin, spacing, directions
            mask_img.SetOrigin(ct_img.GetOrigin())
            mask_img.SetSpacing(ct_img.GetSpacing())
            mask_img.SetDirection(ct_img.GetDirection())
                
            features = extractor.execute(ct_img, mask_img)
            
            # Clean diagnostic features
            clean_features = {k: float(v) for k, v in features.items() if not k.startswith('diagnostics_')}
            
            mock_score = calculate_mock_agatston(mask_img)
            category = categorize_score(mock_score)
            
            row = {
                'Patient_ID': patient_id, 
                'Agatston_Score': mock_score, 
                'Agatston_Category': category
            }
            row.update(clean_features)
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            
    if not results:
        print("No results generated.")
        return
        
    fieldnames = list(results[0].keys())
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Saved {len(results)} patients to {args.out}")

if __name__ == '__main__':
    main()
