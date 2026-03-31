import os
import time
import SimpleITK as sitk

def register_ccta_to_ncct(fixed_image_path, moving_image_path, moving_centerline_path, output_dir):
    """
    Registers an ImageCAS CCTA (moving) to a COCA non-contrast CT (fixed)
    using a multi-resolution Rigid/Affine + B-Spline strategy with Mutual Information.
    Transforms the moving centerlines to the fixed space.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting registration...\nFixed: {fixed_image_path}\nMoving: {moving_image_path}")
    start_time = time.time()
    
    # Load images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    moving_centerline = sitk.ReadImage(moving_centerline_path, sitk.sitkUInt8)
    
    # 1. Initialize Transform (Alignment of centers)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image, 
        sitk.AffineTransform(3), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # 2. Affine Registration (handles patient positioning)
    registration_method = sitk.ImageRegistrationMethod()
    
    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Mattes Mutual Information for Cross-Modality
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=100, 
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    print("Running Affine Registration...")
    affine_transform = registration_method.Execute(fixed_image, moving_image)
    
    # 3. B-Spline Deformable Registration (handles cardiac motion/structural differences)
    transform_domain_mesh_size = [8] * fixed_image.GetDimension()
    bspline_initial = sitk.BSplineTransformInitializer(fixed_image, transform_domain_mesh_size)
    
    registration_method.SetInitialTransform(bspline_initial, inPlace=False)
    registration_method.SetMovingInitialTransform(affine_transform)
    
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=50,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e+7
    )
    
    print("Running B-Spline Deformable Registration...")
    bspline_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Combine transforms
    final_transform = sitk.CompositeTransform(affine_transform)
    final_transform.AddTransform(bspline_transform)
    
    end_time = time.time()
    registration_time = end_time - start_time
    print(f"Registration Complete. Total Time: {registration_time:.2f} seconds.")
    
    # 4. Apply transformation to moving centerline
    print("Transforming centerlines to fixed space...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # NN for labels/centerlines
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    
    transformed_centerline = resampler.Execute(moving_centerline)
    
    # Save transformed centerlines
    output_path = os.path.join(output_dir, "transformed_centerline.nii.gz")
    sitk.WriteImage(transformed_centerline, output_path)
    
    # Save final transform
    sitk.WriteTransform(final_transform, os.path.join(output_dir, "final_transform.tfm"))
    
    print(f"Outputs saved to {output_dir}")
    return transformed_centerline, registration_time

if __name__ == "__main__":
    # Dummy execution block for testing
    print("To run: register_ccta_to_ncct('ncct.nii.gz', 'ccta.nii.gz', 'ccta_centerline.nii.gz', 'output')")
