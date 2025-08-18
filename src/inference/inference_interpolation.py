import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from scipy.ndimage import zoom
import SimpleITK as sitk
from patchify import patchify, unpatchify
from scipy import ndimage

PATCH_SIZE = 40
GAUSSIAN_NOISE = 0.25
SCALING_FACTOR = 2

def gaussian_blur_3d(image, sigma=1.0):
    image_sitk = sitk.GetImageFromArray(image)
    blurred_sitk = sitk.DiscreteGaussian(image_sitk, sigma)
    return sitk.GetArrayFromImage(blurred_sitk)

def get_low_res(hr_image, interpolation_order=3):
    hr_image = gaussian_blur_3d(hr_image, sigma=GAUSSIAN_NOISE)
    scale_factors = [1.0 / SCALING_FACTOR] * 3
    lr_image = zoom(hr_image, zoom=scale_factors, order=interpolation_order)
    
    return lr_image

def up_sample_3d(lr_image, method='bicubic'):
    method_to_order = {
        'nearest': 0,
        'linear': 1,
        'cubic': 3,
        'spline5': 5  
    }
    

    if method.startswith('sitk_') or method == 'lanczos':
        interpolator = method.replace('sitk_', '')
        return sitk_upsample(lr_image, scale_factor=SCALING_FACTOR, interpolator=interpolator)
    

    order = method_to_order.get(method, 3)  
    ups_lr_image = zoom(lr_image, zoom=[SCALING_FACTOR]*3, order=order)
    
    return ups_lr_image

def sitk_upsample(input_image, scale_factor=2, interpolator='lanczos'):

    interpolator_map = {
        'lanczos': sitk.sitkLanczosWindowedSinc,
        'cubic': sitk.sitkBSpline,  
        'linear': sitk.sitkLinear,
        'nearest': sitk.sitkNearestNeighbor
    }

    sitk_image = sitk.GetImageFromArray(input_image)
    size = sitk_image.GetSize()
    spacing = sitk_image.GetSpacing()

    new_size = [int(size[0] * scale_factor), 
                int(size[1] * scale_factor), 
                int(size[2] * scale_factor)]
    
    new_spacing = [spacing[0] / scale_factor,
                   spacing[1] / scale_factor,
                   spacing[2] / scale_factor]

    resample = sitk.ResampleImageFilter()

    sitk_interpolator = interpolator_map.get(interpolator, sitk.sitkBSpline)
    resample.SetInterpolator(sitk_interpolator)
    

    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetOutputDirection(sitk_image.GetDirection())

    output_sitk_image = resample.Execute(sitk_image)
    output_image = sitk.GetArrayFromImage(output_sitk_image)
    
    return output_image

def save_slices(nii_path, description=""):
    img = nib.load(nii_path)
    data = img.get_fdata()

    output_dir = os.path.dirname(nii_path.replace('3D', '2D'))
    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.basename(nii_path)

    prefix = base_filename.replace(".nii.gz", "")

    methods = ['nearest', 'linear', 'cubic', 'spline5', 'lanczos', 
            'sitk_cubic', 'sitk_nearest', 'sitk_linear']

    method_found = None
    for method in methods:
        if method in prefix:
            method_found = method
            parts = prefix.split(f"_{method}")
            prefix = parts[0] + f"_{method}"
            break
            
    x_mid = data.shape[0] // 2
    y_mid = data.shape[1] // 2
    z_mid = data.shape[2] // 2

    slices = {
        "axial": np.rot90(data[:, :, z_mid], k=1),  
        "coronal": np.rot90(data[:, y_mid, :].T, k=2),  
        "sagittal": np.fliplr(np.rot90(np.flipud(np.rot90(data[x_mid, :, :].T, k=2)), k=2))
    }

    for plane, img_slice in slices.items():
        plt.figure(figsize=(4, 4))
        plt.imshow(img_slice, cmap="gray")
        plt.axis("off")
        save_path = os.path.join(output_dir, f"{prefix}_{plane}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        
    print(f"Saved slices of {description} at: {output_dir}")

def super_resolve_nii_with_interpolation(input_nii_path, output_dir="inference_images/3D"):
    print(f"Processing file: {input_nii_path}")
    nii_img = nib.load(input_nii_path)
    img_data = nii_img.get_fdata()
    
    lr_image = get_low_res(img_data)
    print(f"Original image shape: {img_data.shape}")
    print(f"Low-resolution image shape: {lr_image.shape}")

    os.makedirs(output_dir, exist_ok=True)

    methods = ['nearest', 'linear', 'cubic', 'spline5', 'lanczos', 
            'sitk_cubic', 'sitk_nearest', 'sitk_linear']

    output_paths = {}
    
    for method in methods:
        print(f"Processing with method: {method}...")
        output_image = up_sample_3d(lr_image, method=method)

        method_filename = os.path.basename(input_nii_path).replace(".nii.gz", f"_{method}.nii.gz")
        method_nii_path = os.path.join(output_dir, method_filename)

        output_nii = nib.Nifti1Image(output_image, affine=nii_img.affine, header=nii_img.header)
        nib.save(output_nii, method_nii_path)
        
        print(f"{method} image has been saved at: {method_nii_path}")
        output_paths[method] = method_nii_path
        save_slices(method_nii_path, description=method)
    
    return output_paths

if __name__ == "__main__":
    input_file = ""
    output_dir = ""
    output_paths = super_resolve_nii_with_interpolation(input_file, output_dir=output_dir)