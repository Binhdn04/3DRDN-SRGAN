import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import nibabel as nib
import os
from PIL import Image
from model3 import Model3DRDN  
import SimpleITK as sitk
from patchify import patchify, unpatchify
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from scipy import ndimage
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

PATCH_SIZE           = 40
GAUSSIAN_NOISE       = 0.25
SCALING_FACTOR       = 2
interpolation_method = 'bicubic'

def gaussian_blur_3d(image, sigma=1.0):
    image_sitk = sitk.GetImageFromArray(image)
    blurred_sitk = sitk.DiscreteGaussian(image_sitk, sigma)
    return sitk.GetArrayFromImage(blurred_sitk)

def get_low_res(hr_image, interpolation_order=3):
    hr_image = gaussian_blur_3d(hr_image, sigma=GAUSSIAN_NOISE)
    scale_factors = [1.0 / SCALING_FACTOR] * 3
    lr_image = zoom(hr_image, zoom=scale_factors, order=interpolation_order)
    ups_lr_image = zoom(lr_image, zoom=[SCALING_FACTOR]*3, order=interpolation_order)
    ups_lr_image = ups_lr_image.astype(np.int32)

    return ups_lr_image


def process_image_with_average_mode(img_data, generator, patch_size=(40, 40, 40), step_size=10, batch_size=4, use_tta=True):
    img_data_shape = img_data.shape
    
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    if isinstance(step_size, int):
        step_size = (step_size, step_size, step_size)
        
    img_min, img_max = np.min(img_data), np.max(img_data)
    img_data_norm = (img_data - img_min) / (img_max - img_min)
    
    patches = patchify(img_data_norm, patch_size, step=step_size)
    n_patches_x, n_patches_y, n_patches_z, patch_x, patch_y, patch_z = patches.shape
    
    result_image = np.zeros(img_data_shape)
    weight_sum = np.zeros(img_data_shape)
    
    all_patches = []
    all_positions = []
    
    for i in range(n_patches_x):
        for j in range(n_patches_y):
            for k in range(n_patches_z):
                patch = patches[i, j, k]
                all_patches.append(patch)
                
                position = (i * step_size[0], j * step_size[1], k * step_size[2])
                all_positions.append(position)
    
    n_patches = len(all_patches)
    for batch_idx in range(0, n_patches, batch_size):
        batch_end = min(batch_idx + batch_size, n_patches)
        batch_patches = all_patches[batch_idx:batch_end]
        batch_positions = all_positions[batch_idx:batch_end]
        
        for idx, (patch, position) in enumerate(zip(batch_patches, batch_positions)):
            x_start, y_start, z_start = position
            
            if use_tta:
                output_patch = tta_inference(generator, patch)
            else:
                patch_input = np.expand_dims(patch, axis=(0, -1))
                output_patch = generator(patch_input, training=False).numpy()[0, ..., 0]
            
            result_image[x_start:x_start+patch_x, 
                        y_start:y_start+patch_y, 
                        z_start:z_start+patch_z] += output_patch
                
            weight_sum[x_start:x_start+patch_x, 
                      y_start:y_start+patch_y, 
                      z_start:z_start+patch_z] += 1
    
    mask = weight_sum > 0
    output_image = np.zeros_like(result_image)
    output_image[mask] = result_image[mask] / weight_sum[mask]
    
    output_image = np.clip(output_image, 0.0, 1.0)
    
    output_image = output_image * (img_max - img_min) + img_min
    
    return output_image

def tta_inference(generator, img_patch):

    aug_list = [
        lambda x: x,                             
        lambda x: np.flip(x, axis=0),            
        lambda x: np.flip(x, axis=1),             
        lambda x: np.flip(x, axis=2),             
        lambda x: np.rot90(x, k=1, axes=(0,1)),   
        lambda x: np.rot90(x, k=1, axes=(1,2)),  
        lambda x: np.rot90(x, k=1, axes=(0,2)),   
    ]
    
    outputs = []
    for aug in aug_list:
        aug_input = aug(img_patch)
        aug_input = np.expand_dims(aug_input, axis=(0, -1))
        aug_output = generator(aug_input, training=False).numpy()[0, ..., 0]
        
        inv_aug_output = np.copy(aug_output)
        if aug == aug_list[1]:
            inv_aug_output = np.flip(inv_aug_output, axis=0)
        elif aug == aug_list[2]:
            inv_aug_output = np.flip(inv_aug_output, axis=1)
        elif aug == aug_list[3]:
            inv_aug_output = np.flip(inv_aug_output, axis=2)
        elif aug == aug_list[4]:
            inv_aug_output = np.rot90(inv_aug_output, k=-1, axes=(0,1))
        elif aug == aug_list[5]:
            inv_aug_output = np.rot90(inv_aug_output, k=-1, axes=(1,2))
        elif aug == aug_list[6]:
            inv_aug_output = np.rot90(inv_aug_output, k=-1, axes=(0,2))
        
        outputs.append(inv_aug_output)
    
    return np.mean(outputs, axis=0)

def process_image_with_hann_window(img_data, generator, patch_size=40, step_size=10, 
                                 batch_size=4, use_tta=True):

    img_shape = img_data.shape
    
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    if isinstance(step_size, int):
        step_size = (step_size, step_size, step_size)
    
    img_min, img_max = np.min(img_data), np.max(img_data)
    img_data_norm = (img_data - img_min) / (img_max - img_min)
    
    hann_x = np.hanning(patch_size[0])
    hann_y = np.hanning(patch_size[1])
    hann_z = np.hanning(patch_size[2])
    hann_window = np.outer(hann_x, hann_y).reshape(patch_size[0], patch_size[1], 1)
    hann_window = hann_window * hann_z.reshape(1, 1, patch_size[2])
    
    result_image = np.zeros(img_shape, dtype=np.float32)
    weight_sum = np.zeros(img_shape, dtype=np.float32)
    
    patches = patchify(img_data_norm, patch_size, step=step_size)
    n_patches_x, n_patches_y, n_patches_z = patches.shape[:3]
    
    all_patches = []
    all_positions = []
    
    for i in range(n_patches_x):
        for j in range(n_patches_y):
            for k in range(n_patches_z):
                patch = patches[i, j, k]
                all_patches.append(patch)
                
                position = (i * step_size[0], j * step_size[1], k * step_size[2])
                all_positions.append(position)
    
    n_patches = len(all_patches)
    for batch_idx in range(0, n_patches, batch_size):
        batch_end = min(batch_idx + batch_size, n_patches)
        batch_patches = all_patches[batch_idx:batch_end]
        batch_positions = all_positions[batch_idx:batch_end]
        
        for idx, (patch, position) in enumerate(zip(batch_patches, batch_positions)):
            x_start, y_start, z_start = position
            
            if use_tta:
                output_patch = tta_inference(generator, patch)
            else:
                patch_input = np.expand_dims(patch, axis=(0, -1))
                output_patch = generator(patch_input, training=False).numpy()[0, ..., 0]
            
            weighted_patch = output_patch * hann_window
            
            result_image[x_start:x_start+patch_size[0], 
                        y_start:y_start+patch_size[1], 
                        z_start:z_start+patch_size[2]] += weighted_patch
                
            weight_sum[x_start:x_start+patch_size[0], 
                      y_start:y_start+patch_size[1], 
                      z_start:z_start+patch_size[2]] += hann_window
    
    weight_sum = np.where(weight_sum == 0, 1, weight_sum)
    
    output_image = result_image / weight_sum
    
    
    output_image = np.clip(output_image, 0.0, 1.0)
    
    output_image = output_image * (img_max - img_min) + img_min
    print(output_image.shape)
    return output_image


def gaussian_window_3d(shape, sigma=0.5):
    grids = [np.linspace(-1, 1, num=s) for s in shape]
    z, y, x = np.meshgrid(grids[0], grids[1], grids[2], indexing='ij')
    g = np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    return g / np.max(g)

def process_image_with_gaussian_window(img_data, generator, patch_size=40, step_size=10, 
                                       batch_size=4, use_tta=True, sigma=0.5):
    img_shape = img_data.shape
    
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    if isinstance(step_size, int):
        step_size = (step_size, step_size, step_size)
    
    img_min, img_max = np.min(img_data), np.max(img_data)
    img_data_norm = (img_data - img_min) / (img_max - img_min)
    
    gauss_window = gaussian_window_3d(patch_size, sigma=sigma)
    
    result_image = np.zeros(img_shape, dtype=np.float32)
    weight_sum = np.zeros(img_shape, dtype=np.float32)
    
    patches = patchify(img_data_norm, patch_size, step=step_size)
    n_patches_x, n_patches_y, n_patches_z = patches.shape[:3]
    
    all_patches = []
    all_positions = []
    for i in range(n_patches_x):
        for j in range(n_patches_y):
            for k in range(n_patches_z):
                patch = patches[i, j, k]
                all_patches.append(patch)
                position = (i * step_size[0], j * step_size[1], k * step_size[2])
                all_positions.append(position)
    
    n_patches = len(all_patches)
    for batch_idx in range(0, n_patches, batch_size):
        batch_end = min(batch_idx + batch_size, n_patches)
        batch_patches = all_patches[batch_idx:batch_end]
        batch_positions = all_positions[batch_idx:batch_end]
        
        for patch, position in zip(batch_patches, batch_positions):
            x_start, y_start, z_start = position
            
            if use_tta:
                output_patch = tta_inference(generator, patch)
            else:
                patch_input = np.expand_dims(patch, axis=(0, -1))
                output_patch = generator(patch_input, training=False).numpy()[0, ..., 0]
            
            weighted_patch = output_patch * gauss_window
            
            result_image[x_start:x_start+patch_size[0], 
                         y_start:y_start+patch_size[1], 
                         z_start:z_start+patch_size[2]] += weighted_patch
            
            weight_sum[x_start:x_start+patch_size[0], 
                       y_start:y_start+patch_size[1], 
                       z_start:z_start+patch_size[2]] += gauss_window
    
    weight_sum = np.where(weight_sum == 0, 1, weight_sum)
    
    output_image = result_image / weight_sum
    output_image = np.clip(output_image, 0.0, 1.0)
    
    output_image = output_image * (img_max - img_min) + img_min
    
    return output_image


def super_resolve_nii(input_nii_path, output_dir="inference_images/3D"):

    model = Model3DRDN(MODEL="3DRDN")
    generator = model.generator_g


    nii_img = nib.load(input_nii_path)
    img_data = nii_img.get_fdata()
    
    img_data_lr = get_low_res(img_data)
    print(img_data_lr.shape)
    
    lr_filename = os.path.basename(input_nii_path).replace(".nii.gz", "_lr.nii.gz")
    lr_nii_path = os.path.join(output_dir, lr_filename)
    os.makedirs(output_dir, exist_ok=True)
    lr_nii = nib.Nifti1Image(img_data_lr, affine=nii_img.affine, header=nii_img.header)
    nib.save(lr_nii, lr_nii_path)
    print(f"Low-resolution image saved at: {lr_nii_path}")

    save_slices(lr_nii_path)
    
    output_image = process_image_with_hann_window(img_data_lr, generator, use_tta=True)

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_nii_path).replace(".nii.gz", "_output.nii.gz")
    output_nii_path = os.path.join(output_dir, filename)


    output_nii = nib.Nifti1Image(output_image, affine=nii_img.affine, header=nii_img.header)
    nib.save(output_nii, output_nii_path)

    print(f"Processed image saved at: {output_nii_path}")
    return output_nii_path

def save_slices(output_file):

    output_nii = nib.load(output_file)
    output_data = output_nii.get_fdata()

    output_dir = os.path.dirname(output_file.replace('3D', '2D'))
    prefix = os.path.basename(output_file).replace(".nii.gz", "")

    x_mid = output_data.shape[0] // 2
    y_mid = output_data.shape[1] // 2
    z_mid = output_data.shape[2] // 2

    slices = {
        "axial": np.rot90(output_data[:, :, z_mid], k=1),  
        "coronal": np.rot90(output_data[:, y_mid, :].T, k=2),  
        "sagittal": np.fliplr(np.rot90(np.flipud(np.rot90(output_data[x_mid, :, :].T, k=2)), k=2))
    }
    for plane, img_slice in slices.items():
        plt.figure(figsize=(4, 4))
        plt.imshow(img_slice, cmap="gray")
        plt.axis("off")
        save_path = os.path.join(output_dir, f"{prefix}_{plane}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()



if __name__ == "__main__":
    input_file = ""
    start = time.perf_counter()
    output_file = super_resolve_nii(input_file)
    end = time.perf_counter()
    print(f"Processing time: {end - start:.2f} seconds")  
    save_slices(output_file)