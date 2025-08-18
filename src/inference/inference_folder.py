import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import glob
from PIL import Image
from model import Model3DRDN  
import SimpleITK as sitk
from patchify import patchify, unpatchify
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from scipy import ndimage
import onnxruntime as ort

PATCH_SIZE = 40

def process_image_with_patches(img_data, generator, patch_size=(40, 40, 40), step_size=10, 
                              batch_size=4, use_tta=True, mode="average"):
    img_data_shape = img_data.shape
    
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    if isinstance(step_size, int):
        step_size = (step_size, step_size, step_size)
        
    img_min, img_max = np.min(img_data), np.max(img_data)
    img_data_norm = (img_data - img_min) / (img_max - img_min)
    
    if mode == "hann":
        hann_x = np.hanning(patch_size[0])
        hann_y = np.hanning(patch_size[1])
        hann_z = np.hanning(patch_size[2])
        hann_window = np.outer(hann_x, hann_y).reshape(patch_size[0], patch_size[1], 1)
        hann_window = hann_window * hann_z.reshape(1, 1, patch_size[2])
    
    patches = patchify(img_data_norm, patch_size, step=step_size)
    n_patches_x, n_patches_y, n_patches_z, patch_x, patch_y, patch_z = patches.shape
    
    result_image = np.zeros(img_data_shape, dtype=np.float32)
    weight_sum = np.zeros(img_data_shape, dtype=np.float32)

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
                input_dict = {generator.get_inputs()[0].name: patch_input.astype(np.float64)}
                output_patch = generator.run([generator.get_outputs()[0].name], input_dict)[0][0, ..., 0]
            
            if mode == "hann":
                weighted_patch = output_patch * hann_window
                weight_contribution = hann_window
            else:  
                weighted_patch = output_patch
                weight_contribution = np.ones_like(output_patch)
    
            result_image[x_start:x_start+patch_x, 
                        y_start:y_start+patch_y, 
                        z_start:z_start+patch_z] += weighted_patch
                
            weight_sum[x_start:x_start+patch_x, 
                      y_start:y_start+patch_y, 
                      z_start:z_start+patch_z] += weight_contribution

    if mode == "hann":
        weight_sum = np.where(weight_sum == 0, 1, weight_sum)
    else:  
        mask = weight_sum > 0
        output_image = np.zeros_like(result_image)
        output_image[mask] = result_image[mask] / weight_sum[mask]
        result_image = output_image
        weight_sum = np.ones_like(weight_sum)
    
    output_image = result_image / weight_sum
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
        # Predict
        input_dict = {generator.get_inputs()[0].name: aug_input.astype(np.float64)}
        aug_output = generator.run([generator.get_outputs()[0].name], input_dict)[0][0, ..., 0]
        
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

def super_resolve_nii(input_nii_path, output_dir="inference_images/3D", mode="hann"):
    onnx_model_path = ""
    generator = ort.InferenceSession(
        onnx_model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    nii_img = nib.load(input_nii_path)
    img_data = nii_img.get_fdata()

    output_image = process_image_with_patches(img_data, generator, use_tta=True, mode=mode)

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_nii_path).replace(".nii.gz", "_output.nii.gz")
    output_nii_path = os.path.join(output_dir, filename)

    output_nii = nib.Nifti1Image(output_image, affine=nii_img.affine, header=nii_img.header)
    nib.save(output_nii, output_nii_path)

    print(f"✅ Processed image saved at: {output_nii_path}")
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

def process_all_nii_files(input_dir, output_dir="inference_images/3D", mode="hann"):
    nii_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    
    if not nii_files:
        print(f"No .nii.gz files found in directory: {input_dir}")
        return
    
    print(f"Found {len(nii_files)} .nii.gz files to process")
    
    for i, nii_file in enumerate(nii_files, 1):
        print(f"\nProcessing file {i}/{len(nii_files)}: {os.path.basename(nii_file)}")
        try:
            output_file = super_resolve_nii(nii_file, output_dir, mode)
            save_slices(output_file)
            print(f"Finished processing file: {os.path.basename(nii_file)}")
        except Exception as e:
            print(f"Error while processing file {os.path.basename(nii_file)}: {str(e)}")
    
    print(f"\Finished processing all {len(nii_files)} files!")

if __name__ == "__main__":
    """
    # For a folder
    input_dir = ""
    start = time.perf_counter()
    process_all_nii_files(input_dir, mode="hann")
    end = time.perf_counter()
    print(f"⏱️ Total processing time: {end - start:.2f} seconds")
    """
    
    # For a single file
    start = time.perf_counter()
    input_file = ""
    output_file = super_resolve_nii(input_file)
    end = time.perf_counter()
    print(f"Processing time: {end - start:.2f} seconds")  
    save_slices(output_file)
    