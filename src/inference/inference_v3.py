import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import glob
from model import Model3DRDN  
from scipy.fft import fft, ifft

PATCH_SIZE = 40


class SPProcessor:
    
    def __init__(self, psf_fwhm_mm=3.0, slice_thickness_mm=4.0, original_spacing_mm=1.0, reg_lambda=0.01):
        self.psf_fwhm = psf_fwhm_mm
        self.slice_thickness = slice_thickness_mm
        self.original_spacing = original_spacing_mm
        self.downsample_factor = int(slice_thickness_mm / original_spacing_mm)
        self.upsample_factor = int(slice_thickness_mm / original_spacing_mm)
        self.reg_lambda = reg_lambda
        
    def create_sinc_psf(self, length):
        x = np.linspace(-length/2 * self.original_spacing, length/2 * self.original_spacing, length)
        x_norm = x / (self.psf_fwhm / np.pi)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc_profile = np.sinc(x_norm / np.pi)
            sinc_profile[np.isnan(sinc_profile)] = 1.0
        
        mask = np.abs(x) <= 3 * self.psf_fwhm
        psf = sinc_profile * mask
        return psf / np.sum(psf) if np.sum(psf) > 0 else psf
    
    def wiener_deconv_1d(self, signal, psf):
        signal_fft = fft(signal)
        psf_fft = fft(psf, n=len(signal))
        
        psf_power = np.abs(psf_fft) ** 2
        wiener_filter = np.conj(psf_fft) / (psf_power + self.reg_lambda)
        
        deconvolved_fft = signal_fft * wiener_filter
        return np.real(ifft(deconvolved_fft))
    
    def apply_deconv(self, volume, target_axis=2):
        if target_axis == 0:
            upsampled = np.repeat(volume, self.upsample_factor, axis=0)
        elif target_axis == 1:
            upsampled = np.repeat(volume, self.upsample_factor, axis=1)
        else:
            upsampled = np.repeat(volume, self.upsample_factor, axis=2)
        
        psf_length = int(self.psf_fwhm / self.original_spacing * 6)
        psf = self.create_sinc_psf(psf_length)
        
        result = upsampled.copy()
        if target_axis == 2:
            for i in range(upsampled.shape[0]):
                for j in range(upsampled.shape[1]):
                    result[i, j, :] = self.wiener_deconv_1d(upsampled[i, j, :], psf)
        elif target_axis == 1:
            for i in range(upsampled.shape[0]):
                for k in range(upsampled.shape[2]):
                    result[i, :, k] = self.wiener_deconv_1d(upsampled[i, :, k], psf)
        elif target_axis == 0:
            for j in range(upsampled.shape[1]):
                for k in range(upsampled.shape[2]):
                    result[:, j, k] = self.wiener_deconv_1d(upsampled[:, j, k], psf)
        
        return np.clip(result, 0, None)

def process_image_with_patches(img_data, generator, patch_size=(40, 40, 40), step_size=10, 
                              batch_size=4, use_tta=True, mode="average", use_sp_deconv=False, 
                              sp_axis=2):
    
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
    
    sp_deconv = SPProcessor() if use_sp_deconv else None
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
            
            if use_sp_deconv and sp_deconv:
                output_patch = sp_deconv.apply_deconv(output_patch, target_axis=sp_axis)
                if sp_axis == 2:
                    patch_z *= sp_deconv.upsample_factor
                elif sp_axis == 1:
                    patch_y *= sp_deconv.upsample_factor
                elif sp_axis == 0:
                    patch_x *= sp_deconv.upsample_factor
            
            if mode == "hann" and not use_sp_deconv:
                weighted_patch = output_patch * hann_window
                weight_contribution = hann_window
            else:
                weighted_patch = output_patch
                weight_contribution = np.ones_like(output_patch)
            
            try:
                result_image[x_start:x_start+patch_x, 
                            y_start:y_start+patch_y, 
                            z_start:z_start+patch_z] += weighted_patch
                    
                weight_sum[x_start:x_start+patch_x, 
                          y_start:y_start+patch_y, 
                          z_start:z_start+patch_z] += weight_contribution
            except ValueError:
                actual_patch_shape = output_patch.shape
                end_x = min(x_start + actual_patch_shape[0], result_image.shape[0])
                end_y = min(y_start + actual_patch_shape[1], result_image.shape[1])
                end_z = min(z_start + actual_patch_shape[2], result_image.shape[2])
                
                crop_x = end_x - x_start
                crop_y = end_y - y_start
                crop_z = end_z - z_start
                
                result_image[x_start:end_x, y_start:end_y, z_start:end_z] += \
                    weighted_patch[:crop_x, :crop_y, :crop_z]
                weight_sum[x_start:end_x, y_start:end_y, z_start:end_z] += \
                    weight_contribution[:crop_x, :crop_y, :crop_z]
    
    weight_sum = np.where(weight_sum == 0, 1, weight_sum)
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

def super_resolve_nii(input_nii_path, output_dir="inference_images/3D", mode="hann", 
                     use_sp_deconv=False, sp_axis=2):
    
    model = Model3DRDN(MODEL="3DRDN")
    generator = model.generator_g    

    nii_img = nib.load(input_nii_path)
    img_data = nii_img.get_fdata()

    output_image = process_image_with_patches(
        img_data, generator, use_tta=True, mode=mode, 
        use_sp_deconv=use_sp_deconv, sp_axis=sp_axis
    )

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_nii_path).replace(".nii.gz", 
                                                      "_spdeconv.nii.gz" if use_sp_deconv else "_output.nii.gz")
    output_nii_path = os.path.join(output_dir, filename)

    output_nii = nib.Nifti1Image(output_image, affine=nii_img.affine, header=nii_img.header)
    nib.save(output_nii, output_nii_path)

    print(f"Image{' with SP-deconv' if use_sp_deconv else ''} saved at: {output_nii_path}")
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

def process_all_nii_files(input_dir, output_dir="inference_images/3D", mode="hann", 
                         use_sp_deconv=False, sp_axis=2):
    
    nii_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    
    if not nii_files:
        print(f"No .nii.gz files found in directory: {input_dir}")
        return
    
    print(f"Found {len(nii_files)} .nii.gz files to process")
    if use_sp_deconv:
        print(f"SP-deconvolution will be applied along axis {sp_axis}")
    
    for i, nii_file in enumerate(nii_files, 1):
        print(f"\nProcessing file {i}/{len(nii_files)}: {os.path.basename(nii_file)}")
        try:
            output_file = super_resolve_nii(nii_file, output_dir, mode, use_sp_deconv, sp_axis)
            save_slices(output_file)
            print(f"Completed processing file: {os.path.basename(nii_file)}")
        except Exception as e:
            print(f"Error processing file {os.path.basename(nii_file)}: {str(e)}")
    
    print(f"\nCompleted processing all {len(nii_files)} files!")

if __name__ == "__main__":
    input_file = ""

    print("\nStarting inference with SP-deconvolution...")
    start = time.perf_counter()
    super_resolve_nii(input_file, mode="hann", use_sp_deconv=True, sp_axis=2)
    end = time.perf_counter()
    print(f"SP-deconv inference time: {end - start:.2f} seconds")