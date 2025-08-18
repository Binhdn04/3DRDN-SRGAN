import glob
import numpy as np
import scipy.ndimage as ndi
import nibabel as nib
import SimpleITK as sitk
import tensorflow as tf
import os
from pathlib import Path
from scipy.ndimage import zoom

PATCH_SIZE = 40
SCALING_FACTOR = 4
interpolation_method = 'bicubic'
BOUNDARY_VOXELS_1 = 50
BOUNDARY_VOXELS_2 = BOUNDARY_VOXELS_1 - PATCH_SIZE

class SPDownsamplingDataPreprocessor:
    def __init__(self, psf_fwhm_mm=3.0, slice_thickness_mm=4.0, original_spacing_mm=1.0):
        """
        Combined SP-downsampling and data preprocessing for T1 MRI
        
        Args:
            psf_fwhm_mm: FWHM của truncated sinc PSF (3mm)
            slice_thickness_mm: Độ dày slice mong muốn (4mm)
            original_spacing_mm: Khoảng cách voxel gốc (1mm)
        """
        self.psf_fwhm = psf_fwhm_mm
        self.slice_thickness = slice_thickness_mm
        self.original_spacing = original_spacing_mm
        self.downsample_factor = slice_thickness_mm / original_spacing_mm
        
    def create_truncated_sinc_profile(self, length, fwhm):
        """
        Create sinc slice profile based on Bloch equations
        """
        x = np.linspace(-length/2 * self.original_spacing, 
                       length/2 * self.original_spacing, int(length))
        
        x_normalized = x / (fwhm / np.pi)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc_profile = np.sinc(x_normalized / np.pi)
            sinc_profile[np.isnan(sinc_profile)] = 1.0
        
        truncation_width = 3 * fwhm
        mask = np.abs(x) <= truncation_width
        truncated_sinc = sinc_profile * mask
        
        if np.sum(truncated_sinc) > 0:
            truncated_sinc = truncated_sinc / np.sum(truncated_sinc)
        
        return truncated_sinc
    
    def sp_downsample_3d(self, volume_3d, target_axis):
        """
        Applying SP-downsampling to 3D along the axis
        """
        psf_length = int(self.psf_fwhm / self.original_spacing * 6)
        psf = self.create_truncated_sinc_profile(psf_length, self.psf_fwhm)
        
        convolved = ndi.convolve1d(volume_3d, psf, axis=target_axis, mode='constant')
        
        downsample_step = int(self.downsample_factor)
        
        if target_axis == 0:  # LR direction (sagittal scans)
            downsampled = convolved[::downsample_step, :, :]
        elif target_axis == 1:  # AP direction
            downsampled = convolved[:, ::downsample_step, :]
        elif target_axis == 2:  # SI direction (axial scans)
            downsampled = convolved[:, :, ::downsample_step]
        
        return downsampled


@tf.function
def NormalizeImage(image):
    return (image - tf.math.reduce_min(image)) / (tf.math.reduce_max(image) - tf.math.reduce_min(image))

@tf.function
def get_random_patch_dims(image):    
    r_x = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[0]-PATCH_SIZE-BOUNDARY_VOXELS_2, 'int32')
    r_y = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[1]-PATCH_SIZE-BOUNDARY_VOXELS_2, 'int32')
    r_z = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[2]-PATCH_SIZE-BOUNDARY_VOXELS_2, 'int32')
    return r_x, r_y, r_z

def flip_model_x(model):
    return model[::-1, :, :]

def flip_model_y(model):
    return model[:, ::-1, :]

def flip_model_z(model):
    return model[:, :, ::-1]

@tf.function
def data_augmentation(lr_image, hr_image):
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_x(lr_image), flip_model_x(hr_image)
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_y(lr_image), flip_model_y(hr_image)
    if tf.random.uniform(()) > 0.5:
        lr_image, hr_image = flip_model_z(lr_image), flip_model_z(hr_image)
    return lr_image, hr_image

def get_nii_file_pair(nii_file_path, scan_type='axial'):
    img_sitk = sitk.ReadImage(nii_file_path.decode('UTF-8'), sitk.sitkFloat32)
    hr_image = sitk.GetArrayFromImage(img_sitk)
    
    sp_processor = SPDownsamplingDataPreprocessor()
    if scan_type == 'axial':
        lr_image = sp_processor.sp_downsample_3d(hr_image, target_axis=2)
        zoom_factors = (1.0, 1.0, sp_processor.downsample_factor)
        lr_image = zoom(lr_image, zoom_factors, order=1)  # Linear interpolation
        
    elif scan_type == 'sagittal':
        lr_image = sp_processor.sp_downsample_3d(hr_image, target_axis=0)
        zoom_factors = (sp_processor.downsample_factor, 1.0, 1.0)
        lr_image = zoom(lr_image, zoom_factors, order=1)
    
    min_shape = np.minimum(hr_image.shape, lr_image.shape)
    hr_image = hr_image[:min_shape[0], :min_shape[1], :min_shape[2]]
    lr_image = lr_image[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    return hr_image.astype(np.float32), lr_image.astype(np.float32)

@tf.function
def normalize(lr_image, hr_image):
    hr_image = NormalizeImage(hr_image)
    lr_image = NormalizeImage(lr_image)
    return lr_image, hr_image

@tf.function
def extract_patch(lr_image, hr_image):
    r_x, r_y, r_z = get_random_patch_dims(hr_image)
    hr_random_patch = hr_image[r_x:r_x+PATCH_SIZE, r_y:r_y+PATCH_SIZE, r_z:r_z+PATCH_SIZE]
    lr_random_patch = lr_image[r_x:r_x+PATCH_SIZE, r_y:r_y+PATCH_SIZE, r_z:r_z+PATCH_SIZE]
    return tf.expand_dims(lr_random_patch, axis=3), tf.expand_dims(hr_random_patch, axis=3)

def get_preprocessed_data(BATCH_SIZE, VALIDATION_BATCH_SIZE, scan_type='axial', seed=42):

    data_dir = ""
    nii_files = glob.glob(f"{data_dir}/**/*.nii.gz", recursive=True)
    np.random.seed(seed)
    indices = np.random.permutation(len(nii_files))
    nii_files = np.array(nii_files)[indices]
    
    n_files = len(nii_files)
    train_size = int(0.7 * n_files)
    valid_size = int(0.15 * n_files)
    
    train_files = nii_files[:train_size]
    valid_files = nii_files[train_size:train_size+valid_size]
    test_files = nii_files[train_size+valid_size:]
    
    splits_dir = "data_splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    with open(f"{splits_dir}/train_files_{scan_type}.txt", "w") as f:
        for file in train_files:
            f.write(f"{file}\n")
            
    with open(f"{splits_dir}/valid_files_{scan_type}.txt", "w") as f:
        for file in valid_files:
            f.write(f"{file}\n")
            
    with open(f"{splits_dir}/test_files_{scan_type}.txt", "w") as f:
        for file in test_files:
            f.write(f"{file}\n")
    
    print(f"Dataset split for {scan_type} scan type:")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(valid_files)}")
    print(f"Test files: {len(test_files)}")
    
    def create_dataset_pipeline(file_list, batch_size, is_training=False):
        AUTOTUNE = tf.data.AUTOTUNE
        file_ds = tf.data.Dataset.from_tensor_slices(file_list)
        
        ds = file_ds.map(
            lambda x: tf.numpy_function(
                func=lambda path: get_nii_file_pair(path, scan_type), 
                inp=[x], 
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=AUTOTUNE, 
            deterministic=True
        )

        ds = ds.map(
            normalize,
            num_parallel_calls=AUTOTUNE, 
            deterministic=True
        )

        ds = ds.map(
            extract_patch,
            num_parallel_calls=AUTOTUNE, 
            deterministic=True
        )
        if is_training:
            ds = ds.map(
                data_augmentation,
                num_parallel_calls=AUTOTUNE, 
                deterministic=True
            )
        ds = ds.batch(batch_size, drop_remainder=True)
        
        if is_training:
            ds = ds.prefetch(AUTOTUNE)
            
        return ds, len(file_list)
    
    train_dataset, train_dataset_size = create_dataset_pipeline(
        train_files, BATCH_SIZE, is_training=True
    )
    valid_dataset, valid_dataset_size = create_dataset_pipeline(
        valid_files, VALIDATION_BATCH_SIZE
    )
    test_dataset, test_dataset_size = create_dataset_pipeline(
        test_files, BATCH_SIZE
    )
    
    return train_dataset, train_dataset_size, valid_dataset, valid_dataset_size, test_dataset, test_dataset_size