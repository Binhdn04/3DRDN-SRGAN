import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from scipy.ndimage import zoom

PATCH_SIZE           = 40
GAUSSIAN_NOISE       = 0.25
SCALING_FACTOR       = 2
interpolation_method = 'bicubic'
BOUNDARY_VOXELS_1    = 40
BOUNDARY_VOXELS_2    = BOUNDARY_VOXELS_1-PATCH_SIZE

@tf.function
def NormalizeImage(image):
    return (image - tf.math.reduce_min(image)) / (tf.math.reduce_max(image) - tf.math.reduce_min(image))

@tf.function
def get_random_patch_dims(image):    
    r_x = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[0]-PATCH_SIZE-BOUNDARY_VOXELS_2,'int32')
    r_y = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[1]-PATCH_SIZE-BOUNDARY_VOXELS_2,'int32')
    r_z = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[2]-PATCH_SIZE-BOUNDARY_VOXELS_2,'int32')
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

def get_nii_file(nii_file_path):
    img_sitk = sitk.ReadImage(nii_file_path.decode('UTF-8'), sitk.sitkFloat32)
    hr_image = sitk.GetArrayFromImage(img_sitk)
    return hr_image

def gaussian_blur_3d(image, sigma=1.0):
    image_sitk = sitk.GetImageFromArray(image)
    blurred_sitk = sitk.DiscreteGaussian(image_sitk, float(sigma))
    return sitk.GetArrayFromImage(blurred_sitk)

def add_noise(hr_image):
    blurred_image = tf.numpy_function(func=gaussian_blur_3d, inp=[hr_image, GAUSSIAN_NOISE], Tout=tf.float32)
    return blurred_image, hr_image

def get_low_res(blurred_image, hr_image, interpolation_order=3):
    """
    - interpolation_order: 0=nearest, 1=bilinear, 3=cubic
    """
    scale_factors = [1.0 / SCALING_FACTOR] * 3
    lr_image = zoom(blurred_image, zoom=scale_factors, order=interpolation_order)
    ups_lr_image = zoom(lr_image, zoom=[SCALING_FACTOR]*3, order=interpolation_order)
    ups_lr_image = ups_lr_image.astype(np.float32)

    return ups_lr_image, hr_image


@tf.function
def normalize(lr_image, hr_image):
    hr_image = NormalizeImage(hr_image)
    lr_image = NormalizeImage(lr_image)
    return lr_image, hr_image

@tf.function
def extract_patch(lr_image, hr_image):
    r_x, r_y, r_z = get_random_patch_dims(hr_image)
    hr_random_patch = hr_image[r_x:r_x+PATCH_SIZE,r_y:r_y+PATCH_SIZE,r_z:r_z+PATCH_SIZE]
    lr_random_patch = lr_image[r_x:r_x+PATCH_SIZE,r_y:r_y+PATCH_SIZE,r_z:r_z+PATCH_SIZE]
    return tf.expand_dims(lr_random_patch, axis=3), tf.expand_dims(hr_random_patch, axis=3)

def get_preprocessed_data(BATCH_SIZE, VALIDATION_BATCH_SIZE, seed=42):
    import os, glob, numpy as np, tensorflow as tf
    

    data_dir = "/home/nhabdoan/thinclient_drives/Binh/CamCAN"
    nii_files = glob.glob(f"{data_dir}/**/*.nii.gz", recursive=True)

    if len(nii_files) == 0:
        raise ValueError("No .nii.gz files found!")

    np.random.seed(seed)
    indices = np.random.permutation(len(nii_files))
    nii_files = np.array(nii_files)[indices]

    # >>> CONFIG: Choose how to split data here <<<
    mode = "ratio"   

    if mode == "ratio":
        train_ratio, valid_ratio, test_ratio = 0.7, 0.15, 0.15
        n_files = len(nii_files)
        train_size = int(train_ratio * n_files)
        valid_size = int(valid_ratio * n_files)
        test_size = n_files - train_size - valid_size

    elif mode == "fixed":
        train_size, valid_size, test_size = 352, 100, 200
        total_required = train_size + valid_size + test_size
        if len(nii_files) < total_required:
            raise ValueError(f"Not enough files! Required {total_required}, but found {len(nii_files)}.")

    else:
        raise ValueError("mode must be 'ratio' or 'fixed'")

    train_files = nii_files[:train_size]
    valid_files = nii_files[train_size:train_size+valid_size]
    test_files = nii_files[train_size+valid_size:train_size+valid_size+test_size]

    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(valid_files)}")
    print(f"Test files: {len(test_files)}")
    print(f"Total used: {len(train_files) + len(valid_files) + len(test_files)} / {len(nii_files)}")

    splits_dir = "data_splits"
    os.makedirs(splits_dir, exist_ok=True)
    for split_name, file_list in zip(
        ["train", "valid", "test"], [train_files, valid_files, test_files]
    ):
        with open(f"{splits_dir}/{split_name}_files.txt", "w") as f:
            for file in file_list:
                f.write(f"{file}\n")

    def create_dataset_pipeline(file_list, batch_size, is_training=False):
        AUTOTUNE = tf.data.AUTOTUNE
        file_ds = tf.data.Dataset.from_tensor_slices(file_list)

        ds = file_ds.map(
            lambda x: tf.numpy_function(func=get_nii_file, inp=[x], Tout=tf.float32),
            num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.map(add_noise, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.map(
            lambda x, y: tf.numpy_function(func=get_low_res, inp=[x, y], Tout=(tf.float32, tf.float32)),
            num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.map(normalize, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.map(extract_patch, num_parallel_calls=AUTOTUNE, deterministic=True)

        if is_training:
            ds = ds.map(data_augmentation, num_parallel_calls=AUTOTUNE, deterministic=True)

        ds = ds.batch(batch_size, drop_remainder=True)
        if is_training:
            ds = ds.prefetch(AUTOTUNE)
        return ds, len(file_list)


    train_dataset, train_dataset_size = create_dataset_pipeline(train_files, BATCH_SIZE, is_training=True)
    valid_dataset, valid_dataset_size = create_dataset_pipeline(valid_files, VALIDATION_BATCH_SIZE)
    test_dataset, test_dataset_size = create_dataset_pipeline(test_files, BATCH_SIZE)

    return train_dataset, train_dataset_size, valid_dataset, valid_dataset_size, test_dataset, test_dataset_size