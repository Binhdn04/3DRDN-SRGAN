import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from scipy.ndimage import zoom
import os

PATCH_SIZE = 40
LR_PATCH_SIZE = 20
GAUSSIAN_NOISE = 0.25
SCALING_FACTOR = 2  
BOUNDARY_VOXELS_1 = 50
BOUNDARY_VOXELS_2 = BOUNDARY_VOXELS_1-PATCH_SIZE

@tf.function
def NormalizeImage(image):
    return (image - tf.math.reduce_min(image)) / (tf.math.reduce_max(image) - tf.math.reduce_min(image))

@tf.function
def get_random_patch_dims(image):    
    r_x = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[0]-PATCH_SIZE-BOUNDARY_VOXELS_2,'int32')
    r_y = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[1]-PATCH_SIZE-BOUNDARY_VOXELS_2,'int32')
    r_z = tf.random.uniform((), BOUNDARY_VOXELS_1, tf.shape(image)[2]-PATCH_SIZE-BOUNDARY_VOXELS_2,'int32')
    return r_x, r_y, r_z

def get_nii_file(nii_file_path):
    if isinstance(nii_file_path, bytes):
        nii_file_path = nii_file_path.decode('UTF-8')
    img_sitk = sitk.ReadImage(nii_file_path, sitk.sitkInt32)
    hr_image = sitk.GetArrayFromImage(img_sitk)
    return hr_image

def gaussian_blur_3d(image, sigma=1.0):
    image_sitk = sitk.GetImageFromArray(image)
    blurred_sitk = sitk.DiscreteGaussian(image_sitk, sigma)
    return sitk.GetArrayFromImage(blurred_sitk)

def resize_3d(image, scale_factor, method='bicubic'):
    method_to_order = {
        'nearest': 0,
        'linear': 1,
        'bicubic': 3,
        'lanczos': 5
    }
    
    order = method_to_order.get(method, 3)
    
    return zoom(image, zoom=[scale_factor]*3, order=order, mode='reflect', prefilter=True)

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


def create_test_dataset(nii_files, output_dir="test_data", max_patches=72):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "hr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lr_nearest"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lr_bilinear"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lr_bicubic"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lr_lanczos"), exist_ok=True)
    
    patch_count = 0
    
    files_to_process = nii_files[:max_patches]
    
    for file_idx, nii_file in enumerate(files_to_process):
        if patch_count >= max_patches:
            print(f"Reached maximum number of patches ({max_patches})")
            break
            
        print(f"Processing file {file_idx+1}/{len(files_to_process)}: {nii_file}")
        
        hr_image = get_nii_file(nii_file)
        hr_image_tensor = tf.convert_to_tensor(hr_image)
        
        r_x, r_y, r_z = get_random_patch_dims(hr_image_tensor)
        
        hr_patch = hr_image[r_x:r_x+PATCH_SIZE, r_y:r_y+PATCH_SIZE, r_z:r_z+PATCH_SIZE]
        
        blurred_patch = gaussian_blur_3d(hr_patch, sigma=GAUSSIAN_NOISE)
        
        lr_patch_downscaled = resize_3d(blurred_patch, 1/SCALING_FACTOR, method='bicubic')
        
        lr_nearest = sitk_upsample(lr_patch_downscaled, scale_factor=SCALING_FACTOR, interpolator='nearest')
        lr_bilinear = sitk_upsample(lr_patch_downscaled, scale_factor=SCALING_FACTOR, interpolator='linear')
        lr_bicubic = sitk_upsample(lr_patch_downscaled, scale_factor=SCALING_FACTOR, interpolator='cubic')
        lr_lanczos = sitk_upsample(lr_patch_downscaled, scale_factor=SCALING_FACTOR, interpolator='lanczos')

        hr_patch_tensor = tf.convert_to_tensor(hr_patch)
        lr_nearest_tensor = tf.convert_to_tensor(lr_nearest)
        lr_bilinear_tensor = tf.convert_to_tensor(lr_bilinear)
        lr_bicubic_tensor = tf.convert_to_tensor(lr_bicubic)
        lr_lanczos_tensor = tf.convert_to_tensor(lr_lanczos)
        
        hr_patch_norm = NormalizeImage(hr_patch_tensor).numpy()
        lr_nearest_norm = NormalizeImage(lr_nearest_tensor).numpy()
        lr_bilinear_norm = NormalizeImage(lr_bilinear_tensor).numpy()
        lr_bicubic_norm = NormalizeImage(lr_bicubic_tensor).numpy()
        lr_lanczos_norm = NormalizeImage(lr_lanczos_tensor).numpy()
        
        patch_name = f"patch_{patch_count:04d}"
        np.save(os.path.join(output_dir, "hr", f"{patch_name}.npy"), hr_patch_norm)
        np.save(os.path.join(output_dir, "lr_nearest", f"{patch_name}.npy"), lr_nearest_norm)
        np.save(os.path.join(output_dir, "lr_bilinear", f"{patch_name}.npy"), lr_bilinear_norm)
        np.save(os.path.join(output_dir, "lr_bicubic", f"{patch_name}.npy"), lr_bicubic_norm)
        np.save(os.path.join(output_dir, "lr_lanczos", f"{patch_name}.npy"), lr_lanczos_norm)
        
        patch_count += 1
        print(f"  Saved patch {patch_count}/{max_patches}")
    
    print(f"Total patches saved: {patch_count}/{max_patches}")
    return patch_count

def evaluation_loop(dataset, PATCH_SIZE):
    output_data = np.empty((0,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,1), 'float64')
    hr_data     = np.empty((0,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,1), 'float64')
    for lr_image, hr_image in dataset:        
        output = lr_image
        output_data = np.append(output_data, output, axis=0)
        hr_data     = np.append(hr_data, hr_image, axis=0)
    output_data = tf.squeeze(output_data).numpy()
    hr_data     = tf.squeeze(hr_data).numpy()
    mean_errors = []
    std_errors  = []
    psnr        = tf.image.psnr(output_data, hr_data, 1)
    psnr        = psnr[psnr!=float("inf")]
    ssim        = tf.image.ssim(output_data, hr_data, 1)
    abs_err     = tf.math.abs(hr_data-output_data)
    for v in [psnr, ssim, abs_err]:
        mean_errors.append(round(tf.reduce_mean(v).numpy() ,3))
        std_errors.append(round(tf.math.reduce_std(v).numpy() ,3))
    return mean_errors, std_errors

def create_tf_dataset_from_dir(test_dir, method_name, patch_size=40, batch_size=1):
    hr_files = sorted(glob.glob(os.path.join(test_dir, "hr", "*.npy")))
    
    hr_data = []
    lr_data = []
    
    for hr_file in hr_files:
        filename = os.path.basename(hr_file)
        
        hr_image = np.load(hr_file)
        
        lr_file = os.path.join(test_dir, f"lr_{method_name}", filename)
        if os.path.exists(lr_file):
            lr_image = np.load(lr_file)
            
            hr_image = np.expand_dims(hr_image, axis=-1)
            lr_image = np.expand_dims(lr_image, axis=-1)
            
            hr_data.append(hr_image)
            lr_data.append(lr_image)
    
    hr_data = np.array(hr_data)
    lr_data = np.array(lr_data)
    
    dataset = tf.data.Dataset.from_tensor_slices((lr_data, hr_data))
    dataset = dataset.batch(batch_size)
    
    return dataset

def evaluate_all_methods(test_dir, batch_size=1):
    methods = ["nearest", "bilinear", "bicubic", "lanczos"]
    results = {}
    
    print("\n" + "="*70)
    print("EVALUATION OF TRADITIONAL UPSCALING METHODS")
    print("="*70)
    
    for method in methods:
        print(f"\nEvaluating method {method.upper()}:")
        
        dataset = create_tf_dataset_from_dir(test_dir, method, PATCH_SIZE, batch_size)
        
        mean_errors, std_errors = evaluation_loop(dataset, PATCH_SIZE)
        
        results[method] = {
            "psnr": f"{mean_errors[0]:.3f}",
            "ssim": f"{mean_errors[1]:.3f}",
            "abs_err": f"{mean_errors[2]:.3f}",
            "psnr_std": f"{std_errors[0]:.3f}",
            "ssim_std": f"{std_errors[1]:.3f}",
            "abs_err_std": f"{std_errors[2]:.3f}"
        }
        
        print(f"  MAE = {mean_errors[2]:.3f} ± {std_errors[2]:.3f}, PSNR = {mean_errors[0]:.3f} ± {std_errors[0]:.3f}, SSIM = {mean_errors[1]:.3f} ± {std_errors[1]:.3f}")
    
    return results

def load_test_files_from_txt(test_files_path):

    test_files = []
    
    with open(test_files_path, 'r') as f:
        lines = f.readlines()
        

    for line in lines:
        file_path = line.strip()
        if file_path:  
            if os.path.exists(file_path):
                test_files.append(file_path)
            else:
                print(f"Warning: File not found: {file_path}")
    
    return test_files

if __name__ == "__main__":
    test_files_txt = "data_splits/test_files.txt"
    test_nii_files = load_test_files_from_txt(test_files_txt)
    print(f"Number of test files from {test_files_txt}: {len(test_nii_files)}")
    output_dir = "test_upscaling_data"
    create_test_dataset(test_nii_files, output_dir=output_dir, max_patches=len(test_nii_files))
    results = evaluate_all_methods(output_dir, batch_size=4)
    import json
    with open("evaluation_results_custom.json", "w") as f:
        json.dump(results, f, indent=4)