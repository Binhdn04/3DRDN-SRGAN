import nibabel as nib
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
import glob

def read_nifti_image(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return data

def calculate_psnr_3d_whole(vol1_path, vol2_path, max_val=1.0):
    vol1 = read_nifti_image(vol1_path)
    vol2 = read_nifti_image(vol2_path)
    psnr_value = psnr(vol1, vol2, data_range=1.0)
    return psnr_value

def calculate_ssim_3d_whole(vol1_path, vol2_path):
    vol1 = read_nifti_image(vol1_path)
    vol2 = read_nifti_image(vol2_path)
    ssim_value = ssim(vol1, vol2, data_range=1.0)
    return ssim_value

def calculate_psnr_2d(image1_path, image2_path):
    img1 = imread(image1_path).astype(np.float32) / 255.0
    img2 = imread(image2_path).astype(np.float32) / 255.0
    return psnr(img1, img2, data_range=1.0)

def calculate_ssim_2d(image1_path, image2_path):
    img1 = imread(image1_path).astype(np.float32)
    img2 = imread(image2_path).astype(np.float32)

    img1 = imread(image1_path).astype(np.float32) / 255.0
    img2 = imread(image2_path).astype(np.float32) / 255.0    
    if img1.ndim == 3:
        return ssim(img1, img2, data_range=1.0, channel_axis=2, 
                   gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11)
    else:
        return ssim(img1, img2, data_range=1.0, channel_axis=None,
                   gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11)

def evaluate_subject(subject_id, interpolation_methods=None):
    short_id = subject_id.replace("sub-", "")
    
    if interpolation_methods is None:
        interpolation_methods = ['nearest', 'linear', 'cubic', 'spline5', 
              'lanczos', 'sitk_cubic', 'sitk_nearest', 'sitk_linear']
    
    results = {}
    
    views = ["axial", "coronal", "sagittal"]
    
    base_dir = ""
    
    for method in interpolation_methods:
        for view in views:
            inference_path = f"{base_dir}/inference_interpolation/2D/sub-{short_id}_T1w_{method}_{view}.png"
            
            original_path = f"{base_dir}/original_images/{short_id}/{short_id}_{view}.png"
            
            if os.path.exists(inference_path) and os.path.exists(original_path):
                try:
                    psnr_val = calculate_psnr_2d(inference_path, original_path)
                    ssim_val = calculate_ssim_2d(inference_path, original_path)
                    
                    key = f"2D_{method}_{view}"
                    results[key] = {"PSNR": psnr_val, "SSIM": ssim_val}
                    
                    print(f"2D ({method}, {view}) - PSNR: {psnr_val:.3f} dB, SSIM: {ssim_val:.3f}")
                except Exception as e:
                    print(f"Error processing 2D image ({method}, {view}): {str(e)}")
            else:
                print(f"Image not found: \nInference: {inference_path}\nOriginal: {original_path}")

    return results

def evaluate_multiple_subjects(subject_list=None, pattern=None, interpolation_methods=None):
    all_results = {}
    
    if subject_list is None and pattern is not None:
        base_dir = ""
        
        default_method = "cubic" if interpolation_methods is None or len(interpolation_methods) == 0 else interpolation_methods[0]
        inference_files = glob.glob(f"{base_dir}/inference_interpolation/2D/{pattern}_T1w_{default_method}_sagittal.png")
        subject_list = [os.path.basename(f).split('_T1w')[0] for f in inference_files]
    
    for subject_id in subject_list:
        print(f"\nEvaluating {subject_id}:")
        results = evaluate_subject(subject_id, interpolation_methods)
        all_results[subject_id] = results
    
    avg_results = calculate_average_metrics(all_results)
    print("\nAverage results:")
    for key, metrics in avg_results.items():
        print(f"{key} - PSNR: {metrics['PSNR']:.3f} dB, SSIM: {metrics['SSIM']:.3f}")
    
    if interpolation_methods is None:
        interpolation_methods = ['nearest', 'linear', 'cubic', 'spline5', 
              'lanczos', 'sitk_cubic', 'sitk_nearest', 'sitk_linear']
    
    method_results = calculate_method_averages(avg_results, interpolation_methods)
    print("\nAverage results by method:")
    for method, metrics in method_results.items():
        print(f"{method} - PSNR: {metrics['PSNR']:.3f} dB, SSIM: {metrics['SSIM']:.3f}")
    
    return all_results, avg_results, method_results

def calculate_average_metrics(all_results):
    metric_sums = {}
    metric_counts = {}
    
    for subject_id, results in all_results.items():
        for key, metrics in results.items():
            if key not in metric_sums:
                metric_sums[key] = {"PSNR": 0.0, "SSIM": 0.0}
                metric_counts[key] = 0
            
            metric_sums[key]["PSNR"] += metrics["PSNR"]
            metric_sums[key]["SSIM"] += metrics["SSIM"]
            metric_counts[key] += 1
    
    avg_results = {}
    for key, sums in metric_sums.items():
        count = metric_counts[key]
        if count > 0:
            avg_results[key] = {
                "PSNR": sums["PSNR"] / count,
                "SSIM": sums["SSIM"] / count
            }
    
    return avg_results

def calculate_method_averages(avg_results, interpolation_methods):
    method_sums = {}
    method_counts = {}
    
    for method in interpolation_methods:
        method_sums[method] = {"PSNR": 0.0, "SSIM": 0.0}
        method_counts[method] = 0
    
    for key, metrics in avg_results.items():
        parts = key.split('_')
        if len(parts) >= 2:
            method = parts[1]
            if method in interpolation_methods:
                method_sums[method]["PSNR"] += metrics["PSNR"]
                method_sums[method]["SSIM"] += metrics["SSIM"]
                method_counts[method] += 1
    
    method_results = {}
    for method, sums in method_sums.items():
        count = method_counts[method]
        if count > 0:
            method_results[method] = {
                "PSNR": sums["PSNR"] / count,
                "SSIM": sums["SSIM"] / count
            }
    
    return method_results


if __name__ == "__main__":
    interpolation_methods = ['nearest', 'linear', 'cubic', 'spline5', 
              'lanczos', 'sitk_cubic', 'sitk_nearest', 'sitk_linear']
    
    print("Evaluating single subject:")
    results = evaluate_subject("sub-BrainAge000011", interpolation_methods)