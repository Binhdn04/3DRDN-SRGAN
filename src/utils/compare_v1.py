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

def evaluate_subject(subject_id):
    short_id = subject_id.replace("sub-", "")
    
    results = {}
    
    views = ["axial", "coronal", "sagittal"]
    
    models = ["lr", "output"]
    
    for model in models:
        for view in views:
            inference_path = f"inference_images/2D/{subject_id}_T1w_{model}_{view}.png"
            original_path = f"original_images/{short_id}/{short_id}_{view}.png"
            
            if os.path.exists(inference_path) and os.path.exists(original_path):
                try:
                    psnr_val = calculate_psnr_2d(inference_path, original_path)
                    ssim_val = calculate_ssim_2d(inference_path, original_path)
                    
                    key = f"2D_{model}_{view}"
                    results[key] = {"PSNR": psnr_val, "SSIM": ssim_val}
                    
                    print(f"2D ({model}, {view}) - PSNR: {psnr_val:.3f} dB, SSIM: {ssim_val:.3f}")
                except Exception as e:
                    print(f"Error processing 2D image ({model}, {view}): {str(e)}")
            else:
                print(f"2D image not found ({model}, {view})")
    
    for model in models:
        inference_path = f"inference_images/3D/{subject_id}_T1w_{model}.nii.gz"
        original_path = f""
        if os.path.exists(inference_path) and os.path.exists(original_path):
            try:
                psnr_val = calculate_psnr_3d_whole(inference_path, original_path)
                ssim_val = calculate_ssim_3d_whole(inference_path, original_path)
                
                key = f"3D_{model}"
                results[key] = {"PSNR": psnr_val, "SSIM": ssim_val}
                
                print(f"3D ({model}) - PSNR: {psnr_val:.3f} dB, SSIM: {ssim_val:.3f}")
            except Exception as e:
                print(f"Error processing 3D image ({model}): {str(e)}")
        else:
            print(f"3D image not found ({model})")
    
    return results

def evaluate_multiple_subjects(subject_list=None, pattern=None):
    all_results = {}
    
    if subject_list is None and pattern is not None:
        inference_dirs = glob.glob(f"inference_images/2D/{pattern}_T1w_lr_sagittal.png")
        subject_list = [os.path.basename(d).split('_T1w')[0] for d in inference_dirs]
    
    for subject_id in subject_list:
        print(f"\nEvaluating {subject_id}:")
        results = evaluate_subject(subject_id)
        all_results[subject_id] = results
    
    avg_results = calculate_average_metrics(all_results)
    print("\nAverage results:")
    for key, metrics in avg_results.items():
        print(f"{key} - PSNR: {metrics['PSNR']:.3f} dB, SSIM: {metrics['SSIM']:.3f}")
    
    return all_results, avg_results

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

if __name__ == "__main__":
    print("Evaluating single subject:")
    results = evaluate_subject("sub-BrainAge000011")