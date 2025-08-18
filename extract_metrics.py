#!/usr/bin/env python3
import re
import csv
import argparse
import os
from pathlib import Path
from datetime import datetime

def extract_metrics_from_log(log_path):
    """
    Extract training metrics from a log file and split them by model (where each model ends with "After training").
    
    Args:
        log_path (str): Path to the log file
        
    Returns:
        list: List of model data, where each model is a list of dictionaries containing the extracted metrics
    """
    # Read the log file
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # Split the log into segments for each model (each ending with "After training")
    # We'll use the pattern that includes "After training" and its metrics
    after_training_pattern = r"After training: MAE = ([\d.]+) ± ([\d.]+), PSNR = ([\d.]+) ± ([\d.]+), SSIM = ([\d.]+) ± ([\d.]+)"
    
    # Find all occurrences of "After training"
    after_training_matches = list(re.finditer(after_training_pattern, log_content))
    
    # If no "After training" is found, process the entire log as one model
    if not after_training_matches:
        return [extract_single_model_metrics(log_content)]
    
    # Split the log content by "After training" occurrences
    model_logs = []
    start_pos = 0
    
    for i, match in enumerate(after_training_matches):
        # End position is the end of the current "After training" match
        end_pos = match.end()
        
        # Extract the segment for this model
        model_log = log_content[start_pos:end_pos]
        model_logs.append(model_log)
        
        # Update the start position for the next segment
        start_pos = end_pos
    
    # Add any remaining content after the last "After training"
    if start_pos < len(log_content):
        model_logs.append(log_content[start_pos:])
    
    # Process each model segment
    all_model_metrics = []
    for model_log in model_logs:
        model_metrics = extract_single_model_metrics(model_log)
        if model_metrics:
            all_model_metrics.append(model_metrics)
    
    return all_model_metrics

def extract_single_model_metrics(log_segment):
    """
    Extract metrics for a single model from a log segment.
    
    Args:
        log_segment (str): Segment of the log file for a single model
        
    Returns:
        list: List of dictionaries containing the extracted metrics for this model
    """
    results = []
    
    # Extract before training metrics
    before_training_pattern = r"Before training: MAE = ([\d.]+) ± ([\d.]+), PSNR = ([\d.]+) ± ([\d.]+), SSIM = ([\d.]+) ± ([\d.]+)"
    before_training_match = re.search(before_training_pattern, log_segment)
    
    if before_training_match:
        results.append({
            'epoch': 'Before training',
            'mae': float(before_training_match.group(1)),
            'mae_std': float(before_training_match.group(2)),
            'psnr': float(before_training_match.group(3)),
            'psnr_std': float(before_training_match.group(4)),
            'ssim': float(before_training_match.group(5)),
            'ssim_std': float(before_training_match.group(6)),
            'duration': ''
        })
    
    # Extract metrics for each epoch
    epoch_pattern = r"Began epoch (\d+) at .+?\nFinished epoch \d+ at .+?\nEpoch took ([\d:\.]+)\nAfter epoch: MAE = ([\d.]+) ± ([\d.]+), PSNR = ([\d.]+) ± ([\d.]+), SSIM = ([\d.]+) ± ([\d.]+)"
    epoch_matches = re.finditer(epoch_pattern, log_segment)
    
    for match in epoch_matches:
        results.append({
            'epoch': int(match.group(1)),
            'duration': match.group(2),
            'mae': float(match.group(3)),
            'mae_std': float(match.group(4)),
            'psnr': float(match.group(5)),
            'psnr_std': float(match.group(6)),
            'ssim': float(match.group(7)),
            'ssim_std': float(match.group(8))
        })
    
    # Extract after training metrics
    after_training_pattern = r"After training: MAE = ([\d.]+) ± ([\d.]+), PSNR = ([\d.]+) ± ([\d.]+), SSIM = ([\d.]+) ± ([\d.]+)"
    after_training_match = re.search(after_training_pattern, log_segment)
    
    if after_training_match:
        results.append({
            'epoch': 'After training',
            'mae': float(after_training_match.group(1)),
            'mae_std': float(after_training_match.group(2)),
            'psnr': float(after_training_match.group(3)),
            'psnr_std': float(after_training_match.group(4)),
            'ssim': float(after_training_match.group(5)),
            'ssim_std': float(after_training_match.group(6)),
            'duration': ''
        })
    
    return results

def save_to_csv(metrics, output_path):
    """
    Save extracted metrics to a CSV file.
    
    Args:
        metrics (list): List of dictionaries containing the extracted metrics
        output_path (str): Path to save the CSV file
    """
    if not metrics:
        print("No metrics found to save.")
        return
    
    fieldnames = ['epoch', 'mae', 'mae_std', 'psnr', 'psnr_std', 'ssim', 'ssim_std', 'duration']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    
    print(f"Metrics saved to {output_path}")

def generate_output_filename(base_output, model_index):
    """
    Generate a filename for each model.
    
    Args:
        base_output (str): Base output filename
        model_index (int): Index of the model
        
    Returns:
        str: Generated output filename
    """
    # Get the base path without extension
    base_path = os.path.splitext(base_output)[0]
    extension = os.path.splitext(base_output)[1] or '.csv'
    
    # Generate the new filename with model index
    return f"{base_path}_model_{model_index + 1}{extension}"

def main():
    parser = argparse.ArgumentParser(description='Extract metrics from training log file and save to CSV')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the training log file')
    parser.add_argument('--output', '-o', type=str, help='Base path to save the CSV files (default: metrics.csv)')
    parser.add_argument('--output_dir', '-d', type=str, help='Directory to save CSV files (default: current directory)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = input_path.with_suffix('.csv').name
    
    # Create output directory if specified and it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_base = os.path.join(args.output_dir, os.path.basename(args.output))
    else:
        output_base = args.output
    
    # Extract metrics for each model
    all_model_metrics = extract_metrics_from_log(args.input)
    
    if not all_model_metrics:
        print("No metrics found in the log file.")
        return
    
    # Save each model's metrics to a separate CSV file
    for i, model_metrics in enumerate(all_model_metrics):
        if model_metrics:
            output_path = generate_output_filename(output_base, i)
            print(f"Found model {i+1} with {len(model_metrics)} metric entries")
            save_to_csv(model_metrics, output_path)
    
    print(f"Processed {len(all_model_metrics)} models from {args.input}")

if __name__ == "__main__":
    main()