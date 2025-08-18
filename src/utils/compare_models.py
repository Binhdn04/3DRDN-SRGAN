#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def compare_models(csv_files, output_dir=None, dpi=300, model_names=None):
    """
    Compare multiple models by plotting their metrics on the same chart.
    
    Args:
        csv_files (list): List of paths to CSV files containing training metrics
        output_dir (str, optional): Directory to save the plots. If None, save in the current directory
        dpi (int, optional): DPI for saved figures. Default is 300.
        model_names (list, optional): Names to display for each model. If None, use filenames.
    """
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    else:
        output_path = Path.cwd()
    
    if model_names is None:
        model_names = [Path(file).stem for file in csv_files]
    if len(model_names) < len(csv_files):
        for i in range(len(model_names), len(csv_files)):
            model_names.append(f"Model_{i+1}")
    
    processed_data = []
    max_epochs = []
    
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        numeric_df = df[pd.to_numeric(df['epoch'], errors='coerce').notna()]
        numeric_df['epoch'] = pd.to_numeric(numeric_df['epoch'])
        numeric_df = numeric_df.sort_values('epoch')
        
        processed_data.append(numeric_df)
        max_epochs.append(numeric_df['epoch'].max())
    min_max_epoch = min(max_epochs)
    common_data = []
    for df in processed_data:
        common_data.append(df[df['epoch'] <= min_max_epoch])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    colors = colors[:len(csv_files)]
    metrics = [
        {'name': 'PSNR', 'col': 'psnr', 'std_col': 'psnr_std', 'higher_better': True},
        {'name': 'SSIM', 'col': 'ssim', 'std_col': 'ssim_std', 'higher_better': True},
        {'name': 'MAE', 'col': 'mae', 'std_col': 'mae_std', 'higher_better': False}
    ]

    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        for i, df in enumerate(common_data):
            x = df['epoch']
            y = df[metric['col']]
            y_std = df[metric['std_col']]
            plt.plot(x, y, color=colors[i], linewidth=2, label=f"{model_names[i]}")
            plt.fill_between(x, y - y_std, y + y_std, color=colors[i], alpha=0.2)

        plt.legend(fontsize=12)

        metric_title = metric['name']
        if metric['higher_better']:
            metric_title += " (higher is better)"
        else:
            metric_title += " (lower is better)"
            
        plt.title(f"Comparison of {metric['name']} across models", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel(metric_title, fontsize=14)
        

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(left=0)
        plt.tight_layout()
        
        output_file = output_path / f"comparison_{metric['name'].lower()}.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"{metric['name']} comparison plot saved to {output_file}")
        plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, df in enumerate(common_data):
            x = df['epoch']
            y = df[metric['col']]
            y_std = df[metric['std_col']]
            
            ax.plot(x, y, color=colors[j], linewidth=2, label=f"{model_names[j]}")
            ax.fill_between(x, y - y_std, y + y_std, color=colors[j], alpha=0.2)
        if i == 0:
            ax.legend(fontsize=10)

        ax.set_title(metric['name'], fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric['name'], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(left=0)
    
    plt.tight_layout()

    combined_output = output_path / "comparison_all_metrics.png"
    plt.savefig(combined_output, dpi=dpi, bbox_inches='tight')
    print(f"Combined comparison plot saved to {combined_output}")
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, df in enumerate(common_data):
            x = df['epoch']
            y = df[metric['col']]
            y_std = df[metric['std_col']]
            ax.plot(x, y, color=colors[j], linewidth=2.5, label=f"{model_names[j]}")
            ax.fill_between(x, y - y_std, y + y_std, color=colors[j], alpha=0.2)

        if i == 0:
            ax.legend(fontsize=12)
        ax.set_title(metric['name'], fontsize=14, fontweight='bold')
        ax.set_ylabel(metric['name'], fontsize=12)
        if i == 2:
            ax.set_xlabel("Epoch", fontsize=12)
        ax.set_xlim(left=0)
    plt.tight_layout()
    reference_output = output_path / "comparison_reference_style.png"
    plt.savefig(reference_output, dpi=dpi, bbox_inches='tight')
    print(f"Reference-style comparison plot saved to {reference_output}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare multiple model metrics from CSV files')
    parser.add_argument('--csv_files', '-c', type=str, nargs='+', required=True, 
                        help='Paths to the CSV files containing training metrics')
    parser.add_argument('--output_dir', '-o', type=str, default=None, 
                        help='Directory to save the plots (default: current directory)')
    parser.add_argument('--dpi', '-d', type=int, default=300, 
                        help='DPI for saved figures (default: 300)')
    parser.add_argument('--model_names', '-n', type=str, nargs='+', default=None,
                        help='Names to display for each model (default: filenames)')
    
    args = parser.parse_args()
    
    compare_models(args.csv_files, args.output_dir, args.dpi, args.model_names)

if __name__ == "__main__":
    main()