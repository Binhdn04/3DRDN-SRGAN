#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_training_metrics(csv_path, output_dir=None, dpi=300, fontsize=80):
    """
    Plot training metrics (PSNR, SSIM, MAE) with their standard deviations.
    Plots are arranged vertically with unified font size for all text elements.
    
    Args:
        csv_path (str): Path to the CSV file containing training metrics
        output_dir (str, optional): Directory to save the plots. If None, save in the same directory as the CSV
        dpi (int, optional): DPI for saved figures. Default is 300.
        fontsize (int, optional): Base font size for all text elements. Default is 14.
    """
    title_fontsize = fontsize + 4      # Larger than base
    label_fontsize = fontsize + 2      # Slightly larger than base
    tick_fontsize = fontsize           # Same as base
    legend_fontsize = fontsize         # Same as base

    df = pd.read_csv(csv_path)
    numeric_df = df[pd.to_numeric(df['epoch'], errors='coerce').notna()]
    numeric_df['epoch'] = pd.to_numeric(numeric_df['epoch'])
    numeric_df = numeric_df.sort_values('epoch')

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    else:
        output_path = Path(csv_path).parent
    
    base_filename = Path(csv_path).stem
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    ax = axes[0]
    x = numeric_df['epoch']
    y = numeric_df['psnr']
    y_std = numeric_df['psnr_std']

    ax.plot(x, y, color='#EF476F', linewidth=2, label='Mean')
    ax.fill_between(x, y - y_std, y + y_std, color='#EF476F', alpha=0.2, label='Std')
    ax.set_title('PSNR', fontsize=title_fontsize)
    ax.set_xlabel('Epoch', fontsize=label_fontsize)
    ax.set_ylabel('PSNR', fontsize=label_fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=legend_fontsize, loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    ax = axes[1]
    x = numeric_df['epoch']
    y = numeric_df['ssim']
    y_std = numeric_df['ssim_std']
    
    ax.plot(x, y, color='#D99C0E', linewidth=2, label='Mean')
    
    ax.fill_between(x, y - y_std, y + y_std, color='#D99C0E', alpha=0.2, label='Std')
    
    ax.set_title('SSIM', fontsize=title_fontsize)
    ax.set_xlabel('Epoch', fontsize=label_fontsize)
    ax.set_ylabel('SSIM', fontsize=label_fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=legend_fontsize, loc='lower right')
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    ax = axes[2]
    x = numeric_df['epoch']
    y = numeric_df['mae']
    y_std = numeric_df['mae_std']
    
    ax.plot(x, y, color='#04a97e', linewidth=2, label='Mean')
    
    ax.fill_between(x, y - y_std, y + y_std, color='#04a97e', alpha=0.2, label='Std')

    ax.set_title('MAE', fontsize=title_fontsize)
    ax.set_xlabel('Epoch', fontsize=label_fontsize)
    ax.set_ylabel('MAE', fontsize=label_fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=legend_fontsize, loc='lower right')
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    combined_output = output_path / f"{base_filename}_metrics_combined.png"
    plt.savefig(combined_output, dpi=dpi, bbox_inches='tight')
    print(f"Combined plot saved to {combined_output}")
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from CSV file')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Path to the CSV file containing training metrics')
    parser.add_argument('--output_dir', '-o', type=str, 
                        help='Directory to save the plots (default: same as input CSV)')
    parser.add_argument('--dpi', '-d', type=int, default=300, 
                        help='DPI for saved figures (default: 300)')
    parser.add_argument('--fontsize', '-f', type=int, default=16,
                        help='Base font size for all text elements (default: 14)')
    
    args = parser.parse_args()
    
    plot_training_metrics(args.input, args.output_dir, args.dpi, args.fontsize)

if __name__ == "__main__":
    main()