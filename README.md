# 3DRDN-SRGAN
## Introduction
This repository contains the implementation of my bachelor thesis "[Super Resolution of Brain MRI using GANs](https://drive.google.com/file/d/19hfe7AdcNJk70DWP_XEG5a1iWYtiKxqg/view?usp=sharing)", in which a 3D CNN integrated into a WGAN-GP architecture is used to perform super-resolution on 3D T1w MRI scans.

## Data Format Requirements
The image scans must be in the 'nii' format and should be defined in `src/preprocessing/data_preprocessing.py`, where the program will recursively search through all folders to gather all available images.

## Quick Start
```bash
git clone https://github.com/Binhdn04/3DRDN-SRGAN.git
cd 3DRDN-SRGAN
make build && make run
```
## Environment Setup

### Option 1: Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate myenv
```

### Option 2: Pip
```bash
pip install -r requirements.txt
```

### Option 3: Docker + Makefile (Simplest)
```bash
make build   # Build image
make run     # Run container
```

## Makefile Usage

| Command | Description |
|---------|-------------|
| `make build` | Build Docker image |
| `make run` | Run container (interactive) |
| `make run-bg` | Run container in background |
| `make exec` | Execute bash in container |
| `make stop` | Stop & remove container |
| `make clean` | Remove image + container |

## Training

Modify `configs/default.yaml` to set hyperparameters (learning rate, batch size, dataset path, etc.), then run:

```bash
python main.py
```

## Inference

```bash
python inference_folder.py
```

## Contact 
If you have any questions, please file an issue or reach me via email:
```
Doan Nhat Binh: binhdn.uet@gmail.com
```

## 📖 Citation
If my work is useful for your research, please consider citing it:

```bibtex
@misc{binh2025superMRI,
  title        = {Super Resolution of Brain MRI using GANs},
  author       = {DB Binh, ND Kien, NL Trung, G Nagels},
  year         = {2025},
  howpublished = {\url{https://github.com/Binhdn04/3DRDN-SRGAN}},
  note         = {Bachelor Thesis Implementation}
}
```
## 🙏 Acknowledgments

This project was inspired by and builds upon the work of [Omar Hussein](https://github.com/omagdy/3DRDN-CycleGAN). Their open-source implementation provided valuable insights and served as the starting point for our enhancements.
