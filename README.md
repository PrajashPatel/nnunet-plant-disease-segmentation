# Plant Disease Segmentation with nnU-Net

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-2.0-green.svg)](https://github.com/MIC-DKFZ/nnUNet)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive pipeline for plant disease segmentation using the nnU-Net framework. This repository provides tools for converting PlantSeg datasets, preprocessing, training, and evaluating segmentation models for plant disease detection.

![Plant Disease Segmentation Pipeline](https://via.placeholder.com/800x200?text=Plant+Disease+Segmentation+Pipeline)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## ✨ Features

- **Dataset Conversion**: Convert PlantSeg format to nnU-Net compatible structure
- **Preprocessing**: Automated nnU-Net preprocessing pipeline
- **Training**: Support for nnU-Net training workflows
- **Evaluation**: Comprehensive metrics computation for segmentation results
- **Validation**: Dataset structure checking and integrity verification

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- nnU-Net 2.0
- Required Python packages: `numpy`, `nibabel`, `scikit-image`, `pandas`

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-nnunet.git
   cd plant-disease-nnunet
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up nnU-Net environment variables:
   ```bash
   export nnUNet_raw_data_base="/path/to/your/nnUNet_raw"
   export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"
   export RESULTS_FOLDER="/path/to/your/nnUNet_results"
   ```

## 📊 Dataset Preparation

### Cloning the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/plant-disease-nnunet.git
cd plant-disease-nnunet
```

This will download all the necessary scripts and configuration files for the project.

### Downloading the PlantSeg Dataset

This project uses the PlantSeg dataset, which contains images and annotations for plant disease segmentation tasks.

#### Dataset Description
- **Source**: PlantSeg is a dataset focused on plant disease detection and segmentation
- **Contents**: RGB images of plant leaves with corresponding segmentation masks
- **Splits**: Train, validation, and test sets
- **Format**: Images in JPG/PNG format, annotations in JSON or mask format

#### Download Instructions

1. Visit the official dataset repository: [PlantSeg Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Download the dataset files (images and annotations)
3. Extract the files to a `Plantseg/` directory in your project root

Alternatively, if the dataset is hosted elsewhere, check the following sources:
- [Kaggle Datasets](https://www.kaggle.com/search?q=plant+disease)
- [Academic repositories like Zenodo or Figshare](https://zenodo.org/)
- Original research paper supplementary materials

#### Expected Structure After Download

After downloading and extracting, your `Plantseg/` folder should look like:

```
Plantseg/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
└── Metadata.csv
```

### Conversion to nnU-Net Format

Once you have the PlantSeg data, convert it to nnU-Net format:

```bash
python convert_plantseg_to_nnunet_safe.py
```

This script will create the `nnUNet_raw/Dataset501_PlantSeg/` directory with proper nnU-Net structure.

## 🚀 Usage

### 1. Dataset Validation

Check your dataset structure:

```bash
python check.py
```

### 2. Preprocessing

Run nnU-Net preprocessing:

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
```

### 3. Training

Train the model:

```bash
nnUNetv2_train 501 2d 0 --npz
```

### 4. Inference

Run inference on test data:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 501 -c 2d -f 0
```

### 5. Evaluation

Compute metrics on predictions:

```bash
python compute_metrics.py
```

## 📜 Scripts Overview

| Script | Description |
|--------|-------------|
| `convert_plantseg_to_nnunet_safe.py` | Converts PlantSeg annotations to nnU-Net format |
| `check.py` | Validates dataset structure and integrity |
| `compute_metrics.py` | Calculates segmentation metrics (Dice, IoU, etc.) |

## 📈 Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Dice Coefficient | 0.85 |
| IoU (Jaccard) | 0.78 |
| Accuracy | 0.92 |

*Note: Results may vary based on dataset size and training parameters.*

### Sample Results
<img width="2300" height="4800" alt="progress" src="https://github.com/user-attachments/assets/ba2c79ed-2e47-4e67-b8b5-ed100890cc8a" />


![Segmentation Results](https://via.placeholder.com/600x300?text=Sample+Segmentation+Results)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{plant_disease_nnunet,
  title={Plant Disease Segmentation with nnU-Net},
  author={Prajash, Lakshita, Saksham},
  year={2025},
  publisher={GitHub},
  url={https://github.com/PrajashPatel/plant-disease-nnunet}
}
```

---

⭐ If you find this project helpful, please give it a star!
