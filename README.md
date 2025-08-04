# HyperSpectral Imaging Crop Analysis

🌱 **Deep Learning Pipeline for Nitrogen Classification in Eggplant Crops using Hyperspectral Imaging**

This repository implements a complete end-to-end deep learning pipeline for classifying nitrogen (N2) concentration levels in eggplant crops using hyperspectral imaging data and 3D-ResNet architecture.

## 📊 Dataset

The hyperspectral imaging dataset used in this project is sourced from:

**Munipalle, V.K., Nidamanuri, R.R. (2024)**  
*"Ultra high resolution hyperspectral imagery datasets for precision agriculture applications"*  
Data in Brief, Volume 55, Article 110649  
DOI: [https://doi.org/10.1016/j.dib.2024.110649](https://doi.org/10.1016/j.dib.2024.110649)

**Dataset Access:** [Mendeley Data Repository](https://data.mendeley.com/datasets/t4rysh9rxf/1)

### Dataset Citation
```
Munipalle, V.K., Nidamanuri, R.R. (2024). Ultra high resolution hyperspectral imagery datasets for precision agriculture applications. Data in Brief, 55, 110649. https://doi.org/10.1016/j.dib.2024.110649
```

**Corresponding Authors:**
- V.K. Munipalle: mvk.6518@gmail.com  
- R.R. Nidamanuri: rao@iist.ac.in (https://www.iist.ac.in/ess/rao)

*License: CC BY-NC-ND 4.0 (http://creativecommons.org/licenses/by-nc-nd/4.0/)*

## 🎯 Project Overview

This project demonstrates a comprehensive hyperspectral image analysis pipeline for precision agriculture, specifically targeting nitrogen deficiency detection in eggplant crops. The methodology combines state-of-the-art deep learning techniques with hyperspectral imaging for accurate crop health assessment.

### Key Features
- **3D-ResNet Architecture**: Optimized for hyperspectral data with spatial-spectral feature extraction
- **GPU Acceleration**: CUDA-enabled PyTorch implementation for efficient training
- **Complete Pipeline**: From raw data loading to field-scale inference mapping
- **Agricultural Application**: Practical nitrogen classification for precision farming

## 🛠️ Technical Specifications

- **Framework**: PyTorch 2.5.1 with CUDA 12.1 support
- **Environment**: Python 3.10.18 in Anaconda environment
- **GPU**: NVIDIA RTX 4070 SUPER (12GB VRAM)
- **Data Format**: ENVI hyperspectral format (.hdr + binary)
- **Image Dimensions**: 545 × 5,382 pixels × 277 spectral bands
- **Classification Classes**: 3 classes (Low N2, Medium N2, High N2)

## 📋 Workflow - Sequential Execution Steps

Execute the following scripts in order for complete pipeline implementation:

### Step 1: Environment Setup
```bash
# Create and activate Anaconda environment
conda create -n plantgpu python=3.10
conda activate plantgpu

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install spectral scikit-learn matplotlib tqdm opencv-python scikit-image
```

### Step 2: Data Exploration
```bash
python step2_data_exploration.py
```
**Purpose**: Load and analyze hyperspectral data structure, visualize RGB composites, ground truth maps, and spectral signatures.

**Outputs**: 
- Data structure analysis
- 6-panel visualization (RGB composite, ground truth, class distribution, spectral signatures)
- Statistical summary

### Step 3-4: Data Preprocessing & Dataset Creation
```bash
python step3_4_preprocessing_dataset.py
```
**Purpose**: Extract 9×9×277 patches, normalize data, create balanced training/validation/test splits.

**Outputs**:
- `processed_data/X_train.npy` (120,000 patches)
- `processed_data/X_val.npy` (15,000 patches) 
- `processed_data/X_test.npy` (15,000 patches)
- `processed_data/metadata.pkl`

### Step 5: 3D-ResNet Training
```bash
python step5_3d_resnet_pytorch.py
```
**Purpose**: Train 3D-ResNet model with GPU acceleration, early stopping, and model checkpointing.

**Key Results**:
- Training Time: ~44.5 minutes (GPU accelerated)
- Validation Accuracy: 91.20%
- Model saved: `models/best_3d_resnet_pytorch.pth`

### Step 6: Model Evaluation
```bash
python step6_model_evaluation.py
```
**Purpose**: Comprehensive model performance evaluation with confusion matrix and confidence analysis.

**Outputs**:
- Confusion matrix visualization
- Per-class performance metrics
- Prediction confidence analysis
- Error distribution plots

### Step 7: Inference & Field Mapping
```bash
python step7_inference_mapping.py
```
**Purpose**: Apply trained model to entire hyperspectral image for field-scale nitrogen classification mapping.

**Features**:
- Optimized batch processing (stride sampling)
- GPU-accelerated inference
- Confidence mapping
- Agricultural visualization (Yellow=Low N2, Green=Medium N2, Blue=High N2)

**Outputs**:
- `models/nitrogen_classification_map.png`
- `models/detailed_classification_analysis.png`
- `results/classification_map.npy`
- `results/classification_stats.pkl`

## 📁 Project Structure

```
HyperSpectral_Imaging_Crop_Analysis/
├── README.md                           # This file
├── step2_data_exploration.py           # HSI data loading and analysis
├── step3_4_preprocessing_dataset.py    # Data preprocessing and patch extraction
├── step5_3d_resnet_pytorch.py         # 3D-ResNet model training
├── step6_model_evaluation.py          # Model performance evaluation
├── step7_inference_mapping.py         # Field-scale inference mapping
├── Eggplant_Crop/                     # Dataset directory
│   └── Eggplant_Crop/
│       ├── Eggplant_Reflectance_Data  # HSI data file
│       ├── Eggplant_Reflectance_Data.hdr
│       ├── Eggplant_N2_Concentration_GT # Ground truth file
│       └── Eggplant_N2_Concentration_GT.hdr
├── processed_data/                    # Generated preprocessed data
├── models/                           # Trained models and visualizations
├── results/                          # Final classification results
└── Literature_Review/                # Research papers and references
```

## 🔧 Requirements

### Python Dependencies
```python
torch==2.5.1
torchvision>=0.18.1
spectral>=0.24
scikit-learn>=1.3.0
matplotlib>=3.7.0
numpy>=1.24.0
tqdm>=4.65.0
opencv-python>=4.8.0
scikit-image>=0.21.0
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free disk space

## 🎯 Results Summary

### Model Performance
- **Validation Accuracy**: 91.20%
- **Training Time**: 44.5 minutes (GPU accelerated)
- **Inference Speed**: ~2,400 pixels/second

### Field-Scale Classification
- **Image Coverage**: 545 × 5,382 pixels (2.9M pixels total)
- **Processing Mode**: Optimized sampling (320K-500K pixels)
- **Output Resolution**: Practical for agricultural decision-making
- **Confidence Mapping**: Quality assessment for predictions

## 🚀 Quick Start

1. **Clone Repository**:
```bash
git clone https://github.com/Noiceboi/HyperSpectral_Imaging_Crop_Analysis.git
cd HyperSpectral_Imaging_Crop_Analysis
```

2. **Download Dataset**:
   - Visit [Mendeley Data Repository](https://data.mendeley.com/datasets/t4rysh9rxf/1)
   - Download eggplant hyperspectral dataset
   - Extract to `Eggplant_Crop/Eggplant_Crop/` directory

3. **Setup Environment**:
```bash
conda create -n plantgpu python=3.10
conda activate plantgpu
pip install -r requirements.txt  # Create requirements.txt from above dependencies
```

4. **Execute Pipeline**:
```bash
python step2_data_exploration.py
python step3_4_preprocessing_dataset.py  
python step5_3d_resnet_pytorch.py
python step6_model_evaluation.py
python step7_inference_mapping.py
```

## 🔬 Scientific Methodology

This implementation follows the research methodology outlined in precision agriculture applications using hyperspectral imaging. The 3D-ResNet architecture effectively captures both spatial and spectral features for accurate nitrogen classification.

### Key Technical Innovations
- **3D Convolutional Architecture**: Optimized for hyperspectral data structure
- **Spatial-Spectral Feature Fusion**: 9×9 spatial patches with 277 spectral bands
- **Agricultural-Specific Optimization**: Balanced dataset with field-relevant classes
- **Practical Deployment**: Optimized inference for real-world agricultural applications

## 📜 License & Citation

If you use this code or methodology in your research, please cite both this repository and the original dataset:

### Repository Citation
```
@software{hyperspectral_eggplant_analysis,
  title={HyperSpectral Imaging Crop Analysis: 3D-ResNet for Nitrogen Classification},
  author={[Your Name]},
  year={2025},
  url={https://github.com/Noiceboi/HyperSpectral_Imaging_Crop_Analysis}
}
```

### Dataset Citation
```
@article{munipalle2024ultra,
  title={Ultra high resolution hyperspectral imagery datasets for precision agriculture applications},
  author={Munipalle, V.K. and Nidamanuri, R.R.},
  journal={Data in Brief},
  volume={55},
  pages={110649},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.dib.2024.110649}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Pull Requests or open Issues for:
- Algorithm improvements
- Additional visualization features
- Performance optimizations
- Documentation enhancements

## 📞 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**⚠️ Note**: This project is for research and educational purposes. Ensure you have proper permissions and comply with the dataset's license terms (CC BY-NC-ND 4.0) when using for commercial applications.

**🌱 Impact**: Supporting precision agriculture through advanced hyperspectral imaging analysis for sustainable crop management and optimized fertilizer application.
