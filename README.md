---
title: Agricultural Crop Classification System
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Advanced crop classification using self-supervised learning with Vision Transformers
---



# 🌾 Advanced Agricultural Crop Classification System

> **State-of-the-art crop classification using self-supervised learning with Vision Transformers**

**Developed by [Azizul Abedin Azmi](https://github.com/azizulabedinazmi)**  
[![GitHub](https://img.shields.io/badge/GitHub-azizulabedinazmi-blue?logo=github)](https://github.com/azizulabedinazmi)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--5269--6900-green?logo=orcid)](https://orcid.org/0009-0003-5269-6900)

## 🎯 Overview

This application demonstrates advanced agricultural crop classification using multiple self-supervised learning approaches. The system can identify **30 different crop types** with high accuracy using Vision Transformers and various contrastive learning methods.

## ✨ Key Features

### 🧠 **Multiple AI Models**
- **SimCLR v1**: Original contrastive learning with Vision Transformer backbone (default)
- **SimCLR v2**: Enhanced version with improved projection heads
- **NNCLR**: Nearest Neighbor Contrastive Learning
- **Barlow Twins**: Self-supervised learning via redundancy reduction

### 🌱 **30 Crop Classes Supported**
```
Almond • Banana • Cardamom • Cherry • Chilli • Clove • Coconut • Coffee
Cotton • Cucumber • Fox_Nut • Gram • Jowar • Jute • Lemon • Maize
Mung • Mustard • Olive • Papaya • Pearl_Millet • Pineapple • Rice
Soyabean • Sugarcane • Sunflower • Tea • Tobacco • Tomato • Wheat
```

### 🔬 **Advanced Features**
- **🔥 GradCAM Visualization**: See what the model focuses on during classification
- **📊 Real-time Analytics**: Confidence distribution charts and performance metrics
- **📦 Batch Processing**: Multiple model support with CSV export
- **📈 Prediction History**: Track all classifications with timestamps
- **📄 Export Options**: Download results as PDF/CSV reports
- **⚖️ Model Comparison**: Side-by-side performance analysis

## 🚀 How to Use

### 🔍 Single Image Classification
1. **Upload Image**: Drag & drop or browse for a crop image
2. **Select Model**: Choose from SimCLR v1/v2, NNCLR, or Barlow Twins (default: SimCLR v1)
3. **Set Parameters**: Choose top-K predictions (1-10) and confidence threshold
4. **Enable GradCAM**: Toggle visual explanations to see model attention
5. **Classify**: Get instant results with confidence scores!

### 🔥 GradCAM Visualization
- Toggle "Enable GradCAM" to see visual explanations
- View three panels: Original Image, Attention Heatmap, and Overlay
- Understand which parts of the image the model focuses on
- Perfect for model interpretability and debugging

### 📦 Batch Processing
1. **Upload Multiple**: Select multiple crop images at once
2. **Choose Models**: Pick one or more classification models
3. **Set Threshold**: Filter results by confidence level
4. **Process Batch**: Get results for all images simultaneously
5. **Download CSV**: Export comprehensive results spreadsheet

## 🎨 Interface Tabs

| Tab | Description | Features |
|-----|-------------|----------|
| 🔍 **Enhanced Classification** | Single image analysis | GradCAM, confidence charts, PDF export |
| 🔄 **Enhanced Model Comparison** | Compare models on same image | Side-by-side analysis |
| 📦 **Batch Processing** | Multiple image upload | Multi-model support, CSV export |
| 📊 **Analytics Dashboard** | Performance insights | History, metrics, trends |
| ⚖️ **Model Comparison** | Statistical comparison | Performance metrics |

## 🛠️ Technical Details

### Model Architecture
- **Backbone**: Vision Transformer (ViT-base-patch16-224.dino)
- **Training**: Self-supervised contrastive learning
- **Input**: 224×224 RGB images
- **Output**: 30-class crop classification with confidence scores

### Self-Supervised Learning Methods
- **SimCLR v1/v2**: Contrastive learning with data augmentation
- **NNCLR**: Nearest neighbor contrastive learning for better representations
- **Barlow Twins**: Redundancy reduction without negative samples

### Key Technologies
- **PyTorch**: Deep learning framework
- **timm**: Pre-trained Vision Transformers
- **Gradio**: Interactive web interface
- **GradCAM**: Explainable AI visualizations
- **SQLite**: Prediction history storage

## 🌾 Supported Crops

The system can classify the following crop categories:
- **Grains**: Rice, Wheat, Maize, Pearl_Millet, Jowar
- **Legumes**: Gram, Mung, Soyabean
- **Cash Crops**: Cotton, Tobacco, Sugarcane
- **Spices**: Chilli, Clove, Cardamom
- **Fruits**: Banana, Cherry, Lemon, Papaya, Pineapple
- **Nuts**: Almond, Coconut, Fox_Nut
- **Vegetables**: Cucumber, Tomato, Olive
- **Beverages**: Coffee, Tea
- **Fiber**: Jute
- **Others**: Mustard

## 📱 Usage Tips

### For Best Results:
- **Image Quality**: Use clear, high-resolution images
- **Lighting**: Natural daylight provides best accuracy
- **Crop Focus**: Ensure the crop is the main subject
- **Multiple Angles**: Try different perspectives for uncertain results

### Model Selection:
- **SimCLR v1**: Fast processing, good baseline (default)
- **SimCLR v2**: Best overall accuracy, recommended for high precision
- **NNCLR**: Good for distinguishing similar-looking crops
- **Barlow Twins**: Efficient processing, good for batch operations

## 🚀 Quick Start

### For Windows:
```cmd
# Clone the repository
git clone https://github.com/azizulabedinazmi/Crop-Classification.git
cd Crop-Classification

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### For Linux/Mac:
```bash
# Clone the repository
git clone https://github.com/azizulabedinazmi/Crop-Classification.git
cd Crop-Classification

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## 🌐 Access the App

After running `python app.py`, you'll see:
```
🌾 Starting Agricultural Crop Classification App...
🌐 Server: localhost:7860
🔗 Share: True
Running on local URL:  http://localhost:7860
Running on public URL: https://[random-id].gradio.live
```

- **Local:** Open `http://localhost:7860` in your browser
- **Share:** Use the `gradio.live` URL to share with others online

```bash
# Note: You'll need to add model files to respective folders:
# models/simclrv1/, models/simclrv2/, models/nnclr/, models/barlowtwins/
# Each folder should contain:
# - simclr_crop_model_final.pth (backbone)
# - crop_classifier_final.pth (classifier)
```

### What `run.bat` does:
- Activates the virtual environment (`.venv`)
- Launches the Gradio application
- Provides user-friendly startup messages

## 📁 Expected Project Structure

```
crop-classification/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── class_names.txt                 # 30 crop class labels
├── README.md                       # This file
├── run_app.bat                     # Windows startup script
├── .gitattributes                  # Git LFS configuration
└── models/                         # All model files
    ├── simclrv1/                   # SimCLR v1 model files
    │   ├── simclr_crop_model_final.pth
    │   └── crop_classifier_final.pth
    ├── simclrv2/                   # SimCLR v2 model files
    │   ├── simclr_crop_model_final.pth
    │   └── crop_classifier_final.pth
    ├── nnclr/                      # NNCLR model files
    │   ├── simclr_crop_model_final.pth
    │   └── crop_classifier_final.pth
    └── barlowtwins/                # Barlow Twins model files
        ├── simclr_crop_model_final.pth
        └── crop_classifier_final.pth
```

## 🤝 Contributing

This application demonstrates advanced agricultural AI techniques. Feel free to:
- Report issues with specific crop types
- Suggest improvements for model accuracy
- Contribute additional crop datasets
- Enhance the user interface

## 📄 License

MIT License - Feel free to use this for research and educational purposes.

## 👨‍💻 Developer

**Azizul Abedin Azmi**
- 🔗 GitHub: [azizulabedinazmi](https://github.com/azizulabedinazmi)
- 🎓 ORCID: [0009-0003-5269-6900](https://orcid.org/0009-0003-5269-6900)
- 🌐 Portfolio: Advanced Agricultural AI Systems

## 🙏 Acknowledgments

- **timm Library**: Pre-trained Vision Transformers
- **PyTorch Team**: Deep learning framework
- **Agricultural Research Community**: For inspiration and datasets

---

*Built with ❤️ for advancing agricultural AI and supporting farmers worldwide* 🌱
