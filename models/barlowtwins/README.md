# Barlow Twins Model Files

Place your trained model files here:

1. **simclr_crop_model_final.pth** - Self-supervised backbone model (~350MB)
2. **crop_classifier_final.pth** - Crop classification head model (~15MB)

## Training Information
- Model Type: Barlow Twins
- Architecture: Vision Transformer + Cross-Correlation Objective
- Expected Performance: ~91.9% accuracy
- Training Method: Self-supervised learning via redundancy reduction

## File Requirements
- Both files must be PyTorch state dictionaries (.pth format)
- Models should be trained on the same 30 crop classes as listed in class_names.txt
- Use CPU-compatible saving: `torch.save(model.state_dict(), path, map_location='cpu')`

## Verification
After placing files, you can verify they load correctly:

```python
import torch
simclr_model = torch.load('BarlowTwins_crop_model_final.pth', map_location='cpu')
classifier_model = torch.load('crop_barlowtwins_classifier_final.pth', map_location='cpu')
print("✅ Models loaded successfully")
```
