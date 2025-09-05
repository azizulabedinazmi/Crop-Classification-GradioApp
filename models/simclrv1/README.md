# SimCLR v1 Model Files

Place your trained model files here:

1. **simclr_crop_model_final.pth** - Self-supervised backbone model (~350MB)
2. **crop_classifier_final.pth** - Crop classification head model (~15MB)

## Training Information
- Model Type: SimCLR v1 
- Architecture: ResNet + Contrastive Learning
- Expected Performance: ~92.3% accuracy
- Training Method: Original SimCLR contrastive learning

## File Requirements
- Both files must be PyTorch state dictionaries (.pth format)
- Models should be trained on the same 30 crop classes as listed in class_names.txt
- Use CPU-compatible saving: `torch.save(model.state_dict(), path, map_location='cpu')`

## Verification
After placing files, you can verify they load correctly:

```python
import torch
simclr_model = torch.load('simclrv1_crop_model_final.pth', map_location='cpu')
classifier_model = torch.load('crop_simclrv1_classifier_final.pth', map_location='cpu')
print("✅ Models loaded successfully")
```
