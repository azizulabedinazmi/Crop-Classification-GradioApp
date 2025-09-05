import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
import timm
import gradio as gr
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
import pandas as pd
import datetime
import io
import base64
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import cv2
from fpdf import FPDF
import sqlite3
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
FEATURE_DIM = 256
NUM_CLASSES = 30

# Initialize database for prediction history
def init_database():
    """Initialize SQLite database for prediction history"""
    try:
        conn = sqlite3.connect('prediction_history.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      image_name TEXT,
                      model_used TEXT,
                      top_prediction TEXT,
                      confidence REAL,
                      all_predictions TEXT)''')
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# Initialize database
init_database()

class SimCLRv1(nn.Module):
    """Original SimCLR with Vision Transformer backbone"""
    def __init__(self, hidden_dim=2048, feat_dim=FEATURE_DIM):
        super().__init__()
        
        # Vision Transformer backbone
        self.encoder = timm.create_model(
            'vit_base_patch16_224.dino',
            pretrained=True,
            num_classes=0,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        
        # Simple projection head for SimCLR v1 (matching saved model)
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, hidden_dim),  # 768 -> 2048
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feat_dim)                 # 2048 -> 256
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return projections
    
    def extract_features(self, x):
        return self.encoder(x)

class ImprovedSimCLRv2(nn.Module):
    """Improved SimCLRv2 with Vision Transformer backbone"""
    def __init__(self, hidden_dim=2048, feat_dim=FEATURE_DIM):
        super().__init__()
        
        # Vision Transformer backbone
        self.encoder = timm.create_model(
            'vit_base_patch16_224.dino',
            pretrained=True,
            num_classes=0,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        
        # Enhanced projection head
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feat_dim),
            nn.BatchNorm1d(feat_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return projections
    
    def extract_features(self, x):
        return self.encoder(x)

class NNCLR(nn.Module):
    """NNCLR (Nearest Neighbor Contrastive Learning of visual Representations)"""
    def __init__(self, hidden_dim=2048, feat_dim=FEATURE_DIM, queue_size=65536):
        super().__init__()
        
        # Vision Transformer backbone
        self.encoder = timm.create_model(
            'vit_base_patch16_224.dino',
            pretrained=True,
            num_classes=0,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        
        # Projection head (matching the saved model structure)
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, hidden_dim),  # 768 -> 2048
            nn.BatchNorm1d(hidden_dim),                     # BN for 2048
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),              # 2048 -> 2048
            nn.BatchNorm1d(hidden_dim),                     # BN for 2048
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),                # 2048 -> 256
            nn.BatchNorm1d(feat_dim)                        # BN for 256
        )
        
        # Predictor head (specific to NNCLR) - matching saved dimensions
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, 512),          # 256 -> 512
            nn.BatchNorm1d(512),               # BN for 512
            nn.ReLU(),
            nn.Linear(512, feat_dim)           # 512 -> 256
        )
        
        # Queue for storing representations
        self.register_buffer("queue", torch.randn(feat_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        predictions = self.predictor(projections)
        return predictions
    
    def extract_features(self, x):
        return self.encoder(x)

class BarlowTwins(nn.Module):
    """Barlow Twins implementation with ViT-Base backbone"""
    def __init__(self, hidden_dim=512, feat_dim=64):  # Updated to match saved model
        super().__init__()
        
        # Vision Transformer backbone - ViT Base for Barlow Twins
        self.encoder = timm.create_model(
            'vit_base_patch16_224.dino',  # ViT-Base to match saved model
            pretrained=True,
            num_classes=0,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        
        # Projection head for Barlow Twins - matching saved model architecture
        self.projection = nn.Sequential(
            nn.Linear(768, hidden_dim),     # Layer 0: 768 -> 512 (ViT-Base to hidden)
            nn.BatchNorm1d(hidden_dim),     # Layer 1: BN for 512
            nn.ReLU(),                      # Layer 2: ReLU activation
            nn.Dropout(0.1),                # Layer 3: Dropout layer
            nn.Linear(hidden_dim, feat_dim) # Layer 4: 512 -> 64 (final projection)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return projections
    
    def extract_features(self, x):
        return self.encoder(x)

class EnhancedCropClassifier(nn.Module):
    """Enhanced classifier for crop classification"""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        
        # Vision Transformer backbone
        self.backbone = timm.create_model(
            "vit_base_patch16_224.dino",
            pretrained=True,
            num_classes=0,
            drop_rate=dropout
        )
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.embed_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout/4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def extract_features(self, x):
        return self.backbone(x)

class BarlowTwinsClassifier(nn.Module):
    """Enhanced classifier for Barlow Twins with ViT-Base backbone"""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        
        # Vision Transformer backbone - ViT Base for Barlow Twins Classifier
        self.backbone = timm.create_model(
            "vit_base_patch16_224.dino",  # ViT-Base to match saved classifier
            pretrained=True,
            num_classes=0,
            drop_rate=dropout
        )
        
        # Enhanced classifier head (adjusted for ViT-Base 768 dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 512),  # ViT-Base has 768 dimensions
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout/4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def extract_features(self, x):
        return self.backbone(x)

class GradCAM:
    """GradCAM implementation optimized for Vision Transformers"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Save forward activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Save backward gradients"""
        if grad_output[0] is not None:
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = None

    def __call__(self, x, class_idx=None):
        """Generate GradCAM heatmap"""
        # Set to eval mode to avoid BatchNorm issues with single samples
        original_mode = self.model.training
        self.model.eval()
        
        # Enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad_(True)
    
        if not x.requires_grad:
            x.requires_grad_(True)
    
        # Forward pass
        output = self.model(x)
    
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
    
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
    
        if not class_score.requires_grad:
            return np.zeros((224, 224))
    
        class_score.backward(retain_graph=True)
    
        # Check if hooks captured data
        if self.activations is None or self.gradients is None:
            return np.zeros((224, 224))
    
        # Handle different possible shapes
        activations = self.activations[0]  # Remove batch dimension
        gradients = self.gradients[0]      # Remove batch dimension
    
        # Remove CLS token if present (first token)
        if activations.shape[0] > 196:  # Has CLS token
            activations = activations[1:, :]  # [196, 768]
            gradients = gradients[1:, :]      # [196, 768]
    
        # Ensure we have the expected dimensions
        if len(activations.shape) != 2 or len(gradients.shape) != 2:
            return np.zeros((224, 224))
    
        if activations.shape != gradients.shape:
            return np.zeros((224, 224))
    
        # Calculate importance weights for each feature dimension
        weights = gradients.mean(dim=0)  # [768] - average importance per feature
    
        # Apply weights to activations and sum across feature dimension
        weighted_activations = activations * weights  # [196, 768] * [768] -> [196, 768]
        cam = weighted_activations.sum(dim=1)         # [196, 768] -> [196]
    
        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)
    
        # Check if we have valid values
        if torch.all(cam == 0):
            return np.zeros((224, 224))
    
        # Reshape to spatial grid
        num_patches = cam.shape[0]
        patch_size = int(np.sqrt(num_patches))
    
        if patch_size * patch_size != num_patches:
            return np.zeros((224, 224))
    
        # Convert to numpy and reshape
        cam_np = cam.detach().cpu().numpy()
        cam_2d = cam_np.reshape(patch_size, patch_size)
    
        # Resize to input resolution
        cam_resized = cv2.resize(cam_2d, (224, 224))
    
        # Normalize to [0, 1]
        if cam_resized.max() > cam_resized.min():
            cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
        else:
            cam_normalized = np.zeros_like(cam_resized)
    
        # Restore original training mode
        self.model.train(original_mode)
        return cam_normalized

    def cleanup(self):
        """Remove hooks"""
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()

class CropClassificationApp:
    """Advanced Crop Classification Application with Enhanced Features"""
    
    def __init__(self):
        self.models = {}
        self.class_names = self.load_class_names()
        self.transform = self.create_transform()
        self.feature_cache = {}  # For similarity search
        self.prediction_history = []
        self.gradcam_instances = {}
        
        self.model_info = {
            'simclrv1': {
                'name': 'SimCLR v1',
                'description': 'Original SimCLR with Vision Transformer backbone',
                'type': 'SimCLR'
            },
            'simclrv2': {
                'name': 'SimCLR v2',
                'description': 'Enhanced SimCLR with Vision Transformer backbone',
                'type': 'SimCLR'
            },
            'nnclr': {
                'name': 'NNCLR',
                'description': 'Nearest Neighbor Contrastive Learning with Vision Transformer backbone',
                'type': 'NNCLR'
            },
            'barlowtwins': {
                'name': 'Barlow Twins',
                'description': 'Self-supervised learning via redundancy reduction with Vision Transformer backbone',
                'type': 'BarlowTwins'
            }
        }
        
        print(f"🌾 Initializing Enhanced Crop Classification App on {DEVICE}")
        print(f"📊 Number of crop classes: {len(self.class_names)}")
        
    def load_class_names(self) -> List[str]:
        """Load class names from file"""
        try:
            with open('class_names.txt', 'r') as f:
                class_names = [line.strip() for line in f.readlines() if line.strip()]
            
            print(f"📋 Loaded {len(class_names)} classes from class_names.txt")
            print(f"🔍 First 10 classes: {class_names[:10]}")
            
            if len(class_names) != NUM_CLASSES:
                print(f"⚠️ Warning: Found {len(class_names)} classes but NUM_CLASSES is set to {NUM_CLASSES}")
                print("This mismatch could cause incorrect predictions!")
            
            return class_names
        except FileNotFoundError:
            print("⚠️ class_names.txt not found, using default classes")
            return [f"Crop_{i+1}" for i in range(NUM_CLASSES)]
    
    def create_transform(self):
        """Create image preprocessing transform for Vision Transformer"""
        return T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            # ViT standard normalization (same as ImageNet pre-trained models)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def save_prediction_to_db(self, image_name, model_used, predictions):
        """Save prediction to database"""
        conn = sqlite3.connect('prediction_history.db')
        c = conn.cursor()
        
        timestamp = datetime.datetime.now().isoformat()
        top_prediction = predictions[0]['class']
        confidence = predictions[0]['confidence']
        all_predictions = json.dumps(predictions)
        
        c.execute("""INSERT INTO predictions 
                     (timestamp, image_name, model_used, top_prediction, confidence, all_predictions)
                     VALUES (?, ?, ?, ?, ?, ?)""",
                  (timestamp, image_name, model_used, top_prediction, confidence, all_predictions))
        
        conn.commit()
        conn.close()
    
    def get_prediction_history(self, limit=100):
        """Get prediction history from database"""
        conn = sqlite3.connect('prediction_history.db')
        c = conn.cursor()
        
        c.execute("""SELECT timestamp, image_name, model_used, top_prediction, confidence 
                     FROM predictions ORDER BY timestamp DESC LIMIT ?""", (limit,))
        
        history = c.fetchall()
        conn.close()
        
        return pd.DataFrame(history, columns=['Timestamp', 'Image', 'Model', 'Prediction', 'Confidence'])
    
    def calculate_model_performance_metrics(self):
        """Calculate performance metrics for each model and crop"""
        conn = sqlite3.connect('prediction_history.db')
        
        # Get all predictions
        df = pd.read_sql_query("""SELECT model_used, top_prediction, confidence 
                                  FROM predictions""", conn)
        conn.close()
        
        if df.empty:
            return None
        
        # Calculate metrics per model
        model_stats = {}
        for model in df['model_used'].unique():
            model_data = df[df['model_used'] == model]
            model_stats[model] = {
                'total_predictions': len(model_data),
                'avg_confidence': model_data['confidence'].mean(),
                'crop_distribution': model_data['top_prediction'].value_counts().to_dict()
            }
        
        return model_stats
    
    def process_batch_images(self, image_files, model_types, confidence_threshold=0.0):
        """Process multiple images with multiple models and return results as CSV with confidence filtering"""
        print(f"🔄 Starting batch processing: {len(image_files)} images, models: {model_types}, threshold: {confidence_threshold}")
        
        if not image_files:
            print("❌ No image files provided")
            return None, "No images provided"
        
        # Ensure model_types is a list
        if isinstance(model_types, str):
            model_types = [model_types]
            
        # Validate all models are loaded
        for model_type in model_types:
            if model_type not in self.models:
                print(f"❌ Model not loaded: {model_type}. Available: {list(self.models.keys())}")
                return None, f"Model {model_type} not loaded"
        
        results = []
        successful_count = 0
        
        for i, image_data in enumerate(image_files):
            try:
                print(f"🖼️ Processing image {i+1}/{len(image_files)} with threshold {confidence_threshold}...")
                
                # Handle different input formats
                if isinstance(image_data, tuple):
                    # (image, filename) tuple
                    image, image_name = image_data
                    print(f"   📁 Tuple format: {image_name}")
                elif hasattr(image_data, 'name'):
                    # File object with name
                    image_name = image_data.name
                    image = Image.open(image_data)
                    print(f"   📁 File object: {image_name}")
                else:
                    # Direct image or file path
                    image_name = f"image_{i+1}"
                    if isinstance(image_data, str):
                        image = Image.open(image_data)
                        print(f"   📁 File path: {image_data}")
                    else:
                        image = image_data
                        print(f"   📁 Direct image: {type(image_data)}")
                
                # Ensure image is PIL Image
                if not isinstance(image, Image.Image):
                    print(f"   ❌ Not a PIL Image: {type(image)}")
                    continue
                
                print(f"   🔍 Image size: {image.size}, mode: {image.mode}")
                
                # Process with each model
                image_results = {"Image": image_name}
                
                for model_type in model_types:
                    # Classify image with current model
                    result = self.classify_image(image, model_type, top_k=3)
                    print(f"   🤖 {model_type} result: {result.get('predictions', [])[:1] if result else 'None'}")
                    
                    model_name = self.model_info[model_type]['name']
                    
                    if 'predictions' in result and result['predictions']:
                        # Filter predictions based on confidence threshold
                        filtered_predictions = [
                            pred for pred in result['predictions'] 
                            if pred['confidence'] >= confidence_threshold
                        ]
                        
                        if filtered_predictions:
                            # Use filtered predictions
                            top_pred = filtered_predictions[0]
                            second_pred = filtered_predictions[1] if len(filtered_predictions) > 1 else {"class": "N/A", "confidence": 0}
                            third_pred = filtered_predictions[2] if len(filtered_predictions) > 2 else {"class": "N/A", "confidence": 0}
                            
                            # Add threshold indicator to class name if threshold applied
                            threshold_indicator = f" (>{confidence_threshold*100:.0f}%)" if confidence_threshold > 0 else ""
                            
                            # Add model-specific columns with corrected confidence calculation
                            image_results[f"{model_name}_Top_Prediction"] = top_pred["class"] + threshold_indicator
                            image_results[f"{model_name}_Confidence"] = f"{top_pred['confidence']*100:.1f}%"
                            image_results[f"{model_name}_Confidence_Raw"] = top_pred["confidence"] * 100
                            image_results[f"{model_name}_Second_Prediction"] = f"{second_pred['class']} ({second_pred['confidence']*100:.1f}%)"
                            image_results[f"{model_name}_Third_Prediction"] = f"{third_pred['class']} ({third_pred['confidence']*100:.1f}%)"
                            
                            # Save to database (original predictions, not filtered)
                            self.save_prediction_to_db(image_name, model_type, result['predictions'])
                            
                            print(f"   ✅ {model_name}: {top_pred['class']} ({top_pred['confidence']*100:.1f}%) - meets threshold")
                        else:
                            # No predictions meet threshold
                            original_top = result['predictions'][0]
                            image_results[f"{model_name}_Top_Prediction"] = f"Below threshold ({original_top['class']})"
                            image_results[f"{model_name}_Confidence"] = f"{original_top['confidence']*100:.1f}% (below {confidence_threshold*100:.0f}%)"
                            image_results[f"{model_name}_Confidence_Raw"] = original_top["confidence"] * 100
                            image_results[f"{model_name}_Second_Prediction"] = "Below threshold"
                            image_results[f"{model_name}_Third_Prediction"] = "Below threshold"
                            
                            print(f"   ⚠️ {model_name}: {original_top['class']} ({original_top['confidence']*100:.1f}%) - below threshold")
                    else:
                        # Handle errors for this model
                        image_results[f"{model_name}_Top_Prediction"] = "Error"
                        image_results[f"{model_name}_Confidence"] = "0.0%"
                        image_results[f"{model_name}_Confidence_Raw"] = 0.0
                        image_results[f"{model_name}_Second_Prediction"] = "N/A"
                        image_results[f"{model_name}_Third_Prediction"] = "N/A"
                        print(f"   ❌ {model_name}: No predictions returned")
                
                # Add timestamp
                image_results["Timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                results.append(image_results)
                successful_count += 1
                    
            except Exception as e:
                print(f"   ❌ Error processing image {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"📊 Batch processing completed: {successful_count}/{len(image_files)} images successful")
        
        if results:
            df = pd.DataFrame(results)
            
            # Create batch_processing directory if it doesn't exist
            batch_dir = "batch_processing"
            os.makedirs(batch_dir, exist_ok=True)
            
            # Create CSV filename with timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            models_str = "_".join([m.replace(" ", "") for m in model_types])
            csv_filename = f"batch_results_{models_str}_{timestamp}.csv"
            csv_filepath = os.path.join(batch_dir, csv_filename)
            
            # Save CSV file
            df.to_csv(csv_filepath, index=False)
            print(f"✅ CSV saved to: {csv_filepath}")
            
            return df, csv_filepath
        else:
            print("❌ No valid results generated")
            return None, "No valid results generated"
    
    def find_similar_images(self, query_image, model_type: str, top_k: int = 5):
        """Find similar images using feature similarity"""
        if model_type not in self.models:
            return None, "Model not loaded"
        
        # Extract features from query image
        query_features = self.extract_image_features(query_image, model_type)
        
        # Load or compute feature cache
        if not self.feature_cache:
            print("Building feature cache... This may take a moment.")
            # In a real implementation, you would load pre-computed features
            # For now, return a placeholder
            return None, "Feature cache not available. Please build the cache first."
        
        # Calculate similarities (placeholder implementation)
        similarities = []
        # This would be implemented with actual feature comparison
        
        return similarities, "Similarity search completed"
    
    def extract_image_features(self, image: Image.Image, model_type: str):
        """Extract features from an image for similarity search"""
        if model_type not in self.models:
            return None
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        
        # Get model
        simclr_model = self.models[model_type]['simclr']
        
        # Extract features
        with torch.no_grad():
            features = simclr_model.extract_features(input_tensor)
        
        return features.cpu().numpy()
    
    def create_performance_metrics_chart(self):
        """Create clean, professional performance metrics visualization"""
        model_stats = self.calculate_model_performance_metrics()
        
        if not model_stats:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No performance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                font=dict(size=16, color="gray")
            )
            fig.update_layout(template="plotly_white")
            return fig
        
        # Prepare data for metrics comparison
        models = list(model_stats.keys())
        avg_confidences = [model_stats[model]['avg_confidence'] * 100 for model in models]  # Convert to percentage
        total_predictions = [model_stats[model]['total_predictions'] for model in models]
        
        # Calculate usage rates
        max_predictions = max(total_predictions) if total_predictions else 1
        usage_rates = [pred / max_predictions * 100 for pred in total_predictions]
        
        # Simple, clean colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)]
        
        # Create the figure
        fig = go.Figure()
        
        # Add Average Confidence bars
        model_names_display = [self.model_info.get(model, {}).get('name', model) for model in models]
        
        fig.add_trace(go.Bar(
            x=model_names_display,
            y=avg_confidences,
            name='Average Confidence (%)',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f'{conf:.1f}%' for conf in avg_confidences],
            textposition='outside',
            textfont=dict(size=11, color='black'),
            hovertemplate='<b>%{x}</b><br>Average Confidence: %{y:.1f}%<extra></extra>',
            offsetgroup=1
        ))
        
        # Add Usage Rate bars with transparency
        fig.add_trace(go.Bar(
            x=model_names_display,
            y=usage_rates,
            name='Usage Rate (%)',
            marker=dict(
                color=[f'rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, 0.6)' for c in colors],
                line=dict(color='white', width=1)
            ),
            text=[f'{rate:.1f}%' for rate in usage_rates],
            textposition='outside',
            textfont=dict(size=11, color='black'),
            hovertemplate='<b>%{x}</b><br>Usage Rate: %{y:.1f}%<br>Total Predictions: %{customdata}<extra></extra>',
            customdata=total_predictions,
            offsetgroup=2
        ))
        
        # Clean, professional layout
        fig.update_layout(
            title=dict(
                text='Model Performance Metrics Dashboard',
                font=dict(size=16, color='#333'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(
                    text='Model Architecture',
                    font=dict(size=14, color='#333')
                ),
                tickangle=45,
                tickfont=dict(size=11, color='#333'),
                showline=True,
                linecolor='gray',
                linewidth=1
            ),
            yaxis=dict(
                range=[0, 110],
                title=dict(
                    text='Performance Score (%)',
                    font=dict(size=14, color='#333')
                ),
                gridcolor='lightgray',
                gridwidth=1,
                tickfont=dict(size=11, color='#333'),
                showline=True,
                linecolor='gray',
                linewidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=60, t=80, b=120),
            height=500,
            font=dict(family='Arial'),
            barmode='group',
            bargap=0.3,
            bargroupgap=0.1,
            legend=dict(
                title=dict(
                    text="Performance Metrics",
                    font=dict(size=12)
                ),
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=11)
            )
        )
        
        # Add simple reference lines
        fig.add_hline(
            y=90, 
            line=dict(dash="dash", color="green", width=1),
            annotation_text="Excellent (90%)",
            annotation_position="top right",
            annotation_font=dict(size=10, color="green")
        )
        fig.add_hline(
            y=70, 
            line=dict(dash="dash", color="orange", width=1),
            annotation_text="Good (70%)",
            annotation_position="top right",
            annotation_font=dict(size=10, color="orange")
        )
        
        return fig
    
    def export_results_to_pdf(self, predictions, image_name="image"):
        """Export prediction results to PDF"""
        try:
            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Crop Classification Results', 0, 1, 'C')
            pdf.ln(10)
            
            # Image info
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f'Image: {image_name}', 0, 1)
            pdf.cell(0, 10, f'Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
            pdf.ln(5)
            
            # Predictions
            pdf.cell(0, 10, 'Predictions:', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            for i, pred in enumerate(predictions[:5], 1):
                pdf.cell(0, 8, f'{i}. {pred["class"]}: {pred["confidence"]:.4f} ({pred["confidence"]*100:.1f}%)', 0, 1)
            
            # Save to bytes
            pdf_output = pdf.output()
            return pdf_output
            
        except Exception as e:
            print(f"PDF export error: {str(e)}")
            return None
    
    def load_model(self, model_type: str, model_folder: str):
        """Load a specific model"""
        try:
            if model_type in self.models:
                return True
            
            # Different model file patterns for different model types
            if model_type == 'simclrv1':
                simclr_path = os.path.join(model_folder, 'simclrv1_crop_model_final.pth')
            elif model_type == 'simclrv2':
                simclr_path = os.path.join(model_folder, 'simclrv2_crop_model_final.pth')
            elif model_type == 'nnclr':
                simclr_path = os.path.join(model_folder, 'NNCLR_crop_model_final.pth')
            elif model_type == 'barlowtwins':
                simclr_path = os.path.join(model_folder, 'BarlowTwins_crop_model_final.pth')
            else:
                simclr_path = os.path.join(model_folder, 'simclr_crop_model_final.pth')
                
            if model_type == 'simclrv1':   
                classifier_path = os.path.join(model_folder, 'crop_simclrv1_classifier_final.pth')
            elif model_type == 'simclrv2':
                classifier_path = os.path.join(model_folder, 'crop_simclrv2_classifier_final.pth')
            elif model_type == 'nnclr':
                classifier_path = os.path.join(model_folder, 'crop_nnclr_classifier_final.pth')
            elif model_type == 'barlowtwins':
                classifier_path = os.path.join(model_folder, 'crop_barlowtwins_classifier_final.pth')
            else:
                classifier_path = os.path.join(model_folder, 'crop_simclr_classifier_final.pth')

            if not os.path.exists(simclr_path) or not os.path.exists(classifier_path):
                print(f"❌ Model files not found in {model_folder}")
                print(f"   Looking for: {os.path.basename(simclr_path)} and {os.path.basename(classifier_path)}")
                return False
            
            # Load SimCLR/NNCLR/BarlowTwins model
            if model_type == 'simclrv1':
                simclr_model = SimCLRv1().to(DEVICE)
            elif model_type == 'nnclr':
                simclr_model = NNCLR().to(DEVICE)
            elif model_type == 'barlowtwins':
                simclr_model = BarlowTwins().to(DEVICE)
            else:  # simclrv2
                simclr_model = ImprovedSimCLRv2().to(DEVICE)
            
            simclr_state = torch.load(simclr_path, map_location=DEVICE)
            simclr_model.load_state_dict(simclr_state)
            simclr_model.eval()
            
            # Load classifier model
            if model_type == 'barlowtwins':
                classifier_model = BarlowTwinsClassifier().to(DEVICE)
            else:
                classifier_model = EnhancedCropClassifier().to(DEVICE)
            classifier_state = torch.load(classifier_path, map_location=DEVICE)
            classifier_model.load_state_dict(classifier_state)
            classifier_model.eval()
            
            self.models[model_type] = {
                'simclr': simclr_model,
                'classifier': classifier_model
            }
            
            # Verify model output dimensions match class names
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
                dummy_output = classifier_model(dummy_input)
                print(f"   - Model output shape: {dummy_output.shape}")
                print(f"   - Expected classes: {len(self.class_names)}")
                
                if dummy_output.shape[1] != len(self.class_names):
                    print(f"⚠️ WARNING: Model outputs {dummy_output.shape[1]} classes but class_names.txt has {len(self.class_names)} classes!")
                    print("This will cause incorrect predictions. Please check:")
                    print("1. Is your class_names.txt file correct for this model?")
                    print("2. Was the model trained with the same class order?")
            
            print(f"✅ {self.model_info[model_type]['name']} loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading {model_type}: {str(e)}")
            return False
    
    def load_all_models(self):
        """Load all available models"""
        model_folders = [
            ('simclrv1', 'models\\simclrv1'), 
            ('simclrv2', 'models\\simclrv2'), 
            ('nnclr', 'models\\nnclr'), 
            ('barlowtwins', 'models\\barlowtwins')
        ]
        loaded_models = []
        
        for model_type, model_folder in model_folders:
            if os.path.exists(model_folder):
                success = self.load_model(model_type, model_folder)
                if success:
                    loaded_models.append(model_type)
        
        # Initialize GradCAM for all loaded models
        print("\n🔍 Initializing GradCAM for all models...")
        for model_type in loaded_models:
            self.initialize_gradcam(model_type)
        
        return loaded_models
    
    def get_original_training_classes(self):
        """
        Return the class names based on the ACTUAL cluster mapping from pseudo_labels.csv
        This is the cluster-to-class mapping determined by the enhanced K-Means clustering.
        """
        # Based on the cluster analysis from enhanced_pseudo_labels.csv
        # Each index represents a cluster ID, and the value is the dominant class in that cluster
        cluster_to_class_mapping = [
            "Mustard",      # Cluster 0: Mustard (46.71% purity)
            "Chilli",       # Cluster 1: Chilli (16.75% purity)
            "Sunflower",    # Cluster 2: Sunflower (100% purity)
            "Banana",       # Cluster 3: Banana (97.11% purity)
            "Pearl_Millet", # Cluster 4: Pearl_Millet (97.69% purity)
            "Chilli",       # Cluster 5: Chilli (21.14% purity)
            "Rice",         # Cluster 6: Rice (92.31% purity)
            "Jute",         # Cluster 7: Jute (96.97% purity)
            "Fox_Nut",      # Cluster 8: Fox_Nut (96.15% purity)
            "Cardamom",     # Cluster 9: Cardamom (45.38% purity)
            "Almond",       # Cluster 10: Almond (67.89% purity)
            "Soyabean",     # Cluster 11: Soyabean (24.52% purity)
            "Tomato",       # Cluster 12: Tomato (53.69% purity)
            "Pineapple",    # Cluster 13: Pineapple (100% purity)
            "Coffee",       # Cluster 14: Coffee (52.96% purity)
            "Gram",         # Cluster 15: Gram (27.59% purity)
            "Tobacco",      # Cluster 16: Tobacco (30.92% purity)
            "Mustard",      # Cluster 17: Mustard (93.48% purity)
            "Sugarcane",    # Cluster 18: Sugarcane (28.17% purity)
            "Wheat",        # Cluster 19: Wheat (95.14% purity)
            "Olive",        # Cluster 20: Olive (21.37% purity)
            "Papaya",       # Cluster 21: Papaya (89.51% purity)
            "Cucumber",     # Cluster 22: Cucumber (93.05% purity)
            "Coconut",      # Cluster 23: Coconut (94.74% purity)
            "Sugarcane",    # Cluster 24: Sugarcane (36.80% purity)
            "Maize",        # Cluster 25: Maize (92.16% purity)
            "Cotton",       # Cluster 26: Cotton (96.34% purity)
            "Tobacco",      # Cluster 27: Tobacco (38.33% purity)
            "Jowar",        # Cluster 28: Jowar (91.91% purity)
            "Tobacco"       # Cluster 29: Tobacco (24.85% purity)
        ]
        
        print("✅ Using ACTUAL cluster-to-class mapping from enhanced_pseudo_labels.csv")
        print("📋 Cluster mapping (based on dominant class in each cluster):")
        for i, cls in enumerate(cluster_to_class_mapping):
            print(f"   {i:2d}: {cls}")
        
        return cluster_to_class_mapping
    
    def create_class_mapping(self):
        """Create a direct mapping from model cluster indices to class names"""
        cluster_classes = self.get_original_training_classes()
        current_classes = self.class_names
        
        print(f"\n🔄 Creating class mapping:")
        print(f"   Model cluster classes: {len(cluster_classes)}")
        print(f"   Current class names: {len(current_classes)}")
        
        # Direct mapping: model output index -> cluster class name
        mapping = {}
        
        for cluster_idx, cluster_class in enumerate(cluster_classes):
            mapping[cluster_idx] = cluster_class
            print(f"✅ Model index {cluster_idx:2d}: -> {cluster_class}")
        
        print(f"\n📊 Mapping summary: {len(mapping)} clusters mapped directly")
        print(f"🎯 Key mappings:")
        print(f"   Index 25 (Maize) -> {mapping.get(25, 'NOT FOUND')}")
        print(f"   Index 2 (Sunflower) -> {mapping.get(2, 'NOT FOUND')}")
        
        return mapping
    
    def initialize_gradcam(self, model_type: str):
        """Initialize GradCAM for a specific model"""
        if model_type not in self.models:
            return None
            
        try:
            # Get the actual model from the stored dictionary
            model_dict = self.models[model_type]
            if isinstance(model_dict, dict) and 'classifier' in model_dict:
                model = model_dict['classifier']  # Use the classifier model
            else:
                model = model_dict  # Direct model object
            
            # Handle DataParallel models
            if isinstance(model, nn.DataParallel):
                model_for_gradcam = model.module
            else:
                model_for_gradcam = model
            
            # Print model structure for debugging
            print(f"🔍 Initializing GradCAM for {model_type}")
            print(f"📋 Model type: {type(model_for_gradcam).__name__}")
            
            # Try different target layers based on model architecture
            target_layer = None
            
            # Strategy 1: Try Vision Transformer backbone layers
            if hasattr(model_for_gradcam, 'backbone'):
                backbone = model_for_gradcam.backbone
                print(f"✅ Found backbone: {type(backbone).__name__}")
                
                # Try ViT-like structures
                if hasattr(backbone, 'blocks') and len(backbone.blocks) > 0:
                    last_block = backbone.blocks[-1]
                    print(f"✅ Found blocks, last block: {type(last_block).__name__}")
                    
                    # Try norm1 (best for ViT)
                    if hasattr(last_block, 'norm1'):
                        target_layer = last_block.norm1
                        print(f"✅ Using norm1 layer: {type(target_layer).__name__}")
                    # Try norm2 as fallback
                    elif hasattr(last_block, 'norm2'):
                        target_layer = last_block.norm2
                        print(f"✅ Using norm2 layer: {type(target_layer).__name__}")
                    # Try attention projection
                    elif hasattr(last_block, 'attn') and hasattr(last_block.attn, 'proj'):
                        target_layer = last_block.attn.proj
                        print(f"✅ Using attention projection: {type(target_layer).__name__}")
                    # Try MLP
                    elif hasattr(last_block, 'mlp'):
                        target_layer = last_block.mlp
                        print(f"✅ Using MLP layer: {type(target_layer).__name__}")
                    # Use entire last block as fallback
                    else:
                        target_layer = last_block
                        print(f"✅ Using entire last block: {type(target_layer).__name__}")
                
                # Try other backbone structures (CNN-like)
                elif hasattr(backbone, 'features'):
                    target_layer = backbone.features[-1]
                    print(f"✅ Using CNN features layer: {type(target_layer).__name__}")
                elif hasattr(backbone, 'layer4'):
                    target_layer = backbone.layer4[-1]
                    print(f"✅ Using ResNet layer4: {type(target_layer).__name__}")
                else:
                    # Use the backbone itself as last resort
                    target_layer = backbone
                    print(f"⚠️ Using entire backbone: {type(target_layer).__name__}")
            
            # Strategy 2: If no backbone, try direct model layers
            elif hasattr(model_for_gradcam, 'blocks') and len(model_for_gradcam.blocks) > 0:
                last_block = model_for_gradcam.blocks[-1]
                if hasattr(last_block, 'norm1'):
                    target_layer = last_block.norm1
                    print(f"✅ Using direct model norm1: {type(target_layer).__name__}")
                else:
                    target_layer = last_block
                    print(f"✅ Using direct model last block: {type(target_layer).__name__}")
            
            # Strategy 3: Try to find any reasonable layer
            else:
                # Get all named modules and find a suitable one
                named_modules = list(model_for_gradcam.named_modules())
                print(f"🔍 Model has {len(named_modules)} modules")
                
                # Look for common layer types
                for name, module in reversed(named_modules):
                    if any(layer_type in name.lower() for layer_type in ['norm', 'bn', 'layernorm']):
                        target_layer = module
                        print(f"✅ Using normalization layer '{name}': {type(target_layer).__name__}")
                        break
                
                # If still no layer found, use a convolutional or linear layer
                if target_layer is None:
                    for name, module in reversed(named_modules):
                        if isinstance(module, (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d)):
                            target_layer = module
                            print(f"✅ Using fallback layer '{name}': {type(target_layer).__name__}")
                            break
            
            # Final check
            if target_layer is not None:
                print(f"🎯 Selected target layer for {model_type}: {type(target_layer).__name__}")
                gradcam = GradCAM(model_for_gradcam, target_layer)
                self.gradcam_instances[model_type] = gradcam
                print(f"✅ GradCAM initialized successfully for {model_type}")
                return gradcam
            else:
                print(f"❌ Could not find suitable target layer for {model_type}")
                return None
                
        except Exception as e:
            print(f"❌ Error initializing GradCAM for {model_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def classify_image(self, image: Image.Image, model_type: str, top_k: int = 5) -> Dict:
        """Classify an image using specified model"""
        try:
            if model_type not in self.models:
                return {
                    "error": f"Model {model_type} not loaded. Please ensure model files exist.",
                    "predictions": []
                }
            
            # Handle different image input types
            if isinstance(image, tuple):
                # If it's a tuple, take the first element (usually the file path)
                image_path = image[0] if image[0] is not None else image[1]
                if isinstance(image_path, str):
                    try:
                        image = Image.open(image_path)
                    except Exception as e:
                        return {
                            "error": f"Could not load image: {str(e)}",
                            "predictions": []
                        }
                else:
                    return {
                        "error": "Invalid image input format",
                        "predictions": []
                    }
            elif not isinstance(image, Image.Image):
                return {
                    "error": "Image must be a PIL Image object",
                    "predictions": []
                }

            # Create class mapping to handle potential class order mismatch
            class_mapping = self.create_class_mapping()
            
            # Preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
            
            # Get model
            classifier = self.models[model_type]['classifier']
            
            # Make prediction
            with torch.no_grad():
                outputs = classifier(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.topk(probabilities, top_k, dim=1)
            
            # Debug information
            print(f"🔍 Debug Info for {model_type}:")
            print(f"   - Output shape: {outputs.shape}")
            print(f"   - Top predicted indices: {predicted_indices[0][:5].tolist()}")
            print(f"   - Top confidences: {confidences[0][:5].tolist()}")
            
            # Prepare results with corrected class mapping
            predictions = []
            for i in range(top_k):
                model_class_idx = predicted_indices[0][i].item()
                confidence = confidences[0][i].item()
                
                # Use the class mapping to get the correct class name
                if model_class_idx in class_mapping:
                    class_name = class_mapping[model_class_idx]
                    print(f"   - Prediction {i+1}: Model index {model_class_idx} -> Mapped to '{class_name}' ({confidence:.3f})")
                else:
                    print(f"⚠️ Warning: Model predicted class index {model_class_idx} not found in mapping")
                    class_name = f"Unknown_Index_{model_class_idx}"
                
                predictions.append({
                    "class": class_name,
                    "confidence": confidence,
                    "class_id": model_class_idx,
                    "mapped_class": class_name
                })
            
            # Save to prediction history
            image_name = f"classified_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.save_prediction_to_db(image_name, model_type, predictions)
            
            return {
                "predictions": predictions,
                "model_used": self.model_info[model_type]['name']
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": f"Classification error: {str(e)}",
                "predictions": []
            }
    
    def classify_with_gradcam(self, image: Image.Image, model_type: str, top_k: int = 5, target_class: Optional[int] = None) -> Dict:
        """Classify an image and generate GradCAM visualization"""
        try:
            if model_type not in self.models:
                return {"error": f"Model {model_type} not loaded"}
            
            # Get regular classification first
            classification_result = self.classify_image(image, model_type, top_k)
            if "error" in classification_result:
                return classification_result
            
            # Initialize GradCAM if not already done
            if model_type not in self.gradcam_instances:
                gradcam = self.initialize_gradcam(model_type)
                if gradcam is None:
                    return {
                        "error": f"Could not initialize GradCAM for {model_type}",
                        "predictions": classification_result["predictions"]
                    }
            
            gradcam = self.gradcam_instances[model_type]
            
            # Prepare image for GradCAM
            input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
            
            # Generate GradCAM for target class or top prediction
            if target_class is None:
                target_class = classification_result["predictions"][0]["class_id"]
            
            # Generate GradCAM heatmap
            cam = gradcam(input_tensor, class_idx=target_class)
            
            # Create visualization
            gradcam_visualization = self.create_gradcam_visualization(image, cam, classification_result, target_class)
            
            return {
                "predictions": classification_result["predictions"],
                "model_used": classification_result["model_used"],
                "gradcam_heatmap": cam,
                "gradcam_visualization": gradcam_visualization,
                "target_class": target_class,
                "target_class_name": classification_result["predictions"][0]["class"]
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": f"GradCAM classification error: {str(e)}",
                "predictions": []
            }
    
    def create_gradcam_visualization(self, original_image: Image.Image, cam: np.ndarray, 
                                   classification_result: Dict, target_class: int) -> Image.Image:
        """Create a combined GradCAM visualization"""
        try:
            # Convert PIL image to numpy
            img_np = np.array(original_image.resize((224, 224)))
            if len(img_np.shape) == 2:  # Grayscale
                img_np = np.stack([img_np] * 3, axis=-1)
            
            # Normalize image
            img_np = img_np.astype(np.float32) / 255.0
            
            # Create figure with subplots and better spacing
            fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
            
            # Original image
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold', pad=15)
            axes[0].axis('off')
            
            # GradCAM heatmap
            im1 = axes[1].imshow(cam, cmap='jet')
            axes[1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold', pad=15)
            axes[1].axis('off')
            cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, shrink=0.8)
            cbar.set_label('Activation', rotation=270, labelpad=12, fontsize=10)
            cbar.ax.tick_params(labelsize=9)
            
            # Overlay
            axes[2].imshow(img_np)
            overlay = axes[2].imshow(cam, cmap='jet', alpha=0.5)
            axes[2].set_title('Overlay', fontsize=12, fontweight='bold', pad=15)
            axes[2].axis('off')
            
            # Add prediction info with better positioning
            pred_text = f"Prediction: {classification_result['predictions'][0]['class']} (Confidence: {classification_result['predictions'][0]['confidence']:.1%})"
            fig.suptitle(pred_text, fontsize=14, fontweight='bold', y=0.95)
            
            # Adjust layout to prevent overlapping
            plt.subplots_adjust(top=0.85, bottom=0.05, left=0.02, right=0.98, wspace=0.15)
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.2)
            buf.seek(0)
            result_image = Image.open(buf)
            plt.close()
            
            return result_image
            
        except Exception as e:
            print(f"Error creating GradCAM visualization: {e}")
            # Return original image if visualization fails
            return original_image.resize((450, 150))  # Approximate size for 3 panels
    
    def compare_models(self, image: Image.Image, selected_models: List[str]) -> Dict:
        """Compare predictions across multiple models"""
        try:
            results = {}
            
            for model_type in selected_models:
                if model_type in self.models:
                    result = self.classify_image(image, model_type, top_k=3)
                    if "predictions" in result and result["predictions"]:
                        results[model_type] = {
                            "model_name": self.model_info[model_type]["name"],
                            "top_prediction": result["predictions"][0],
                            "all_predictions": result["predictions"]
                        }
            
            # Create comparison visualization
            comparison_plot = self.create_comparison_plot(results)
            
            return {
                "results": results,
                "comparison_plot": comparison_plot
            }
            
        except Exception as e:
            return {
                "error": f"Comparison error: {str(e)}",
                "results": {},
                "comparison_plot": None
            }
    
    def create_confidence_plot(self, predictions: List[Dict], top_k: int = 5):
        """Create an interactive Plotly confidence plot"""
        try:
            if not predictions:
                # Create empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text="No predictions available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    font=dict(size=16, color="gray")
                )
                fig.update_layout(
                    title="Confidence Visualization",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    template="plotly_white"
                )
                return fig
            
            # Prepare data
            top_predictions = predictions[:top_k]
            classes = [pred['class'] for pred in top_predictions]
            # Convert to percentage (multiply by 100)
            confidences = [pred['confidence'] * 100 for pred in top_predictions]
            
            # Create DataFrame for Plotly
            df = pd.DataFrame({
                'Crop': classes,
                'Confidence': confidences,
                'Rank': [f"#{i+1}" for i in range(len(classes))]
            })
            
            # Create interactive horizontal bar chart
            fig = px.bar(
                df,
                x='Confidence',
                y='Crop',
                orientation='h',
                text='Confidence',
                title='🌾 Crop Classification Confidence Scores',
                labels={'Confidence': 'Confidence Score (%)', 'Crop': 'Crop Type'},
                color='Confidence',
                color_continuous_scale='Viridis',
                height=max(400, len(classes) * 60)
            )
            
            # Update text on bars with bold formatting and rank
            fig.update_traces(
                texttemplate="<b>%{text:.1f}%</b>",
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>"
            )
            
            # Update layout for better appearance
            fig.update_layout(
                xaxis=dict(
                    range=[0, 100],
                    title=dict(text="<b>Confidence Score (%)</b>", font=dict(size=14)),
                    gridcolor='lightgray',
                    gridwidth=1
                ),
                yaxis=dict(
                    title=dict(text="<b>Crop Predictions</b>", font=dict(size=14)),
                    categoryorder='total ascending'
                ),
                title=dict(
                    text="<b>🌾 Crop Classification Confidence Scores</b>",
                    font=dict(size=16),
                    x=0.5
                ),
                template="plotly_white",
                showlegend=False,
                margin=dict(l=150, r=100, t=80, b=50),
                font=dict(size=12)
            )
            
            # Add confidence threshold line at 50% for reference
            fig.add_vline(
                x=50, 
                line_dash="dash", 
                line_color="orange",
                annotation_text="50% Reference Line",
                annotation_position="top right"
            )
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating Plotly confidence plot: {e}")
            # Simple fallback figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Plot Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                font=dict(size=14, color="red")
            )
            fig.update_layout(template="plotly_white")
            return fig
    
    def create_confidence_plot_with_threshold(self, predictions: List[Dict], top_k: int = 5, threshold: float = 0.0):
        """Create a clean, professional Plotly confidence plot with threshold visualization"""
        try:
            if not predictions:
                # Create empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text="No predictions available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    font=dict(size=16, color="gray")
                )
                fig.update_layout(template="plotly_white")
                return fig
            
            # Prepare data
            top_predictions = predictions[:top_k]
            classes = [pred['class'] for pred in top_predictions]
            # Convert to percentage (multiply by 100)
            confidences = [pred['confidence'] * 100 for pred in top_predictions]
            threshold_percent = threshold * 100
            
            # Simple color scheme - above/below threshold
            colors = []
            for conf in confidences:
                if conf >= threshold_percent:
                    colors.append('#2E8B57')  # Sea Green - Above threshold
                else:
                    colors.append('#D3D3D3')  # Light Gray - Below threshold
            
            # Create the main bar chart
            fig = go.Figure()
            
            # Add main bars - clean and simple
            fig.add_trace(go.Bar(
                y=classes,
                x=confidences,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=1)
                ),
                text=[f'{conf:.1f}%' for conf in confidences],
                textposition='outside',
                textfont=dict(size=12, color='black'),
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>',
                name='Confidence',
                showlegend=False
            ))
            
            # Add threshold line - simple and clean
            if threshold > 0:
                fig.add_vline(
                    x=threshold_percent,
                    line=dict(
                        color='red',
                        width=2,
                        dash='dash'
                    ),
                    annotation=dict(
                        text=f'Threshold: {threshold_percent:.1f}%',
                        textangle=90,
                        font=dict(size=12, color='red'),
                        bgcolor='white',
                        bordercolor='red',
                        borderwidth=1
                    )
                )
            
            # Clean, professional layout
            fig.update_layout(
                title=dict(
                    text=f'Crop Classification Confidence Analysis<br><span style="font-size:12px;">Threshold: {threshold_percent:.1f}% | Top {len(classes)} Predictions</span>',
                    font=dict(size=16, color='#333'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(
                        text='Confidence Score (%)',
                        font=dict(size=14, color='#333')
                    ),
                    range=[0, 100],
                    gridcolor='lightgray',
                    gridwidth=1,
                    tickfont=dict(size=11, color='#333'),
                    showline=True,
                    linecolor='gray',
                    linewidth=1
                ),
                yaxis=dict(
                    title=dict(
                        text='Crop Predictions',
                        font=dict(size=14, color='#333')
                    ),
                    tickfont=dict(size=11, color='#333'),
                    categoryorder='total ascending',
                    showline=True,
                    linecolor='gray',
                    linewidth=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=150, r=100, t=80, b=60),
                height=max(400, len(classes) * 60),
                font=dict(family='Arial'),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating Plotly threshold plot: {e}")
            # Fallback figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Plot Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                font=dict(size=14, color="red")
            )
            fig.update_layout(template="plotly_white")
            return fig
    
    def create_comparison_plot(self, results: Dict):
        """Create clean, professional Plotly model comparison visualization"""
        try:
            if not results:
                fig = go.Figure()
                fig.add_annotation(
                    text="No comparison data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    font=dict(size=16, color="gray")
                )
                fig.update_layout(template="plotly_white")
                return fig
            
            # Prepare data for comparison
            model_names = []
            top_classes = []
            top_confidences = []
            
            for model_type, result in results.items():
                model_names.append(result["model_name"])
                top_classes.append(result["top_prediction"]["class"])
                top_confidences.append(result["top_prediction"]["confidence"] * 100)  # Convert to percentage
            
            # Simple, clean colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'][:len(model_names)]
            
            # Create the main bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=model_names,
                y=top_confidences,
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=1)
                ),
                text=[f'{conf:.1f}%<br>{crop}' for conf, crop in zip(top_confidences, top_classes)],
                textposition='outside',
                textfont=dict(size=11, color='black'),
                hovertemplate='<b>%{x}</b><br>Top Prediction: <b>%{customdata}</b><br>Confidence: <b>%{y:.1f}%</b><extra></extra>',
                customdata=top_classes,
                name='Top Prediction Confidence'
            ))
            
            # Clean, professional layout
            fig.update_layout(
                title=dict(
                    text='Model Performance Comparison',
                    font=dict(size=16, color='#333'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(
                        text='Model Architecture',
                        font=dict(size=14, color='#333')
                    ),
                    tickangle=45,
                    tickfont=dict(size=11, color='#333'),
                    showline=True,
                    linecolor='gray',
                    linewidth=1
                ),
                yaxis=dict(
                    range=[0, 105],
                    title=dict(
                        text='Confidence Score (%)',
                        font=dict(size=14, color='#333')
                    ),
                    gridcolor='lightgray',
                    gridwidth=1,
                    tickfont=dict(size=11, color='#333'),
                    showline=True,
                    linecolor='gray',
                    linewidth=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=60, r=60, t=80, b=100),
                height=500,
                font=dict(family='Arial'),
                showlegend=False
            )
            
            # Add simple reference lines
            fig.add_hline(
                y=90, 
                line=dict(dash="dash", color="green", width=1),
                annotation_text="Excellent (90%)",
                annotation_position="top right",
                annotation_font=dict(size=10, color="green")
            )
            fig.add_hline(
                y=70, 
                line=dict(dash="dash", color="orange", width=1),
                annotation_text="Good (70%)",
                annotation_position="top right",
                annotation_font=dict(size=10, color="orange")
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating Plotly comparison plot: {str(e)}")
            # Fallback figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Comparison Error: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                font=dict(size=14, color="red")
            )
            fig.update_layout(template="plotly_white")
            return fig

# Initialize the app
app = CropClassificationApp()

def classify_single_image(image, model_type, top_k, confidence_threshold=0.0):
    """Gradio interface for single image classification with confidence threshold"""
    if image is None:
        # Return empty plot when no image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Please upload an image to see confidence visualization', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=14, color='white')
        ax.set_facecolor('#1e2025')
        fig.patch.set_facecolor('#0e1117')
        plt.tight_layout()
        return "Please upload an image.", fig
    
    try:
        result = app.classify_image(image, model_type, top_k)
        
        if "error" in result:
            # Return error plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Error: {result["error"]}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_facecolor('#1e2025')
            fig.patch.set_facecolor('#0e1117')
            plt.tight_layout()
            return result["error"], fig
        
        # Filter predictions based on confidence threshold
        filtered_predictions = [
            pred for pred in result["predictions"] 
            if pred['confidence'] >= confidence_threshold
        ]
        
        # If no predictions meet the threshold, show a message
        if not filtered_predictions and confidence_threshold > 0:
            threshold_msg = f"**Model:** {result['model_used']}\n\n"
            threshold_msg += f"⚠️ **No predictions above confidence threshold {confidence_threshold:.2f} ({confidence_threshold*100:.0f}%)**\n\n"
            threshold_msg += "**All Predictions (Below Threshold):**\n\n"
            for i, pred in enumerate(result["predictions"], 1):
                confidence_emoji = "🔴"  # Red since below threshold
                threshold_msg += f"{i}. {confidence_emoji} **{pred['class']}** - {pred['confidence']:.3f} ({pred['confidence']*100:.1f}%)\n"
            
            # Create plot with all predictions but mark threshold
            confidence_plot = app.create_confidence_plot_with_threshold(result["predictions"], top_k, confidence_threshold)
            return threshold_msg, confidence_plot
        
        # Use filtered predictions or all if threshold is 0
        display_predictions = filtered_predictions if confidence_threshold > 0 else result["predictions"]
        
        # Format results for display
        output_text = f"**Model:** {result['model_used']}\n\n"
        if confidence_threshold > 0:
            output_text += f"**Confidence Threshold:** {confidence_threshold:.2f} ({confidence_threshold*100:.0f}%)\n\n"
        output_text += "**Predictions:**\n\n"
        
        for i, pred in enumerate(display_predictions, 1):
            confidence_emoji = "🟢" if pred['confidence'] >= 0.90 else "🟡" if pred['confidence'] >= 0.70 else "🟠" if pred['confidence'] >= 0.50 else "🔴"
            output_text += f"{i}. {confidence_emoji} **{pred['class']}** - {pred['confidence']:.3f} ({pred['confidence']*100:.1f}%)\n"
        
        # Show count if filtered
        if confidence_threshold > 0 and len(filtered_predictions) < len(result["predictions"]):
            excluded_count = len(result["predictions"]) - len(filtered_predictions)
            output_text += f"\n*({excluded_count} prediction(s) excluded due to low confidence)*"
        
        # Create confidence plot - ensure it always returns a plot
        if confidence_threshold > 0:
            confidence_plot = app.create_confidence_plot_with_threshold(result["predictions"], top_k, confidence_threshold)
        else:
            confidence_plot = app.create_confidence_plot(result["predictions"], top_k)
        
        # Fallback if plot creation fails
        if confidence_plot is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Plot generation failed', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, color='orange')
            ax.set_facecolor('#1e2025')
            fig.patch.set_facecolor('#0e1117')
            plt.tight_layout()
            confidence_plot = fig
        
        return output_text, confidence_plot
        
    except Exception as e:
        # Return exception plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Exception: {str(e)}', 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=12, color='red')
        ax.set_facecolor('#1e2025')
        fig.patch.set_facecolor('#0e1117')
        plt.tight_layout()
        return f"Error: {str(e)}", fig

def test_batch_function(files, model_type, confidence_threshold=0.0):
    """Simple test function to debug batch processing with confidence threshold"""
    return f"Received {len(files) if files else 0} files and model: {model_type}", None

def classify_with_gradcam_interface(image, model_type, top_k, confidence_threshold=0.0, enable_gradcam=False):
    """Gradio interface for classification with optional GradCAM visualization"""
    if image is None:
        return "Please upload an image.", None, None
    
    try:
        if enable_gradcam:
            # Use GradCAM classification
            result = app.classify_with_gradcam(image, model_type, top_k)
        else:
            # Use regular classification
            result = app.classify_image(image, model_type, top_k)
        
        if "error" in result:
            return result["error"], None, None
        
        # Filter predictions based on confidence threshold
        filtered_predictions = [
            pred for pred in result["predictions"] 
            if pred['confidence'] >= confidence_threshold
        ]
        
        # Use filtered predictions or all if threshold is 0
        display_predictions = filtered_predictions if confidence_threshold > 0 else result["predictions"]
        
        # Format results for display
        output_text = f"**Model:** {result['model_used']}\n\n"
        if confidence_threshold > 0:
            output_text += f"**Confidence Threshold:** {confidence_threshold:.2f} ({confidence_threshold*100:.0f}%)\n\n"
        
        if enable_gradcam and "gradcam_visualization" in result:
            output_text += f"**GradCAM Target:** {result.get('target_class_name', 'Top Prediction')}\n\n"
        
        output_text += "**Predictions:**\n\n"
        
        for i, pred in enumerate(display_predictions, 1):
            confidence_emoji = "🟢" if pred['confidence'] >= 0.90 else "🟡" if pred['confidence'] >= 0.70 else "🟠" if pred['confidence'] >= 0.50 else "🔴"
            output_text += f"{i}. {confidence_emoji} **{pred['class']}** - {pred['confidence']:.3f} ({pred['confidence']*100:.1f}%)\n"
        
        # Create confidence plot
        if confidence_threshold > 0:
            confidence_plot = app.create_confidence_plot_with_threshold(result["predictions"], top_k, confidence_threshold)
        else:
            confidence_plot = app.create_confidence_plot(result["predictions"], top_k)
        
        # Return GradCAM visualization if enabled
        gradcam_image = None
        if enable_gradcam and "gradcam_visualization" in result:
            gradcam_image = result["gradcam_visualization"]
        
        return output_text, confidence_plot, gradcam_image
        
    except Exception as e:
        return f"Error: {str(e)}", None, None

def compare_models_interface(image, selected_models):
    """Gradio interface for model comparison"""
    if image is None:
        return "Please upload an image.", None
    
    if not selected_models:
        return "Please select at least one model.", None
    
    try:
        result = app.compare_models(image, selected_models)
        
        if "error" in result:
            return result["error"], None
        
        # Format comparison results
        output_text = "**Model Comparison Results:**\n\n"
        
        for model_type, model_result in result["results"].items():
            model_name = model_result["model_name"]
            top_pred = model_result["top_prediction"]
            
            output_text += f"**{model_name}:**\n"
            output_text += f"• Top Prediction: **{top_pred['class']}** ({top_pred['confidence']:.3f})\n"
            output_text += f"• All Predictions:\n"
            
            for i, pred in enumerate(model_result["all_predictions"][:3], 1):
                output_text += f"  {i}. {pred['class']} - {pred['confidence']:.3f}\n"
            output_text += "\n"
        
        return output_text, result.get("comparison_plot")
        
    except Exception as e:
        return f"Comparison error: {str(e)}", None

def process_batch_classification(files, model_types, confidence_threshold=0.0):
    """Gradio interface for batch processing with multiple models and confidence threshold"""
    print(f"🔍 Batch processing called with files: {files}, model_types: {model_types}, threshold: {confidence_threshold}")
    
    if not files:
        return "❌ No files uploaded. Please upload images first.", None
    
    if not model_types:
        return "❌ No models selected. Please select at least one model.", None
    
    try:
        print(f"📁 Processing {len(files)} files with confidence threshold {confidence_threshold}...")
        
        # Convert file objects to images
        images = []
        for i, file in enumerate(files):
            try:
                print(f"🔄 Processing file {i+1}: {file}")
                
                # Handle different file input types
                if hasattr(file, 'name'):
                    # File object with name attribute
                    print(f"   📄 File has name: {file.name}")
                    image = Image.open(file.name)
                    images.append((image, file.name))
                elif hasattr(file, 'orig_name'):
                    # Gradio file object
                    print(f"   📄 Gradio file: {file.orig_name}")
                    image = Image.open(file.name)
                    images.append((image, file.orig_name))
                elif isinstance(file, str):
                    # File path as string
                    print(f"   📄 File path: {file}")
                    image = Image.open(file)
                    images.append((image, file))
                else:
                    # Try to open directly
                    print(f"   📄 Direct file object: {type(file)}")
                    if hasattr(file, 'name'):
                        image = Image.open(file.name)
                        images.append((image, getattr(file, 'orig_name', file.name)))
                    else:
                        image = Image.open(file)
                        images.append((image, f"image_{len(images)+1}"))
                
                print(f"   ✅ Successfully loaded image {i+1}")
                
            except Exception as e:
                print(f"   ❌ Error opening file {i+1}: {e}")
                continue
        
        if not images:
            return "❌ No valid images could be processed. Please check your image files.", None
        
        print(f"🖼️ Successfully loaded {len(images)} images")
        
        # Check if all models are loaded
        for model_type in model_types:
            if model_type not in app.models:
                return f"❌ Model '{model_type}' is not loaded. Available models: {list(app.models.keys())}", None
        
        # Process batch with multiple models
        print(f"🤖 Processing with models: {model_types}")
        df, csv_filepath = app.process_batch_images(images, model_types, confidence_threshold)
        
        if df is not None and len(df) > 0:
            print(f"✅ Batch processing successful: {len(df)} results")
            
            # Create summary
            model_names = [app.model_info[mt]['name'] for mt in model_types]
            threshold_info = f"**🎚️ Confidence Threshold:** {confidence_threshold:.2f} ({confidence_threshold*100:.0f}%)\n" if confidence_threshold > 0 else ""
            
            summary = f"""
## 📊 Batch Processing Complete! ✅

**🎯 Successfully processed:** {len(df)} images
**🤖 Models used:** {', '.join(model_names)}
{threshold_info}**📄 CSV file saved to:** `{csv_filepath}`

### 📋 Processing Summary:

**Images processed:** {len(df)}
**Models:** {len(model_types)}
**Total predictions:** {len(df) * len(model_types)}

### 📋 Results Preview:
"""
            
            # Show first few rows
            for i, (_, row) in enumerate(df.head(5).iterrows()):
                summary += f"\n**{row['Image']}:**"
                for model_type in model_types:
                    model_name = app.model_info[model_type]['name']
                    if f"{model_name}_Top_Prediction" in row:
                        pred = row[f"{model_name}_Top_Prediction"]
                        conf = row[f"{model_name}_Confidence"]
                        summary += f"\n- {model_name}: {pred} ({conf})"
                summary += "\n"
            
            if len(df) > 5:
                summary += f"\n... and {len(df) - 5} more images"
            
            return summary, csv_filepath
        else:
            return "❌ No valid results generated. Please check your images and model selection.", None
            
    except Exception as e:
        import traceback
        error_msg = f"""
## ❌ Batch Processing Error

**Error:** {str(e)}

**Debug Info:**
- Files received: {len(files) if files else 0}
- Model types: {model_types}
- Available models: {list(app.models.keys()) if hasattr(app, 'models') else 'Not loaded'}

**Traceback:**
```
{traceback.format_exc()}
```
"""
        print(error_msg)
        return error_msg, None

def show_prediction_history():
    """Show prediction history dashboard"""
    try:
        history_df = app.get_prediction_history(100)
        
        if history_df.empty:
            return "No prediction history available.", None
        
        # Format confidence values properly
        history_df['Confidence'] = history_df['Confidence'].apply(lambda x: f"{float(x)*100:.1f}%" if x < 1 else f"{float(x):.1f}%")
        
        # Create summary statistics
        # Convert confidence back to float for calculations
        numeric_confidence = history_df['Confidence'].str.rstrip('%').astype(float) / 100
        
        summary = f"""
        ## 📊 Prediction History Dashboard
        
        **📈 Total Predictions:** {len(history_df)}
        **🎯 Average Confidence:** {numeric_confidence.mean()*100:.1f}%
        **🤖 Most Used Model:** {history_df['Model'].mode().iloc[0] if not history_df['Model'].mode().empty else 'N/A'}
        **🌱 Most Predicted Crop:** {history_df['Prediction'].mode().iloc[0] if not history_df['Prediction'].mode().empty else 'N/A'}
        
        ### 📋 Recent Predictions:
        
        """
        
        # Add clean formatted rows without pipes
        for i, (_, row) in enumerate(history_df.head(15).iterrows(), 1):
            # Truncate long names for better display
            image_name = row['Image'][:30] + "..." if len(str(row['Image'])) > 30 else row['Image']
            model_name = row['Model']
            prediction = row['Prediction']
            timestamp = str(row['Timestamp'])[:16]  # Show only date and time, no seconds
            
            summary += f"**{i}.** 🕐 {timestamp} • 🤖 {model_name} • 🌾 {prediction} • 📊 {row['Confidence']}\n"
            summary += f"    📁 {image_name}\n\n"
        
        if len(history_df) > 15:
            summary += f"\n\n... and {len(history_df) - 15} more predictions"
        
        # Return the DataFrame with proper column names for the Gradio table
        display_df = history_df.head(50).copy()
        display_df.columns = ['🕐 Timestamp', '🖼️ Image Name', '🤖 Model', '🌾 Crop Prediction', '📊 Confidence']
        
        return summary, display_df
        
    except Exception as e:
        error_msg = f"""
        ## ❌ Error Loading History
        
        **Error:** {str(e)}
        
        **Possible Solutions:**
        1. Check if the database file exists
        2. Ensure predictions have been made
        3. Verify database permissions
        
        **Technical Details:**
        ```
        {type(e).__name__}: {str(e)}
        ```
        """
        return error_msg, pd.DataFrame(columns=['🕐 Timestamp', '🖼️ Image Name', '🤖 Model', '🌾 Crop Prediction', '📊 Confidence'])

def show_performance_metrics():
    """Show model performance metrics"""
    try:
        metrics_plot = app.create_performance_metrics_chart()
        model_stats = app.calculate_model_performance_metrics()
        
        if not model_stats:
            return "No performance data available.", None
        
        # Create summary text
        summary = "## Model Performance Metrics\n\n"
        for model, stats in model_stats.items():
            summary += f"**{model}:**\n"
            summary += f"- Total Predictions: {stats['total_predictions']}\n"
            summary += f"- Average Confidence: {stats['avg_confidence']:.3f}\n"
            summary += f"- Top Crop: {max(stats['crop_distribution'].items(), key=lambda x: x[1])[0] if stats['crop_distribution'] else 'N/A'}\n\n"
        
        return summary, metrics_plot
        
    except Exception as e:
        return f"Error generating metrics: {str(e)}", None

def export_prediction_results(image, model_type, top_k, confidence_threshold=0.0):
    """Export results to PDF with confidence threshold filtering"""
    if image is None:
        return None, "Please upload an image first."
    
    try:
        # Get predictions
        result = app.classify_image(image, model_type, top_k)
        
        if "error" in result:
            return None, result["error"]
        
        # Filter predictions based on confidence threshold
        filtered_predictions = [
            pred for pred in result["predictions"] 
            if pred['confidence'] >= confidence_threshold
        ]
        
        # Use filtered predictions or all if threshold is 0
        export_predictions = filtered_predictions if confidence_threshold > 0 else result["predictions"]
        
        # Generate PDF with threshold info
        image_name = f"classified_image_threshold_{int(confidence_threshold*100)}pct" if confidence_threshold > 0 else "classified_image"
        pdf_bytes = app.export_results_to_pdf(export_predictions, image_name)
        
        if pdf_bytes:
            # Save to temporary file
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            threshold_suffix = f"_threshold_{int(confidence_threshold*100)}pct" if confidence_threshold > 0 else ""
            pdf_path = f"classification_results_{timestamp}{threshold_suffix}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(pdf_bytes)
            
            return pdf_path, "PDF generated successfully!"
        else:
            return None, "Failed to generate PDF."
            
    except Exception as e:
        return None, f"Export error: {str(e)}"

def compare_multiple_models(image, *selected_models):
    """Gradio interface for model comparison"""
    if image is None:
        return "Please upload an image.", None
    
    # Filter selected models
    models_to_compare = [model for model in selected_models if model]
    
    if len(models_to_compare) < 2:
        return "Please select at least 2 models for comparison.", None
    
    result = app.compare_models(image, models_to_compare)
    
    if "error" in result:
        return result["error"], None
    
    # Format results for display
    output_text = "**Model Comparison Results:**\n\n"
    for model_type, model_result in result["results"].items():
        model_name = model_result["model_name"]
        top_pred = model_result["top_prediction"]
        output_text += f"**{model_name}:**\n"
        output_text += f"- Top Prediction: **{top_pred['class']}** ({top_pred['confidence']:.3f})\n"
        output_text += f"- Confidence: {top_pred['confidence']*100:.1f}%\n\n"
    
    return output_text, result["comparison_plot"]

def get_available_models():
    """Get list of available models"""
    loaded_models = app.load_all_models()
    available_models = [(app.model_info[model]['name'], model) for model in loaded_models]
    return available_models if available_models else [("No models found", "none")]

# Load models on startup
available_models = get_available_models()
model_choices = [choice[1] for choice in available_models if choice[1] != "none"]

# Create Gradio interface
with gr.Blocks(
    title="🌾 Advanced Agricultural Crop Classification",
    theme="Nymbo/Nymbo_Theme",
    css="""
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #00d4ff, #0099cc, #0066ff);
        color: white !important;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 102, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .model-info {
        background: rgba(0, 102, 255, 0.1);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 102, 255, 0.2);
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .model-info h3 {
        color: #00d4ff !important;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .model-info ul {
        color: #e0e0e0 !important;
    }
    
    .model-info li {
        color: #e0e0e0 !important;
        margin-bottom: 0.3rem;
    }
    
    .model-info strong {
        color: #00d4ff !important;
        text-shadow: 0 0 5px rgba(0, 212, 255, 0.3);
    }
    
    /* Enhanced button styling */
    .gr-button {
        background: linear-gradient(135deg, #0066ff, #0099cc) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .gr-button:hover {
        background: linear-gradient(135deg, #0080ff, #00b3e6) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(0, 102, 255, 0.4) !important;
    }
    
    /* Enhanced tab styling */
    .gradio-tab {
        background: rgba(0, 102, 255, 0.1) !important;
        border: 1px solid rgba(0, 102, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    
    /* Input field enhancements */
    .gr-textbox, .gr-dropdown {
        border: 1px solid rgba(0, 102, 255, 0.3) !important;
        border-radius: 8px !important;
        background: rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Progress bar styling */
    .gr-progress {
        background: linear-gradient(90deg, #00d4ff, #0066ff) !important;
        border-radius: 4px !important;
    }
    """
) as demo:
    
    gr.HTML("""
        <div class="main-header">
            <h1>🌾 Advanced Agricultural Crop Classification System</h1>
            <p>State-of-the-art self-supervised learning models for crop identification</p>
            <p><strong>✨ Powered by Vision Transformers • SimCLR v2 • NNCLR • Barlow Twins • GradCAM • Analytics ✨</strong></p>
        </div>
    """)
    
    with gr.Tab("🔍 Enhanced Single Classification"):
        gr.HTML("""
            <div class="model-info">
                <h3>🔍 Enhanced Classification Features</h3>
                <ul>
                    <li><strong>Real-time Confidence Analysis:</strong> Distribution charts and statistics</li>
                    <li><strong>GradCAM Visualization:</strong> See what the model focuses on</li>
                    <li><strong>Prediction History:</strong> Automatic logging to database</li>
                    <li><strong>Export Options:</strong> PDF and CSV report generation</li>
                </ul>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📸 Upload Crop Image",
                    type="pil",
                    height=400
                )
                
                model_selector = gr.Dropdown(
                    choices=available_models,
                    label="🤖 Select Model",
                    value="simclrv1" if "simclrv1" in model_choices else (model_choices[0] if model_choices else None)
                )
                
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="🎯 Top K Predictions"
                )
                
                confidence_threshold_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.01,
                    label="🎚️ Confidence Threshold (0 = Show All)",
                    info="Only show predictions above this confidence level"
                )
                
                gradcam_checkbox = gr.Checkbox(
                    label="🔍 Enable GradCAM Visualization",
                    value=False,
                    info="Generate attention heatmaps showing model focus areas"
                )
                
                with gr.Row():
                    classify_btn = gr.Button(
                        "🚀 Classify Image",
                        variant="primary",
                        size="lg"
                    )
                    export_pdf_btn = gr.Button(
                        "📄 Export PDF",
                        variant="secondary"
                    )
            
            with gr.Column(scale=2):
                result_text = gr.Textbox(
                    label="📊 Classification Results",
                    lines=10,
                    max_lines=15
                )
                
                confidence_plot = gr.Plot(
                    label="📈 Confidence Analysis"
                )
        
        # GradCAM visualization (conditionally visible)
        with gr.Row(visible=False) as gradcam_row:
            gradcam_output = gr.Image(
                label="🔍 GradCAM Visualization",
                interactive=False
            )
        
        # Toggle GradCAM row visibility
        gradcam_checkbox.change(
            fn=lambda x: gr.Row.update(visible=x),
            inputs=[gradcam_checkbox],
            outputs=[gradcam_row]
        )
        
        # Updated classify function call
        classify_btn.click(
            fn=classify_with_gradcam_interface,
            inputs=[
                input_image, 
                model_selector, 
                top_k_slider, 
                confidence_threshold_slider,
                gradcam_checkbox
            ],
            outputs=[result_text, confidence_plot, gradcam_output]
        )
    
    with gr.Tab("🔄 Enhanced Model Comparison"):
        gr.HTML("""
            <div class="model-info">
                <h3>� Model Comparison Features</h3>
                <ul>
                    <li><strong>Multi-Model Analysis:</strong> Compare up to 4 SSL models simultaneously</li>
                    <li><strong>Interactive Visualizations:</strong> Professional Plotly charts</li>
                    <li><strong>Performance Metrics:</strong> Confidence and accuracy comparisons</li>
                    <li><strong>Model Rankings:</strong> Side-by-side performance evaluation</li>
                </ul>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                comp_input_image = gr.Image(
                    label="� Upload Image for Comparison",
                    type="pil"
                )
                
                model_checkboxes = gr.CheckboxGroup(
                    choices=model_choices,
                    label="🤖 Select Models to Compare",
                    value=model_choices[:2] if len(model_choices) >= 2 else model_choices
                )
                
                compare_btn = gr.Button(
                    "🔄 Compare Models",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                comparison_results = gr.Textbox(
                    label="📊 Comparison Results",
                    lines=10,
                    max_lines=15
                )
                
                comparison_plot = gr.Plot(
                    label="📈 Model Performance Comparison"
                )
        
        compare_btn.click(
            fn=compare_models_interface,
            inputs=[comp_input_image, model_checkboxes],
            outputs=[comparison_results, comparison_plot]
        )
    
    with gr.Tab("📦 Batch Processing"):
        gr.HTML("""
            <div class="model-info">
                <h3>⚡ Batch Classification Features</h3>
                <ul>
                    <li><strong>Multi-Image Upload:</strong> Process multiple images simultaneously</li>
                    <li><strong>CSV Export:</strong> Download results in spreadsheet format</li>
                    <li><strong>Progress Tracking:</strong> Real-time processing status</li>
                    <li><strong>Bulk Analysis:</strong> Comprehensive batch statistics</li>
                </ul>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                batch_files = gr.File(
                    label="📁 Upload Multiple Images",
                    file_count="multiple",
                    file_types=["image"]
                )
                
                batch_model_selector = gr.CheckboxGroup(
                    choices=available_models,
                    label="🤖 Select Models for Batch Processing (Multiple Selection)",
                    value=[model_choices[0]] if model_choices else []
                )
                
                batch_confidence_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.01,
                    label="🎚️ Batch Confidence Threshold (0 = Show All)",
                    info="Filter predictions below this confidence for all models"
                )
                
                batch_process_btn = gr.Button(
                    "⚡ Process Batch",
                    variant="primary",
                    size="lg"
                )
                
                test_batch_btn = gr.Button(
                    "🧪 Test Batch Input",
                    variant="secondary"
                )
            
            with gr.Column(scale=2):
                batch_output_text = gr.Textbox(
                    label="📊 Batch Processing Results",
                    lines=10
                )
                
                batch_csv_output = gr.File(label="📄 Download CSV Results")
        
        batch_process_btn.click(
            fn=process_batch_classification,
            inputs=[batch_files, batch_model_selector, batch_confidence_threshold],
            outputs=[batch_output_text, batch_csv_output]
        )
        
        test_batch_btn.click(
            fn=test_batch_function,
            inputs=[batch_files, batch_model_selector, batch_confidence_threshold],
            outputs=[batch_output_text, batch_csv_output]
        )
    
    with gr.Tab("📊 Analytics Dashboard"):
        gr.HTML("""
            <div class="model-info">
                <h3>📈 Analytics & Performance Metrics</h3>
                <ul>
                    <li><strong>Prediction History:</strong> Track all classifications with timestamps</li>
                    <li><strong>Model Performance:</strong> Compare accuracy and usage statistics</li>
                    <li><strong>Confidence Analysis:</strong> Distribution and trend analysis</li>
                    <li><strong>Crop Statistics:</strong> Most/least predicted crops</li>
                </ul>
            </div>
        """)
        
        with gr.Tab("📜 Prediction History"):
            with gr.Row():
                history_refresh_btn = gr.Button("🔄 Refresh History", variant="primary")
            
            history_text = gr.Markdown(label="📋 History Summary")
            history_table = gr.Dataframe(label="🗂️ Recent Predictions")
            
            history_refresh_btn.click(
                fn=show_prediction_history,
                outputs=[history_text, history_table]
            )
        
        with gr.Tab("⚡ Performance Metrics"):
            with gr.Row():
                metrics_refresh_btn = gr.Button("📊 Generate Metrics", variant="primary")
            
            metrics_text = gr.Markdown(label="📈 Performance Summary")
            metrics_plot = gr.Plot(label="📊 Performance Visualization")
            
            metrics_refresh_btn.click(
                fn=show_performance_metrics,
                outputs=[metrics_text, metrics_plot]
            )
    
    with gr.Tab("⚖️ Model Comparison"):
        gr.HTML("""
            <div class="model-info">
                <h3>🔬 Enhanced Model Comparison</h3>
                <ul>
                    <li><strong>Side-by-side Analysis:</strong> Compare predictions across models</li>
                    <li><strong>Confidence Comparison:</strong> Visual confidence distribution</li>
                    <li><strong>Performance Heatmaps:</strong> Model vs. crop performance matrix</li>
                    <li><strong>Statistical Analysis:</strong> Detailed comparison metrics</li>
                </ul>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                comparison_image = gr.Image(
                    label="📸 Upload Crop Image",
                    type="pil",
                    height=400
                )
                
                gr.HTML("<h4>🎛️ Select Models to Compare:</h4>")
                
                model_checkboxes = []
                for model_name, model_id in available_models:
                    if model_id != "none":
                        checkbox = gr.Checkbox(
                            label=f"📐 {model_name}",
                            value=False
                        )
                        model_checkboxes.append(checkbox)
                
                compare_btn = gr.Button(
                    "⚖️ Compare Models",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                comparison_text = gr.Markdown(label="📊 Comparison Results")
                comparison_plot = gr.Plot(label="📈 Model Performance Visualization")
        
        # Create dynamic input list for comparison
        compare_inputs = [comparison_image] + model_checkboxes
        
        compare_btn.click(
            fn=lambda image, *checkboxes: compare_multiple_models(
                image, 
                *[model_choices[i] for i, checked in enumerate(checkboxes) if checked and i < len(model_choices)]
            ),
            inputs=compare_inputs,
            outputs=[comparison_text, comparison_plot]
        )
    
    with gr.Tab("📚 Model Information"):
        gr.HTML(f"""
            <div style="padding: 2rem;">
                <h2>🔬 Technical Specifications</h2>
                
                <h3>📊 Dataset Information</h3>
                <ul>
                    <li><strong>Classes:</strong> {len(app.class_names)} agricultural crop types</li>
                    <li><strong>Resolution:</strong> 224×224 pixels</li>
                    <li><strong>Preprocessing:</strong> Vision Transformer (ViT) normalization standards</li>
                </ul>
                
                <h3>🏗️ Model Architectures</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; color: #333333 !important;">
                        <h4 style="color: #2c5530 !important; margin-bottom: 0.5rem;">🔵 SimCLR v1</h4>
                        <p style="color: #444444 !important; margin: 0;">Original contrastive learning approach with Vision Transformer backbone</p>
                    </div>
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; color: #333333 !important;">
                        <h4 style="color: #2c5530 !important; margin-bottom: 0.5rem;">🟢 SimCLR v2</h4>
                        <p style="color: #444444 !important; margin: 0;">Enhanced version with Vision Transformer and improved training</p>
                    </div>
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; color: #333333 !important;">
                        <h4 style="color: #2c5530 !important; margin-bottom: 0.5rem;">🟣 NNCLR</h4>
                        <p style="color: #444444 !important; margin: 0;">Nearest neighbor contrastive learning for better representations</p>
                    </div>
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; color: #333333 !important;">
                        <h4 style="color: #2c5530 !important; margin-bottom: 0.5rem;">🟠 Barlow Twins</h4>
                        <p style="color: #444444 !important; margin: 0;">Self-supervised learning via redundancy reduction</p>
                    </div>
                </div>
                
                <h3>🌱 Supported Crops</h3>
                <div style="columns: 3; column-gap: 2rem;">
                    {''.join([f'<p>• {crop}</p>' for crop in app.class_names])}
                </div>
                
                <h3>⚙️ System Requirements</h3>
                <ul>
                    <li><strong>Device:</strong> {DEVICE}</li>
                    <li><strong>PyTorch Version:</strong> {torch.__version__}</li>
                    <li><strong>CUDA Available:</strong> {torch.cuda.is_available()}</li>
                    <li><strong>Models Loaded:</strong> {len(app.models)}</li>
                </ul>
            </div>
        """)

    gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 12px; border: 2px solid #2196F3; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <p style="color: #1565C0 !important; margin-bottom: 0.8rem; font-weight: bold; font-size: 1.3rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">
                🌾 Advanced Agricultural Crop Classification System
            </p>
            <p style="color: #424242 !important; margin-bottom: 0.8rem; font-size: 1.1rem; font-weight: 500;">
                Developed using PyTorch, Transformers, and Gradio
            </p>
            <p style="color: #1565C0 !important; margin-bottom: 0; font-weight: bold; font-style: italic; font-size: 1.2rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">
                Developed by <a href="https://github.com/azizulabedinazmi" target="_blank" rel="noopener noreferrer" style="color: #1565C0; text-decoration: none;">Azizul Abedin Azmi</a>
            </p>
        </div>
    """)

if __name__ == "__main__":
    # Azure-compatible launch configuration
    import os
    
    # Get Azure environment variables or use defaults
    server_name = os.getenv("GRADIO_SERVER_NAME", "localhost")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    # Enable sharing for both local development and Azure deployment
    share_app = os.getenv("GRADIO_SHARE", "true").lower() == "true"
    
    print(f"🌾 Starting Agricultural Crop Classification App...")
    print(f"🌐 Server: {server_name}:{server_port}")
    print(f"🔗 Share: {share_app}")
    
    demo.launch(
        share=share_app,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        show_api=False,
        quiet=False,
        favicon_path=None,
        ssl_verify=False
    )
