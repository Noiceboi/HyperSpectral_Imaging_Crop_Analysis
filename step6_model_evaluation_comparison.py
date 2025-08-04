"""
Bước 6: Đánh giá và So sánh Mô hình 3D-ResNet
Mục tiêu: So sánh performance giữa Standard ResNet và Enhanced ResNet với Channel Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support,
                           cohen_kappa_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import model architectures from step5
import sys
sys.path.append('.')

print("=" * 80)
print("🚀 BƯỚC 6: ĐÁNH GIÁ VÀ SO SÁNH MÔ HÌNH")
print("📊 Standard ResNet vs Enhanced ResNet with Channel Attention")
print("=" * 80)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Device: {device}")

class ChannelAttention3D(nn.Module):
    """
    Channel Attention Module dành riêng cho Hyperspectral data
    """
    def __init__(self, in_channels, reduction=16, discriminative_bands=None):
        super(ChannelAttention3D, self).__init__()
        
        self.in_channels = in_channels
        self.discriminative_bands = discriminative_bands
        
        # Global Average Pooling và Max Pooling trên spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # Discriminative bands emphasis weights
        if discriminative_bands is not None:
            self.disc_weight = nn.Parameter(torch.ones(len(discriminative_bands)) * 2.0)
        else:
            self.disc_weight = None
            
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Global pooling
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine average and max pooling
        attention_weights = self.sigmoid(avg_out + max_out)
        
        # Apply discriminative bands emphasis if available
        if self.disc_weight is not None and self.discriminative_bands is not None:
            band_emphasis = torch.ones_like(attention_weights)
            
            for i, band_idx in enumerate(self.discriminative_bands):
                if band_idx < attention_weights.shape[2]:
                    band_emphasis[:, :, band_idx, :, :] *= self.disc_weight[i]
            
            attention_weights = attention_weights * band_emphasis
        
        return x * attention_weights

class SpectralAttention3D(nn.Module):
    """
    Spectral Attention Module
    """
    def __init__(self, spectral_size, reduction=8):
        super(SpectralAttention3D, self).__init__()
        
        self.spectral_fc = nn.Sequential(
            nn.Linear(spectral_size, spectral_size // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(spectral_size // reduction, spectral_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        N, C, D, H, W = x.size()
        
        # Global average pooling over spatial dimensions
        spectral_features = x.mean(dim=[3, 4])  # (N, C, D)
        
        # Apply attention to each channel's spectral signature
        attended_features = []
        for c in range(C):
            channel_spectral = spectral_features[:, c, :]  # (N, D)
            attention_weights = self.spectral_fc(channel_spectral)  # (N, D)
            attended_features.append(attention_weights.unsqueeze(1))  # (N, 1, D)
        
        spectral_attention = torch.cat(attended_features, dim=1)  # (N, C, D)
        spectral_attention = spectral_attention.unsqueeze(-1).unsqueeze(-1)  # (N, C, D, 1, 1)
        
        return x * spectral_attention

class ResidualBlock3D(nn.Module):
    """
    Khối Residual 3D
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock3D, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet3D_Eggplant_WithAttention(nn.Module):
    """
    Enhanced 3D-ResNet với Channel và Spectral Attention
    """
    def __init__(self, num_classes=3, input_channels=1, discriminative_bands=None):
        super(ResNet3D_Eggplant_WithAttention, self).__init__()
        
        self.discriminative_bands = discriminative_bands
        
        # Giai đoạn 1: Trích xuất đặc trưng ban đầu
        self.initial_conv = nn.Conv3d(input_channels, 32, kernel_size=(7, 3, 3), 
                                     stride=(2, 1, 1), padding=(3, 1, 1))
        self.initial_bn = nn.BatchNorm3d(32)
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # Channel Attention
        self.channel_attention1 = ChannelAttention3D(32, reduction=8, 
                                                   discriminative_bands=discriminative_bands)
        
        # Spectral Attention
        estimated_spectral_size = 69
        self.spectral_attention1 = SpectralAttention3D(estimated_spectral_size, reduction=8)
        
        # Giai đoạn 2: Các khối residual
        self.res_block1_1 = ResidualBlock3D(32, 32)
        self.res_block1_2 = ResidualBlock3D(32, 32)
        self.channel_attention2 = ChannelAttention3D(32, reduction=8,
                                                   discriminative_bands=discriminative_bands)
        
        # Giai đoạn 3: Transition
        self.transition_conv = nn.Conv3d(32, 64, kernel_size=1, stride=(2, 1, 1))
        self.transition_bn = nn.BatchNorm3d(64)
        self.transition_relu = nn.ReLU(inplace=True)
        self.transition_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.channel_attention3 = ChannelAttention3D(64, reduction=8,
                                                   discriminative_bands=discriminative_bands)
        
        self.res_block2_1 = ResidualBlock3D(64, 64)
        self.res_block2_2 = ResidualBlock3D(64, 64)
        self.channel_attention4 = ChannelAttention3D(64, reduction=8,
                                                   discriminative_bands=discriminative_bands)
        
        # Giai đoạn 4: Classifier
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Giai đoạn 1
        x = self.initial_relu(self.initial_bn(self.initial_conv(x)))
        x = self.initial_pool(x)
        x = self.channel_attention1(x)
        
        # Apply spectral attention with error handling
        try:
            if x.shape[2] != self.spectral_attention1.spectral_fc[0].in_features:
                actual_spectral_size = x.shape[2]
                self.spectral_attention1 = SpectralAttention3D(actual_spectral_size, reduction=8).to(x.device)
            x = self.spectral_attention1(x)
        except:
            pass
        
        # Giai đoạn 2
        x = self.res_block1_1(x)
        x = self.res_block1_2(x)
        x = self.channel_attention2(x)
        
        # Giai đoạn 3
        x = self.transition_relu(self.transition_bn(self.transition_conv(x)))
        x = self.transition_pool(x)
        x = self.channel_attention3(x)
        
        x = self.res_block2_1(x)
        x = self.res_block2_2(x)
        x = self.channel_attention4(x)
        
        # Giai đoạn 4
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class ResNet3D_Eggplant(nn.Module):
    """
    Standard 3D-ResNet
    """
    def __init__(self, num_classes=3, input_channels=1):
        super(ResNet3D_Eggplant, self).__init__()
        
        # Giai đoạn 1
        self.initial_conv = nn.Conv3d(input_channels, 32, kernel_size=(7, 3, 3), 
                                     stride=(2, 1, 1), padding=(3, 1, 1))
        self.initial_bn = nn.BatchNorm3d(32)
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # Giai đoạn 2
        self.res_block1_1 = ResidualBlock3D(32, 32)
        self.res_block1_2 = ResidualBlock3D(32, 32)
        
        # Giai đoạn 3
        self.transition_conv = nn.Conv3d(32, 64, kernel_size=1, stride=(2, 1, 1))
        self.transition_bn = nn.BatchNorm3d(64)
        self.transition_relu = nn.ReLU(inplace=True)
        self.transition_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.res_block2_1 = ResidualBlock3D(64, 64)
        self.res_block2_2 = ResidualBlock3D(64, 64)
        
        # Giai đoạn 4
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Giai đoạn 1
        x = self.initial_relu(self.initial_bn(self.initial_conv(x)))
        x = self.initial_pool(x)
        
        # Giai đoạn 2
        x = self.res_block1_1(x)
        x = self.res_block1_2(x)
        
        # Giai đoạn 3
        x = self.transition_relu(self.transition_bn(self.transition_conv(x)))
        x = self.transition_pool(x)
        x = self.res_block2_1(x)
        x = self.res_block2_2(x)
        
        # Giai đoạn 4
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def load_test_data():
    """
    Tải test data và metadata
    """
    print("\n" + "=" * 60)
    print("📂 BƯỚC 6.1: TẢI TEST DATA")
    print("=" * 60)
    
    DATA_DIR = 'processed_data'
    
    try:
        # Load test data
        X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        
        # Load metadata
        with open(os.path.join(DATA_DIR, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Extract discriminative bands
        discriminative_info = metadata.get('discriminative_bands', None)
        discriminative_indices = None
        if discriminative_info:
            discriminative_indices = discriminative_info.get('indices', None)
            target_wavelengths = discriminative_info.get('target_wavelengths', None)
            print(f"🎯 Discriminative bands loaded: {len(discriminative_indices) if discriminative_indices else 0}")
        
        # Prepare PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)  # Add channel dim
        X_test_tensor = X_test_tensor.permute(0, 1, 4, 2, 3)    # (N, C, D, H, W)
        y_test_tensor = torch.LongTensor(y_test)
        
        print(f"✅ Test data loaded successfully!")
        print(f"📊 Test set shape: {X_test_tensor.shape}")
        print(f"📊 Test labels shape: {y_test_tensor.shape}")
        print(f"📊 Number of classes: {metadata['n_classes']}")
        
        return X_test_tensor, y_test_tensor, metadata, discriminative_indices
        
    except Exception as e:
        print(f"❌ Error loading test data: {str(e)}")
        return None, None, None, None

def load_models(metadata, discriminative_indices):
    """
    Load cả 2 models: Standard và Attention
    """
    print("\n" + "=" * 60)
    print("🏗️ BƯỚC 6.2: TẢI CÁC MODELS")
    print("=" * 60)
    
    models = {}
    NUM_CLASSES = metadata['n_classes']
    
    # Model paths
    model_paths = {
        'Standard_ResNet': 'models/best_3d_resnet_pytorch.pth',
        'Attention_ResNet': 'models/best_3d_resnet_pytorch_attention.pth'
    }
    
    # Load Standard ResNet
    if os.path.exists(model_paths['Standard_ResNet']):
        print("🔄 Loading Standard ResNet...")
        model_standard = ResNet3D_Eggplant(num_classes=NUM_CLASSES, input_channels=1)
        model_standard.load_state_dict(torch.load(model_paths['Standard_ResNet'], map_location=device))
        model_standard.to(device)
        model_standard.eval()
        models['Standard_ResNet'] = model_standard
        print("✅ Standard ResNet loaded successfully!")
    else:
        print("⚠️ Standard ResNet model not found!")
    
    # Load Attention ResNet
    if os.path.exists(model_paths['Attention_ResNet']):
        print("🔄 Loading Attention ResNet...")
        model_attention = ResNet3D_Eggplant_WithAttention(
            num_classes=NUM_CLASSES, 
            input_channels=1,
            discriminative_bands=discriminative_indices
        )
        model_attention.load_state_dict(torch.load(model_paths['Attention_ResNet'], map_location=device))
        model_attention.to(device)
        model_attention.eval()
        models['Attention_ResNet'] = model_attention
        print("✅ Attention ResNet loaded successfully!")
    else:
        print("⚠️ Attention ResNet model not found!")
    
    print(f"📊 Total models loaded: {len(models)}")
    return models

def evaluate_model(model, test_loader, model_name):
    """
    Đánh giá một model trên test set
    """
    print(f"\n🔍 Evaluating {model_name}...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Testing {model_name}"):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    results = {
        'model_name': model_name,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'kappa': kappa,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    print(f"✅ {model_name} evaluation completed!")
    print(f"   🎯 Accuracy: {accuracy:.4f}")
    print(f"   📊 F1-Score: {f1:.4f}")
    print(f"   🔍 Kappa: {kappa:.4f}")
    
    return results

def plot_comparison_results(results_list, class_names):
    """
    Vẽ biểu đồ so sánh kết quả các models
    """
    print("\n" + "=" * 60)
    print("📊 BƯỚC 6.3: TRỰC QUAN HÓA SO SÁNH")
    print("=" * 60)
    
    # Thiết lập style
    plt.style.use('seaborn-v0_8')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Confusion Matrix Comparison
    fig, axes = plt.subplots(1, len(results_list), figsize=(6*len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]
    
    for idx, results in enumerate(results_list):
        cm = results['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=axes[idx],
                   cbar_kws={'label': 'Normalized Frequency'})
        
        axes[idx].set_title(f'{results["model_name"]}\nAccuracy: {results["accuracy"]:.4f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_attention_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Confusion matrix comparison saved!")
    
    # 2. Metrics Comparison Bar Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    model_names = [r['model_name'] for r in results_list]
    
    # Overall metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa']
    metric_values = {
        'Accuracy': [r['accuracy'] for r in results_list],
        'Precision': [r['precision'] for r in results_list],
        'Recall': [r['recall'] for r in results_list],
        'F1-Score': [r['f1_score'] for r in results_list],
        'Kappa': [r['kappa'] for r in results_list]
    }
    
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        ax1.bar(x + i*width, metric_values[metric], width, 
               label=metric, color=colors[i % len(colors)], alpha=0.8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Performance Comparison', fontweight='bold')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Per-class F1 scores
    for i, class_name in enumerate(class_names):
        class_f1_scores = [r['f1_per_class'][i] for r in results_list]
        ax2.bar([j + i*width for j in range(len(model_names))], class_f1_scores, 
               width, label=f'{class_name}', alpha=0.8)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Per-Class F1-Score Comparison', fontweight='bold')
    ax2.set_xticks([j + width for j in range(len(model_names))])
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Per-class Precision
    for i, class_name in enumerate(class_names):
        class_precision_scores = [r['precision_per_class'][i] for r in results_list]
        ax3.bar([j + i*width for j in range(len(model_names))], class_precision_scores, 
               width, label=f'{class_name}', alpha=0.8)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Precision')
    ax3.set_title('Per-Class Precision Comparison', fontweight='bold')
    ax3.set_xticks([j + width for j in range(len(model_names))])
    ax3.set_xticklabels(model_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)
    
    # Per-class Recall
    for i, class_name in enumerate(class_names):
        class_recall_scores = [r['recall_per_class'][i] for r in results_list]
        ax4.bar([j + i*width for j in range(len(model_names))], class_recall_scores, 
               width, label=f'{class_name}', alpha=0.8)
    
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Recall')
    ax4.set_title('Per-Class Recall Comparison', fontweight='bold')
    ax4.set_xticks([j + width for j in range(len(model_names))])
    ax4.set_xticklabels(model_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('models/metrics_comparison_attention.png', dpi=300, bbox_inches='tight')
    print("📊 Metrics comparison saved!")
    
    plt.show()

def generate_detailed_report(results_list, class_names):
    """
    Tạo báo cáo chi tiết so sánh các models
    """
    print("\n" + "=" * 60)
    print("📋 BƯỚC 6.4: TẠO BÁO CÁO CHI TIẾT")
    print("=" * 60)
    
    report_path = 'models/detailed_comparison_report_attention.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED MODEL COMPARISON REPORT\n")
        f.write("Standard ResNet vs Enhanced ResNet with Channel Attention\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary table
        f.write("SUMMARY COMPARISON:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<25} {'Accuracy':<10} {'F1-Score':<10} {'Kappa':<10} {'Precision':<10} {'Recall':<10}\n")
        f.write("-" * 80 + "\n")
        
        for results in results_list:
            f.write(f"{results['model_name']:<25} ")
            f.write(f"{results['accuracy']:<10.4f} ")
            f.write(f"{results['f1_score']:<10.4f} ")
            f.write(f"{results['kappa']:<10.4f} ")
            f.write(f"{results['precision']:<10.4f} ")
            f.write(f"{results['recall']:<10.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Detailed per-model analysis
        for results in results_list:
            f.write(f"DETAILED ANALYSIS: {results['model_name']}\n")
            f.write("-" * 50 + "\n")
            
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Overall F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"Cohen's Kappa: {results['kappa']:.4f}\n\n")
            
            f.write("Per-Class Performance:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {results['precision_per_class'][i]:.4f}\n")
                f.write(f"    Recall:    {results['recall_per_class'][i]:.4f}\n")
                f.write(f"    F1-Score:  {results['f1_per_class'][i]:.4f}\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"{'':>12}")
            for class_name in class_names:
                f.write(f"{class_name:>12}")
            f.write("\n")
            
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name:>12}")
                for j in range(len(class_names)):
                    f.write(f"{results['confusion_matrix'][i][j]:>12}")
                f.write("\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
        
        # Performance improvement analysis
        if len(results_list) == 2:
            f.write("PERFORMANCE IMPROVEMENT ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            
            standard_results = results_list[0] if 'Standard' in results_list[0]['model_name'] else results_list[1]
            attention_results = results_list[1] if 'Attention' in results_list[1]['model_name'] else results_list[0]
            
            acc_improvement = attention_results['accuracy'] - standard_results['accuracy']
            f1_improvement = attention_results['f1_score'] - standard_results['f1_score']
            
            f.write(f"Accuracy Improvement: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)\n")
            f.write(f"F1-Score Improvement: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)\n\n")
            
            f.write("Per-Class Improvements:\n")
            for i, class_name in enumerate(class_names):
                prec_imp = attention_results['precision_per_class'][i] - standard_results['precision_per_class'][i]
                rec_imp = attention_results['recall_per_class'][i] - standard_results['recall_per_class'][i]
                f1_imp = attention_results['f1_per_class'][i] - standard_results['f1_per_class'][i]
                
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {prec_imp:+.4f}\n")
                f.write(f"    Recall:    {rec_imp:+.4f}\n")
                f.write(f"    F1-Score:  {f1_imp:+.4f}\n")
    
    print(f"📋 Detailed report saved: {report_path}")

def main():
    """
    Hàm chính thực hiện đánh giá và so sánh models
    """
    start_time = time.time()
    
    # Load test data
    X_test, y_test, metadata, discriminative_indices = load_test_data()
    if X_test is None:
        print("❌ Failed to load test data!")
        return
    
    # Create test DataLoader
    BATCH_SIZE = 64
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load models
    models = load_models(metadata, discriminative_indices)
    if not models:
        print("❌ No models found!")
        return
    
    # Evaluate each model
    results_list = []
    class_names = metadata['class_names']
    
    print("\n" + "=" * 60)
    print("🔍 BƯỚC 6.3: ĐÁNH GIÁ MODELS")
    print("=" * 60)
    
    for model_name, model in models.items():
        results = evaluate_model(model, test_loader, model_name)
        results_list.append(results)
    
    # Plot comparison results
    plot_comparison_results(results_list, class_names)
    
    # Generate detailed report
    generate_detailed_report(results_list, class_names)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("🎉 MODEL COMPARISON COMPLETED!")
    print("=" * 80)
    
    print("📊 FINAL RESULTS SUMMARY:")
    for results in results_list:
        print(f"   {results['model_name']:>20}: Acc={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
    
    if len(results_list) == 2:
        standard_results = results_list[0] if 'Standard' in results_list[0]['model_name'] else results_list[1]
        attention_results = results_list[1] if 'Attention' in results_list[1]['model_name'] else results_list[0]
        
        acc_improvement = attention_results['accuracy'] - standard_results['accuracy']
        print(f"\n🚀 CHANNEL ATTENTION IMPROVEMENT:")
        print(f"   📈 Accuracy: {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
        print(f"   🎯 Attention Model Accuracy: {attention_results['accuracy']:.4f}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total evaluation time: {total_time/60:.1f} minutes")
    print("✅ All results saved in models/ directory")
    print("=" * 80)

if __name__ == "__main__":
    main()
