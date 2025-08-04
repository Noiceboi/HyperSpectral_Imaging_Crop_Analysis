"""
Bước 6: Đánh giá và Test Mô hình 3D-ResNet
Mục tiêu: Đánh giá hiệu suất mô hình trên test set và tạo báo cáo chi tiết
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Import mô hình từ bước 5
import sys
sys.path.append('.')
from step5_3d_resnet_pytorch import ResNet3D_Eggplant

print("=" * 60)
print("🚀 BƯỚC 6: ĐÁNH GIÁ VÀ TEST MÔ HÌNH")
print("=" * 60)

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Device: {device}")

def load_test_data():
    """
    6.1. Tải dữ liệu test
    """
    print("\n" + "=" * 60)
    print("📂 BƯỚC 6.1: TẢI DỮ LIỆU TEST")
    print("=" * 60)
    
    DATA_DIR = 'processed_data'
    
    try:
        print("🔄 Đang tải test dataset...")
        
        # Tải dữ liệu test
        X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        
        # Tải metadata
        with open(os.path.join(DATA_DIR, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        print("✅ Tải dữ liệu test thành công!")
        print(f"📊 Test set: {X_test.shape} | Labels: {y_test.shape}")
        
        # Kiểm tra phân bố nhãn test
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        print("📊 Phân bố nhãn Test:")
        class_names = metadata['class_names']
        for label, count in zip(unique_test, counts_test):
            if label < len(class_names):
                print(f"   {class_names[label]} (Class {label}): {count:,} samples")
        
        return X_test, y_test, metadata
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu test: {str(e)}")
        return None, None, None

def prepare_test_data_pytorch(X_test, y_test):
    """
    6.2. Chuẩn bị dữ liệu test cho PyTorch
    """
    print("\n🔄 CHUẨN BỊ DỮ LIỆU TEST CHO PYTORCH:")
    
    # Chuyển đổi sang tensor và thêm channel dimension
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)  # (N, 1, 9, 9, 277)
    
    # Permute để có format chuẩn của PyTorch 3D: (N, C, D, H, W)
    X_test_tensor = X_test_tensor.permute(0, 1, 4, 2, 3)  # (N, 1, 277, 9, 9)
    
    y_test_tensor = torch.LongTensor(y_test)
    
    print(f"✅ Test tensor conversion hoàn thành!")
    print(f"📊 X_test tensor shape: {X_test_tensor.shape}")
    print(f"📊 y_test tensor shape: {y_test_tensor.shape}")
    
    return X_test_tensor, y_test_tensor

def load_trained_model(model_path, num_classes):
    """
    6.3. Tải mô hình đã huấn luyện
    """
    print("\n" + "=" * 60)
    print("🔄 BƯỚC 6.3: TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN")
    print("=" * 60)
    
    try:
        # Tạo mô hình với cùng architecture
        model = ResNet3D_Eggplant(num_classes=num_classes, input_channels=1)
        
        # Tải trọng số đã huấn luyện
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()  # Chế độ evaluation
        
        print(f"✅ Tải mô hình thành công từ: {model_path}")
        print(f"🎯 Mô hình đã được chuyển sang device: {device}")
        
        return model
        
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {str(e)}")
        return None

def evaluate_model(model, test_loader, class_names):
    """
    6.4. Đánh giá mô hình trên test set
    """
    print("\n" + "=" * 60)
    print("🧪 BƯỚC 6.4: ĐÁNH GIÁ MÔ HÌNH TRÊN TEST SET")
    print("=" * 60)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("🔄 Đang thực hiện prediction...")
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Lấy predictions
            _, predicted = torch.max(output, 1)
            
            # Lưu trữ kết quả
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Chuyển sang numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    print("✅ Hoàn thành prediction!")
    
    # Tính toán metrics
    print("\n📊 TÍNH TOÁN METRICS:")
    
    # Overall accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, labels=range(len(class_names))
    )
    
    # Macro và weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted'
    )
    
    print(f"\n📈 PER-CLASS METRICS:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 65)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
    
    print(f"\n📊 MACRO AVERAGES:")
    print(f"   Precision: {precision_macro:.4f}")
    print(f"   Recall: {recall_macro:.4f}")
    print(f"   F1-Score: {f1_macro:.4f}")
    
    print(f"\n⚖️ WEIGHTED AVERAGES:")
    print(f"   Precision: {precision_weighted:.4f}")
    print(f"   Recall: {recall_weighted:.4f}")
    print(f"   F1-Score: {f1_weighted:.4f}")
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }

def create_confusion_matrix(results, class_names, save_path='models/confusion_matrix.png'):
    """
    6.5. Tạo Confusion Matrix
    """
    print("\n" + "=" * 60)
    print("📊 BƯỚC 6.5: TẠO CONFUSION MATRIX")
    print("=" * 60)
    
    # Tính confusion matrix
    cm = confusion_matrix(results['targets'], results['predictions'])
    
    # Tính phần trăm
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix với số lượng mẫu
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Confusion matrix với phần trăm
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix đã được lưu: {save_path}")
    
    plt.show()
    
    # In thông tin chi tiết về confusion matrix
    print(f"\n📈 CONFUSION MATRIX ANALYSIS:")
    print(f"{'Class':<15} {'True Pos':<10} {'False Pos':<10} {'False Neg':<10} {'True Neg':<10}")
    print("-" * 65)
    
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        print(f"{class_name:<15} {tp:<10} {fp:<10} {fn:<10} {tn:<10}")

def create_classification_report(results, class_names, save_path='models/classification_report.txt'):
    """
    6.6. Tạo báo cáo phân loại chi tiết
    """
    print("\n" + "=" * 60)
    print("📋 BƯỚC 6.6: TẠO BÁO CÁO PHÂN LOẠI")
    print("=" * 60)
    
    # Tạo classification report
    report = classification_report(
        results['targets'], 
        results['predictions'], 
        target_names=class_names,
        digits=4
    )
    
    print("📊 CLASSIFICATION REPORT:")
    print(report)
    
    # Lưu báo cáo vào file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH 3D-RESNET\n")
        f.write("PHÂN LOẠI TÌNH TRẠNG DINH DƯỠNG N2 CÂY CÀ TÍM\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 40 + "\n")
        f.write(report)
        f.write("\n\n")
        
        f.write("PER-CLASS DETAILED METRICS:\n")
        f.write("-" * 40 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {results['precision_per_class'][i]:.4f}\n")
            f.write(f"  Recall: {results['recall_per_class'][i]:.4f}\n")
            f.write(f"  F1-Score: {results['f1_per_class'][i]:.4f}\n")
            f.write(f"  Support: {results['support_per_class'][i]}\n")
    
    print(f"📄 Báo cáo đã được lưu: {save_path}")

def create_prediction_confidence_analysis(results, class_names):
    """
    6.7. Phân tích độ tin cậy của predictions
    """
    print("\n" + "=" * 60)
    print("🔍 BƯỚC 6.7: PHÂN TÍCH ĐỘ TIN CẬY PREDICTIONS")
    print("=" * 60)
    
    # Tính confidence scores (max probability)
    confidence_scores = np.max(results['probabilities'], axis=1)
    
    # Tạo figure với multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram của confidence scores
    axes[0, 0].hist(confidence_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Prediction Confidence', fontweight='bold')
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confidence by class
    confidence_by_class = []
    for i, class_name in enumerate(class_names):
        class_mask = results['targets'] == i
        class_confidences = confidence_scores[class_mask]
        confidence_by_class.append(class_confidences)
    
    axes[0, 1].boxplot(confidence_by_class, labels=class_names)
    axes[0, 1].set_title('Confidence Distribution by Class', fontweight='bold')
    axes[0, 1].set_ylabel('Confidence Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Accuracy vs Confidence
    confidence_bins = np.linspace(0, 1, 11)
    accuracies_by_confidence = []
    counts_by_confidence = []
    
    for i in range(len(confidence_bins)-1):
        low, high = confidence_bins[i], confidence_bins[i+1]
        mask = (confidence_scores >= low) & (confidence_scores < high)
        if mask.sum() > 0:
            acc = (results['predictions'][mask] == results['targets'][mask]).mean()
            accuracies_by_confidence.append(acc)
            counts_by_confidence.append(mask.sum())
        else:
            accuracies_by_confidence.append(0)
            counts_by_confidence.append(0)
    
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    axes[1, 0].bar(bin_centers, accuracies_by_confidence, width=0.08, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Accuracy vs Confidence Level', fontweight='bold')
    axes[1, 0].set_xlabel('Confidence Level')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Error analysis - low confidence predictions
    low_confidence_threshold = 0.6
    low_conf_mask = confidence_scores < low_confidence_threshold
    error_mask = results['predictions'] != results['targets']
    
    low_conf_errors = low_conf_mask & error_mask
    high_conf_errors = (~low_conf_mask) & error_mask
    
    error_types = ['Low Confidence\nErrors', 'High Confidence\nErrors', 'Correct\nPredictions']
    error_counts = [low_conf_errors.sum(), high_conf_errors.sum(), 
                   (results['predictions'] == results['targets']).sum()]
    
    colors = ['red', 'orange', 'green']
    axes[1, 1].pie(error_counts, labels=error_types, colors=colors, autopct='%1.1f%%')
    axes[1, 1].set_title('Error Analysis by Confidence', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/confidence_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Biểu đồ confidence analysis đã được lưu: models/confidence_analysis.png")
    
    plt.show()
    
    # In thống kê
    print(f"\n📈 CONFIDENCE STATISTICS:")
    print(f"   Mean Confidence: {confidence_scores.mean():.4f}")
    print(f"   Std Confidence: {confidence_scores.std():.4f}")
    print(f"   Min Confidence: {confidence_scores.min():.4f}")
    print(f"   Max Confidence: {confidence_scores.max():.4f}")
    
    print(f"\n⚠️ LOW CONFIDENCE ANALYSIS (< {low_confidence_threshold}):")
    print(f"   Low confidence predictions: {low_conf_mask.sum()}")
    print(f"   Low confidence errors: {low_conf_errors.sum()}")
    print(f"   High confidence errors: {high_conf_errors.sum()}")

def main():
    """
    Hàm chính thực hiện toàn bộ quy trình Bước 6
    """
    print("🎯 Bắt đầu đánh giá mô hình...")
    
    # Bước 6.1: Tải dữ liệu test
    X_test, y_test, metadata = load_test_data()
    if X_test is None:
        return
    
    # Bước 6.2: Chuẩn bị dữ liệu test
    X_test_tensor, y_test_tensor = prepare_test_data_pytorch(X_test, y_test)
    
    # Tạo test DataLoader
    BATCH_SIZE = 64
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📦 Test DataLoader created: {len(test_loader)} batches")
    
    # Bước 6.3: Tải mô hình đã huấn luyện
    MODEL_PATH = 'models/best_3d_resnet_pytorch.pth'
    NUM_CLASSES = metadata['n_classes']
    CLASS_NAMES = metadata['class_names']
    
    model = load_trained_model(MODEL_PATH, NUM_CLASSES)
    if model is None:
        return
    
    # Bước 6.4: Đánh giá mô hình
    results = evaluate_model(model, test_loader, CLASS_NAMES)
    
    # Bước 6.5: Tạo confusion matrix
    create_confusion_matrix(results, CLASS_NAMES)
    
    # Bước 6.6: Tạo báo cáo phân loại
    create_classification_report(results, CLASS_NAMES)
    
    # Bước 6.7: Phân tích độ tin cậy
    create_prediction_confidence_analysis(results, CLASS_NAMES)
    
    # Tổng kết
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH BƯỚC 6")
    print("=" * 60)
    print("✅ Đánh giá mô hình trên test set hoàn tất")
    print("✅ Confusion matrix đã được tạo")
    print("✅ Báo cáo phân loại đã được lưu")
    print("✅ Phân tích độ tin cậy hoàn thành")
    print(f"🎯 Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"📊 F1-Score (Macro): {results['f1_macro']:.4f}")
    print(f"📊 F1-Score (Weighted): {results['f1_weighted']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
