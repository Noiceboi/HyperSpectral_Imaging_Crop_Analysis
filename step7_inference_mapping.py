"""
Bước 7: Inference và Tạo Bản đồ Phân loại Dinh dưỡng N2
Mục tiêu: Áp dụng mô hình đã huấn luyện lên toàn bộ ảnh HSI để tạo bản đồ phân loại dinh dưỡng
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import spectral.io.envi as envi
import pickle
from tqdm import tqdm
import cv2
from skimage import filters
import time

# Import mô hình từ bước 5
import sys
sys.path.append('.')
from step5_3d_resnet_pytorch import ResNet3D_Eggplant

print("=" * 60)
print("🚀 BƯỚC 7: INFERENCE VÀ TẠO BẢN ĐỒ PHÂN LOẠI")
print("=" * 60)

# Kiểm tra GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Device: {device}")

def load_full_hsi_data():
    """
    7.1. Tải toàn bộ dữ liệu HSI
    """
    print("\n" + "=" * 60)
    print("📂 BƯỚC 7.1: TẢI TOÀN BỘ DỮ LIỆU HSI")
    print("=" * 60)
    
    try:
        # Đường dẫn đến file HSI
        hsi_file = 'D:\HyperSpectral_Imaging_Crop_Analysis\Eggplant_Crop\Eggplant_Crop\Eggplant_Reflectance_Data'
        gt_file = 'D:\HyperSpectral_Imaging_Crop_Analysis\Eggplant_Crop\Eggplant_Crop\Eggplant_N2_Concentration_GT'
        
        print("🔄 Đang tải Hyperspectral Image...")
        
        # Tải dữ liệu HSI
        hsi_data = envi.open(hsi_file + '.hdr', hsi_file)
        hsi_array = hsi_data.load()
        
        print("🔄 Đang tải Ground Truth...")
        
        # Tải ground truth
        gt_data = envi.open(gt_file + '.hdr', gt_file)
        gt_array = gt_data.load()
        
        print("✅ Tải dữ liệu thành công!")
        print(f"📊 HSI Data shape: {hsi_array.shape}")
        print(f"📊 Ground Truth shape: {gt_array.shape}")
        
        # Kiểm tra thống kê cơ bản
        print(f"\n📈 HSI STATISTICS:")
        print(f"   Min: {hsi_array.min():.4f}")
        print(f"   Max: {hsi_array.max():.4f}")
        print(f"   Mean: {hsi_array.mean():.4f}")
        print(f"   Std: {hsi_array.std():.4f}")
        
        print(f"\n📈 GROUND TRUTH STATISTICS:")
        unique_values, counts = np.unique(gt_array, return_counts=True)
        print(f"   Unique values: {unique_values}")
        print(f"   Counts: {counts}")
        
        return hsi_array, gt_array
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        return None, None

def load_trained_model_and_metadata():
    """
    7.2. Tải mô hình và metadata
    """
    print("\n" + "=" * 60)
    print("🧠 BƯỚC 7.2: TẢI MÔ HÌNH VÀ METADATA")
    print("=" * 60)
    
    try:
        # Tải metadata
        with open('processed_data/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        print("✅ Tải metadata thành công!")
        print(f"📊 Number of classes: {metadata['n_classes']}")
        print(f"📊 Class names: {metadata['class_names']}")
        print(f"📊 Patch size: {metadata['patch_size']}")
        print(f"📊 Number of bands: {metadata['n_bands']}")
        
        # Tạo và tải mô hình
        model = ResNet3D_Eggplant(num_classes=metadata['n_classes'], input_channels=1)
        model_path = 'models/best_3d_resnet_pytorch.pth'
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        
        print(f"✅ Tải mô hình thành công từ: {model_path}")
        
        return model, metadata
        
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {str(e)}")
        return None, None

def normalize_hsi_data(hsi_array, metadata):
    """
    7.3. Chuẩn hóa dữ liệu HSI (áp dụng cùng normalization như training)
    """
    print("\n🔄 CHUẨN HÓA DỮ LIỆU HSI...")
    
    # Áp dụng cùng phương pháp normalization như trong training
    # Min-Max normalization cho từng band
    hsi_normalized = np.zeros_like(hsi_array, dtype=np.float32)
    
    for band in range(hsi_array.shape[2]):
        band_data = hsi_array[:, :, band].astype(np.float32)
        band_min = band_data.min()
        band_max = band_data.max()
        
        if band_max > band_min:
            hsi_normalized[:, :, band] = (band_data - band_min) / (band_max - band_min)
        else:
            hsi_normalized[:, :, band] = 0
    
    print("✅ Chuẩn hóa hoàn thành!")
    print(f"📊 Normalized range: [{hsi_normalized.min():.4f}, {hsi_normalized.max():.4f}]")
    
    return hsi_normalized

def extract_patch_at_position(hsi_data, row, col, patch_size=9):
    """
    7.4. Trích xuất patch 3D tại vị trí cụ thể
    """
    height, width, bands = hsi_data.shape
    half_size = patch_size // 2
    
    # Tính toán boundaries với padding
    row_start = max(0, row - half_size)
    row_end = min(height, row + half_size + 1)
    col_start = max(0, col - half_size)
    col_end = min(width, col + half_size + 1)
    
    # Trích xuất patch
    patch = hsi_data[row_start:row_end, col_start:col_end, :]
    
    # Padding nếu cần thiết
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        padded_patch = np.zeros((patch_size, patch_size, bands), dtype=np.float32)
        
        # Tính toán vị trí để đặt patch vào center
        start_row = (patch_size - patch.shape[0]) // 2
        start_col = (patch_size - patch.shape[1]) // 2
        
        padded_patch[start_row:start_row + patch.shape[0], 
                    start_col:start_col + patch.shape[1], :] = patch
        
        return padded_patch
    
    return patch

def predict_batch_patches(model, patches_batch):
    """
    7.5. Dự đoán cho một batch patches
    """
    with torch.no_grad():
        # Chuyển đổi sang tensor
        patches_tensor = torch.FloatTensor(patches_batch).unsqueeze(1)  # (N, 1, 9, 9, 277)
        patches_tensor = patches_tensor.permute(0, 1, 4, 2, 3)  # (N, 1, 277, 9, 9)
        patches_tensor = patches_tensor.to(device)
        
        # Forward pass
        outputs = model(patches_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()

def create_classification_map(hsi_data, model, metadata, batch_size=512, stride=10, max_pixels=50000):
    """
    7.6. Tạo bản đồ phân loại cho toàn bộ ảnh HSI (Tối ưu hóa cho demo)
    """
    print("\n" + "=" * 60)
    print("🗺️ BƯỚC 7.6: TẠO BẢN ĐỒ PHÂN LOẠI (DEMO MODE)")
    print("=" * 60)
    
    height, width, bands = hsi_data.shape
    patch_size = metadata['patch_size']
    
    # Tạo bản đồ kết quả
    classification_map = np.zeros((height, width), dtype=np.uint8)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    
    print(f"📊 Creating classification map (optimized):")
    print(f"   Image size: {height} x {width}")
    print(f"   Patch size: {patch_size} x {patch_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Stride: {stride} (sampling every {stride}th pixel)")
    print(f"   Max pixels limit: {max_pixels:,}")
    
    # Tính toán tổng số pixels cần xử lý với stride lớn hơn
    total_pixels = 0
    valid_positions = []
    
    for row in range(patch_size//2, height - patch_size//2, stride):
        for col in range(patch_size//2, width - patch_size//2, stride):
            valid_positions.append((row, col))
            total_pixels += 1
            # Giới hạn số pixels để demo nhanh
            if total_pixels >= max_pixels:
                break
        if total_pixels >= max_pixels:
            break
    
    print(f"   Actual pixels to process: {total_pixels:,}")
    print(f"   Estimated processing time: ~{total_pixels/1000:.1f} minutes")
    
    # Xử lý theo batch
    patches_batch = []
    positions_batch = []
    
    start_time = time.time()
    
    with tqdm(total=total_pixels, desc="🔄 Processing pixels") as pbar:
        for i, (row, col) in enumerate(valid_positions):
            # Trích xuất patch
            patch = extract_patch_at_position(hsi_data, row, col, patch_size)
            patches_batch.append(patch)
            positions_batch.append((row, col))
            
            # Xử lý batch khi đủ số lượng hoặc cuối danh sách
            if len(patches_batch) == batch_size or i == len(valid_positions) - 1:
                # Dự đoán cho batch
                predictions, probabilities = predict_batch_patches(model, np.array(patches_batch))
                
                # Gán kết quả vào bản đồ
                for j, (pred, prob, (r, c)) in enumerate(zip(predictions, probabilities, positions_batch)):
                    classification_map[r, c] = pred
                    confidence_map[r, c] = prob.max()
                
                # Reset batch
                patches_batch = []
                positions_batch = []
                
                # Update progress bar với số lượng thực tế đã xử lý
                pbar.update(len(predictions))
    
    processing_time = time.time() - start_time
    print(f"\n✅ Hoàn thành tạo bản đồ phân loại (demo mode)!")
    print(f"⏱️ Thời gian xử lý: {processing_time/60:.1f} phút")
    print(f"📊 Pixels per second: {total_pixels/processing_time:.1f}")
    print(f"🎯 Processed {total_pixels:,} pixels out of {height*width:,} total pixels")
    
    return classification_map, confidence_map

def visualize_classification_map(classification_map, confidence_map, metadata, gt_array=None):
    """
    7.7. Trực quan hóa bản đồ phân loại
    """
    print("\n" + "=" * 60)
    print("🎨 BƯỚC 7.7: TRỰC QUAN HÓA BẢN ĐỒ PHÂN LOẠI")
    print("=" * 60)
    
    class_names = metadata['class_names']
    
    # Định nghĩa màu sắc cho từng class
    colors = ['#FFD700', '#32CD32', '#1E90FF']  # Vàng, Xanh lá, Xanh dương
    class_colors = {
        0: colors[0],  # Low N2 - Vàng
        1: colors[1],  # Medium N2 - Xanh lá  
        2: colors[2]   # High N2 - Xanh dương
    }
    
    # Tạo custom colormap
    cmap = ListedColormap(colors)
    
    # Tạo figure với multiple subplots
    if gt_array is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax1 = axes[0]
        ax2 = axes[1]
    
    # 1. Bản đồ phân loại chính
    im1 = ax1.imshow(classification_map, cmap=cmap, vmin=0, vmax=2)
    ax1.set_title('🗺️ Nitrogen Classification Map', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    
    # Tạo legend
    patches = [mpatches.Patch(color=colors[i], label=f'{class_names[i]}') 
              for i in range(len(class_names))]
    ax1.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1))
    
    # 2. Bản đồ confidence
    im2 = ax2.imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
    ax2.set_title('🎯 Prediction Confidence Map', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    
    # Colorbar cho confidence
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Confidence Score', rotation=270, labelpad=20)
    
    # 3. Ground Truth (nếu có)
    if gt_array is not None:
        ax3 = axes[1, 0]
        # Đảm bảo gt_array có cùng shape với classification_map
        if len(gt_array.shape) == 3:
            gt_array = gt_array.squeeze()  # Loại bỏ dimension thừa
        
        # Chuyển đổi GT values sang class indices
        gt_display = np.zeros_like(gt_array, dtype=np.uint8)
        unique_gt = np.unique(gt_array)
        unique_gt = unique_gt[unique_gt > 0]  # Loại bỏ background
        
        for i, gt_val in enumerate(sorted(unique_gt)):
            gt_display[gt_array == gt_val] = i
        
        im3 = ax3.imshow(gt_display, cmap=cmap, vmin=0, vmax=2)
        ax3.set_title('📋 Ground Truth', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Width (pixels)')
        ax3.set_ylabel('Height (pixels)')
        
        # 4. Difference map
        ax4 = axes[1, 1]
        diff_map = np.abs(classification_map.astype(int) - gt_display.astype(int))
        im4 = ax4.imshow(diff_map, cmap='Reds', vmin=0, vmax=2)
        ax4.set_title('❌ Prediction Errors', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Width (pixels)')
        ax4.set_ylabel('Height (pixels)')
        
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8)
        cbar4.set_label('Absolute Error', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Lưu hình
    save_path = 'models/nitrogen_classification_map.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ Bản đồ phân loại đã được lưu: {save_path}")
    
    plt.show()
    
    # In thống kê
    print(f"\n📊 THỐNG KÊ BẢN ĐỒ PHÂN LOẠI:")
    unique_classes, class_counts = np.unique(classification_map, return_counts=True)
    total_pixels = classification_map.size
    
    for class_idx, count in zip(unique_classes, class_counts):
        if class_idx < len(class_names):
            percentage = (count / total_pixels) * 100
            print(f"   {class_names[class_idx]}: {count:,} pixels ({percentage:.1f}%)")
    
    print(f"\n🎯 CONFIDENCE STATISTICS:")
    print(f"   Mean confidence: {confidence_map.mean():.4f}")
    print(f"   Std confidence: {confidence_map.std():.4f}")
    print(f"   Min confidence: {confidence_map.min():.4f}")
    print(f"   Max confidence: {confidence_map.max():.4f}")

def create_detailed_analysis_plots(classification_map, confidence_map, metadata, gt_array=None):
    """
    7.8. Tạo các biểu đồ phân tích chi tiết
    """
    print("\n" + "=" * 60)
    print("📈 BƯỚC 7.8: PHÂN TÍCH CHI TIẾT")
    print("=" * 60)
    
    class_names = metadata['class_names']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Histogram phân bố class
    unique_classes, class_counts = np.unique(classification_map, return_counts=True)
    colors = ['#FFD700', '#32CD32', '#1E90FF']
    
    axes[0, 0].bar([class_names[i] for i in unique_classes], class_counts, 
                   color=[colors[i] for i in unique_classes])
    axes[0, 0].set_title('Class Distribution', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Pixels')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Confidence distribution by class
    confidence_by_class = []
    for class_idx in unique_classes:
        class_mask = classification_map == class_idx
        class_confidences = confidence_map[class_mask]
        confidence_by_class.append(class_confidences)
    
    axes[0, 1].boxplot(confidence_by_class, tick_labels=[class_names[i] for i in unique_classes])
    axes[0, 1].set_title('Confidence by Class', fontweight='bold')
    axes[0, 1].set_ylabel('Confidence Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Spatial distribution heatmap
    # Tạo meshgrid cho spatial distribution
    height, width = classification_map.shape
    y_coords, x_coords = np.mgrid[0:height:complex(0, 50), 0:width:complex(0, 50)]
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    
    # Sample classification values tại các điểm grid
    weights = []
    for y, x in zip(y_flat, x_flat):
        y_idx = int(y) if int(y) < height else height - 1
        x_idx = int(x) if int(x) < width else width - 1
        weights.append(classification_map[y_idx, x_idx])
    
    axes[0, 2].hist2d(x_flat, y_flat, weights=weights, bins=50, cmap='viridis')
    axes[0, 2].set_title('Spatial Class Distribution', fontweight='bold')
    axes[0, 2].set_xlabel('Width')
    axes[0, 2].set_ylabel('Height')
    
    # 4. Confidence histogram
    axes[1, 0].hist(confidence_map.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Confidence Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Confidence Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(confidence_map.mean(), color='red', linestyle='--', 
                       label=f'Mean: {confidence_map.mean():.3f}')
    axes[1, 0].legend()
    
    # 5. Low confidence regions
    low_conf_threshold = 0.6
    low_conf_mask = confidence_map < low_conf_threshold
    axes[1, 1].imshow(low_conf_mask, cmap='Reds')
    axes[1, 1].set_title(f'Low Confidence Regions (<{low_conf_threshold})', fontweight='bold')
    axes[1, 1].set_xlabel('Width')
    axes[1, 1].set_ylabel('Height')
    
    # 6. Class transition analysis
    # Tạo gradient map để hiển thị vùng chuyển tiếp giữa các class
    grad_x = np.abs(np.gradient(classification_map.astype(float), axis=1))
    grad_y = np.abs(np.gradient(classification_map.astype(float), axis=0))
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    axes[1, 2].imshow(gradient_magnitude, cmap='hot')
    axes[1, 2].set_title('Class Transition Boundaries', fontweight='bold')
    axes[1, 2].set_xlabel('Width')
    axes[1, 2].set_ylabel('Height')
    
    plt.tight_layout()
    plt.savefig('models/detailed_classification_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Biểu đồ phân tích chi tiết đã được lưu: models/detailed_classification_analysis.png")
    
    plt.show()

def save_classification_results(classification_map, confidence_map, metadata):
    """
    7.9. Lưu kết quả phân loại
    """
    print("\n" + "=" * 60)
    print("💾 BƯỚC 7.9: LƯU KẾT QUẢ PHÂN LOẠI")
    print("=" * 60)
    
    # Tạo thư mục results nếu chưa có
    os.makedirs('results', exist_ok=True)
    
    # Lưu classification map
    np.save('results/classification_map.npy', classification_map)
    print("✅ Đã lưu: results/classification_map.npy")
    
    # Lưu confidence map  
    np.save('results/confidence_map.npy', confidence_map)
    print("✅ Đã lưu: results/confidence_map.npy")
    
    # Lưu thống kê
    stats = {
        'classification_stats': {},
        'confidence_stats': {
            'mean': float(confidence_map.mean()),
            'std': float(confidence_map.std()),
            'min': float(confidence_map.min()),
            'max': float(confidence_map.max())
        },
        'metadata': metadata
    }
    
    # Thống kê từng class
    unique_classes, class_counts = np.unique(classification_map, return_counts=True)
    total_pixels = classification_map.size
    
    for class_idx, count in zip(unique_classes, class_counts):
        if class_idx < len(metadata['class_names']):
            class_name = metadata['class_names'][class_idx]
            stats['classification_stats'][class_name] = {
                'pixel_count': int(count),
                'percentage': float((count / total_pixels) * 100)
            }
    
    # Lưu stats
    with open('results/classification_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    print("✅ Đã lưu: results/classification_stats.pkl")

def main():
    """
    Hàm chính thực hiện toàn bộ quy trình Bước 7
    """
    start_time = time.time()
    
    print("🎯 Bắt đầu tạo bản đồ phân loại dinh dưỡng N2...")
    
    # Bước 7.1: Tải dữ liệu HSI
    hsi_data, gt_array = load_full_hsi_data()
    if hsi_data is None:
        return
    
    # Bước 7.2: Tải mô hình
    model, metadata = load_trained_model_and_metadata()
    if model is None:
        return
    
    # Bước 7.3: Chuẩn hóa dữ liệu
    hsi_normalized = normalize_hsi_data(hsi_data, metadata)
    
    # Bước 7.6: Tạo bản đồ phân loại (tối ưu hóa cho demo)
    classification_map, confidence_map = create_classification_map(
        hsi_normalized, model, metadata, batch_size=512, stride=3, max_pixels=500000)
    
    # Bước 7.7: Trực quan hóa
    visualize_classification_map(classification_map, confidence_map, metadata, gt_array)
    
    # Bước 7.8: Phân tích chi tiết
    create_detailed_analysis_plots(classification_map, confidence_map, metadata, gt_array)
    
    # Bước 7.9: Lưu kết quả
    save_classification_results(classification_map, confidence_map, metadata)
    
    # Tổng kết
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH BƯỚC 7")
    print("=" * 60)
    print("✅ Bản đồ phân loại dinh dưỡng N2 đã được tạo")
    print("✅ Trực quan hóa hoàn thành")
    print("✅ Phân tích chi tiết đã được thực hiện")
    print("✅ Kết quả đã được lưu vào thư mục results/")
    print(f"⏱️ Tổng thời gian: {total_time/60:.1f} phút")
    print("\n🌱 BẢN ĐỒ DINH DƯỠNG N2 CÂY CÀ TÍM HOÀN THÀNH! 🗺️")
    print("=" * 60)

if __name__ == "__main__":
    main()
