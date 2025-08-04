"""
Bước 3 & 4: Tiền xử lý Dữ liệu và Tạo Dataset Huấn luyện
Mục tiêu: Chuẩn bị dữ liệu sạch và tạo 3D patches cho Deep Learning
"""

import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import savgol_filter
import pickle
import time
from tqdm import tqdm

# Cấu hình
np.random.seed(42)  # Đảm bảo reproducibility

def load_data():
    """
    Tải dữ liệu HSI và Ground Truth (tái sử dụng từ Bước 2)
    """
    print("=" * 60)
    print("🔄 BƯỚC 3 & 4: TIỀN XỬ LÝ VÀ TạO DATASET")
    print("=" * 60)
    
    data_path = r"d:\HyperSpectral_Imaging_Crop_Analysis\Eggplant_Crop\Eggplant_Crop"
    
    # File paths
    hsi_header = os.path.join(data_path, "Eggplant_Reflectance_Data.hdr")
    hsi_data = os.path.join(data_path, "Eggplant_Reflectance_Data")
    gt_header = os.path.join(data_path, "Eggplant_N2_Concentration_GT.hdr")
    gt_data = os.path.join(data_path, "Eggplant_N2_Concentration_GT")
    
    try:
        print("🔄 Đang tải lại dữ liệu...")
        hsi_img = envi.open(hsi_header, hsi_data)
        hsi_array = hsi_img.load()
        
        gt_img = envi.open(gt_header, gt_data)
        gt_array = gt_img.load()
        
        # Xử lý GT array shape
        if len(gt_array.shape) == 3 and gt_array.shape[2] == 1:
            gt_array = gt_array[:, :, 0]
        
        gt_array = gt_array.astype(int)
        
        # Extract wavelength information for discriminative bands
        wavelengths = None
        if hasattr(hsi_img, 'metadata') and 'wavelength' in hsi_img.metadata:
            wavelengths = [float(w) for w in hsi_img.metadata['wavelength']]
            print(f"🌊 Wavelength range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
        
        print(f"✅ HSI Shape: {hsi_array.shape}")
        print(f"✅ GT Shape: {gt_array.shape}")
        
        return hsi_array, gt_array, hsi_img, wavelengths
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        return None, None, None, None

def get_discriminative_bands(wavelengths):
    """
    Xác định các bands phân biệt dựa trên phân tích từ bước 2
    Từ advanced discrimination analysis: [691.497986, 695.872009, 693.684998]
    """
    if wavelengths is None:
        return None, None
    
    # Discriminative wavelengths từ analysis
    target_wavelengths = [691.497986, 695.872009, 693.684998]
    wavelengths_array = np.array(wavelengths)
    
    # Tìm indices gần nhất cho các wavelengths quan trọng
    discriminative_indices = []
    discriminative_wavelengths = []
    
    for target_wl in target_wavelengths:
        closest_idx = np.argmin(np.abs(wavelengths_array - target_wl))
        discriminative_indices.append(closest_idx)
        discriminative_wavelengths.append(wavelengths_array[closest_idx])
    
    print(f"\n🎯 DISCRIMINATIVE BANDS MAPPING:")
    for i, (target, actual, idx) in enumerate(zip(target_wavelengths, discriminative_wavelengths, discriminative_indices)):
        print(f"   Band {i+1}: Target {target:.1f}nm → Actual {actual:.1f}nm (Index {idx})")
    
    return discriminative_indices, discriminative_wavelengths

def step3_preprocessing(hsi_array, apply_smoothing=False):
    """
    Bước 3: Tiền xử lý dữ liệu HSI
    """
    print("\n" + "=" * 60)
    print("🧹 BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    
    print(f"📊 Dữ liệu gốc - Min: {np.min(hsi_array):.4f}, Max: {np.max(hsi_array):.4f}")
    
    # 3.1: Savitzky-Golay Smoothing (tùy chọn)
    if apply_smoothing:
        print("🔄 Áp dụng Savitzky-Golay smoothing...")
        hsi_smoothed = np.zeros_like(hsi_array)
        
        for i in tqdm(range(hsi_array.shape[0]), desc="Smoothing rows"):
            for j in range(hsi_array.shape[1]):
                spectrum = hsi_array[i, j, :]
                # Áp dụng SG filter với window=5, polynomial=2
                try:
                    smoothed_spectrum = savgol_filter(spectrum, window_length=5, polyorder=2)
                    hsi_smoothed[i, j, :] = smoothed_spectrum
                except:
                    hsi_smoothed[i, j, :] = spectrum  # Giữ nguyên nếu lỗi
        
        hsi_array = hsi_smoothed
        print("✅ Smoothing hoàn thành!")
    else:
        print("⏭️ Bỏ qua Savitzky-Golay smoothing (dữ liệu đã được lọc)")
    
    # 3.2: Chuẩn hóa dữ liệu về [0, 1]
    print("🔄 Chuẩn hóa dữ liệu về [0, 1]...")
    
    # Tìm min/max toàn cục
    global_min = float(np.min(hsi_array))
    global_max = float(np.max(hsi_array))
    
    print(f"   Global Min: {global_min:.4f}")
    print(f"   Global Max: {global_max:.4f}")
    
    # Chuẩn hóa
    hsi_normalized = (hsi_array - global_min) / (global_max - global_min)
    hsi_normalized = np.clip(hsi_normalized, 0, 1)
    
    print(f"✅ Sau chuẩn hóa - Min: {np.min(hsi_normalized):.4f}, Max: {np.max(hsi_normalized):.4f}")
    
    return hsi_normalized, global_min, global_max

def step3_label_processing(gt_array):
    """
    Bước 3: Xử lý nhãn - chuyển từ (1,2,3) thành (0,1,2)
    """
    print("\n🏷️ XỬ LÝ NHÃN:")
    
    # Hiển thị phân bố gốc
    unique_labels, counts = np.unique(gt_array, return_counts=True)
    print("📊 Phân bố nhãn gốc:")
    class_names_old = ['Unclassified', 'Low N2', 'Medium N2', 'High N2']
    for label, count in zip(unique_labels, counts):
        if label < len(class_names_old):
            print(f"   {class_names_old[label]} (Lớp {label}): {count:,} pixels")
    
    # Tạo nhãn mới cho Deep Learning
    gt_processed = gt_array.copy()
    
    # Chuyển đổi: 1→1, 2→2, 3→3, giữ nguyên 0 (Unclassified)
    # Sau đó sẽ trừ 1 cho các class training để có 0,1,2
    # Không cần ánh xạ ở đây
    
    # Hiển thị kết quả
    print("\n📋 Ánh xạ nhãn cho Deep Learning:")
    print("   Lớp 1 (Low N2)    → Nhãn 0 (sau khi trừ 1)")
    print("   Lớp 2 (Medium N2) → Nhãn 1 (sau khi trừ 1)") 
    print("   Lớp 3 (High N2)   → Nhãn 2 (sau khi trừ 1)")
    print("   Lớp 0 (Unclassified) → Loại bỏ khỏi training")
    
    # Thống kê nhãn mới (chỉ các class có thể train được)
    training_mask = gt_processed > 0  # Loại bỏ Unclassified (0)
    
    # Hiển thị phân bố sau khi tạo training mask
    training_labels = gt_processed[training_mask] - 1  # Chuyển về 0,1,2
    unique_mapped, counts_mapped = np.unique(training_labels, return_counts=True)
    print(f"\n📈 Phân bố nhãn training (sau khi trừ 1):")
    class_names_mapped = {0: 'Low N2', 1: 'Medium N2', 2: 'High N2'}
    for label, count in zip(unique_mapped, counts_mapped):
        if label in class_names_mapped:
            print(f"   {class_names_mapped[label]} (Nhãn {label}): {count:,} pixels")
    
    total_training = np.sum(counts_mapped)
    print(f"📊 Tổng số pixels training: {total_training:,}")
    
    return gt_processed, training_mask

def step4_extract_patches(hsi_normalized, gt_processed, training_mask, patch_size=9, max_samples_per_class=50000):
    """
    Bước 4: Trích xuất 3D patches từ dữ liệu HSI
    """
    print("\n" + "=" * 60)
    print("🎯 BƯỚC 4: TRÍCH XUẤT 3D PATCHES")
    print("=" * 60)
    
    print(f"📐 Patch size: {patch_size}x{patch_size}x{hsi_normalized.shape[2]}")
    print(f"🎲 Max samples per class: {max_samples_per_class:,}")
    
    height, width, n_bands = hsi_normalized.shape
    half_patch = patch_size // 2
    
    patches = []
    labels = []
    
    # Tạo class mapping cho balanced sampling
    class_pixels = {}
    for class_id in range(3):  # 0: Low, 1: Medium, 2: High
        # Sử dụng nhãn gốc (1,2,3) để tìm pixels, sau đó ánh xạ thành (0,1,2)
        original_class_id = class_id + 1  # 0→1, 1→2, 2→3
        mask = (gt_processed == original_class_id) & training_mask
        indices = np.where(mask)
        class_pixels[class_id] = list(zip(indices[0], indices[1]))
        print(f"🔍 Class {class_id} ({['Low', 'Medium', 'High'][class_id]} N2): {len(class_pixels[class_id]):,} pixels")
    
    # Balanced sampling từ mỗi class
    for class_id in range(3):
        pixels = class_pixels[class_id]
        
        # Random sample nếu có quá nhiều pixels
        if len(pixels) > max_samples_per_class:
            pixels = np.random.choice(len(pixels), max_samples_per_class, replace=False)
            pixels = [class_pixels[class_id][i] for i in pixels]
            print(f"   📊 Downsampled to {len(pixels):,} pixels")
        
        class_patches = []
        valid_patches = 0
        
        for i, j in tqdm(pixels, desc=f"Extracting Class {class_id} patches"):
            # Kiểm tra boundaries
            if (i >= half_patch and i < height - half_patch and 
                j >= half_patch and j < width - half_patch):
                
                # Trích xuất patch 3D
                patch = hsi_normalized[i-half_patch:i+half_patch+1, 
                                     j-half_patch:j+half_patch+1, :]
                
                # Kiểm tra patch shape
                if patch.shape == (patch_size, patch_size, n_bands):
                    class_patches.append(patch)
                    valid_patches += 1
        
        # Thêm vào dataset
        patches.extend(class_patches)
        labels.extend([class_id] * len(class_patches))
        
        print(f"✅ Class {class_id}: {len(class_patches):,} patches extracted")
    
    # Convert to numpy arrays
    patches = np.array(patches, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   📐 Patches shape: {patches.shape}")
    print(f"   🏷️ Labels shape: {labels.shape}")
    print(f"   💾 Memory usage: {patches.nbytes / (1024**2):.1f} MB")
    
    # Thống kê phân bố final
    unique_final, counts_final = np.unique(labels, return_counts=True)
    for label, count in zip(unique_final, counts_final):
        print(f"   Class {label}: {count:,} patches ({count/len(labels)*100:.1f}%)")
    
    return patches, labels

def step4_train_val_test_split(patches, labels, test_size=0.15, val_size=0.15):
    """
    Bước 4: Chia dataset thành train/validation/test
    """
    print("\n" + "=" * 60)
    print("🔄 CHIA DATASET TRAIN/VAL/TEST")
    print("=" * 60)
    
    # Tính toán tỷ lệ
    train_size = 1.0 - test_size - val_size
    print(f"📊 Tỷ lệ chia: Train {train_size:.0%} / Val {val_size:.0%} / Test {test_size:.0%}")
    
    # Chia train vs (val+test) trước
    X_train, X_temp, y_train, y_temp = train_test_split(
        patches, labels, 
        test_size=(test_size + val_size),
        stratify=labels,
        random_state=42
    )
    
    # Chia val vs test
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - relative_val_size),
        stratify=y_temp,
        random_state=42
    )
    
    # Thống kê kết quả
    print(f"\n📈 KẾT QUẢ CHIA DATASET:")
    print(f"   🏋️ Training:   {X_train.shape[0]:,} samples ({X_train.shape[0]/len(patches)*100:.1f}%)")
    print(f"   ✅ Validation: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(patches)*100:.1f}%)")
    print(f"   🧪 Testing:    {X_test.shape[0]:,} samples ({X_test.shape[0]/len(patches)*100:.1f}%)")
    
    # Kiểm tra stratification
    print(f"\n🎯 KIỂM TRA STRATIFICATION:")
    datasets = [('Train', y_train), ('Val', y_val), ('Test', y_test)]
    
    for name, y_data in datasets:
        unique_labels, counts = np.unique(y_data, return_counts=True)
        percentages = counts / len(y_data) * 100
        print(f"   {name:10}: ", end="")
        for label, pct in zip(unique_labels, percentages):
            print(f"Class {label}: {pct:.1f}%  ", end="")
        print()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_dataset(X_train, X_val, X_test, y_train, y_val, y_test, 
                 global_min, global_max, discriminative_indices=None, discriminative_wavelengths=None, 
                 save_dir="processed_data"):
    """
    Lưu dataset đã xử lý
    """
    print("\n" + "=" * 60)
    print("💾 LƯU DATASET")
    print("=" * 60)
    
    # Tạo thư mục nếu chưa có, xóa dữ liệu cũ nếu có
    if os.path.exists(save_dir):
        print(f"🗑️ Xóa dữ liệu cũ trong thư mục: {save_dir}")
        import shutil
        shutil.rmtree(save_dir)
    
    os.makedirs(save_dir)
    print(f"📁 Tạo thư mục mới: {save_dir}")
    
    # Lưu các file
    datasets = {
        'X_train.npy': X_train,
        'X_val.npy': X_val, 
        'X_test.npy': X_test,
        'y_train.npy': y_train,
        'y_val.npy': y_val,
        'y_test.npy': y_test
    }
    
    total_size = 0
    for filename, data in datasets.items():
        filepath = os.path.join(save_dir, filename)
        np.save(filepath, data)
        size_mb = os.path.getsize(filepath) / (1024**2)
        total_size += size_mb
        print(f"✅ Saved {filename:12} - {data.shape} - {size_mb:.1f} MB")
    
    # Lưu metadata với discriminative bands info
    metadata = {
        'patch_size': X_train.shape[1],  # Assuming square patches
        'n_bands': X_train.shape[3],
        'n_classes': len(np.unique(y_train)),
        'global_min': global_min,
        'global_max': global_max,
        'class_names': ['Low N2', 'Medium N2', 'High N2'],
        'total_samples': len(X_train) + len(X_val) + len(X_test),
        'discriminative_bands': {
            'indices': discriminative_indices,
            'wavelengths': discriminative_wavelengths,
            'target_wavelengths': [691.497986, 695.872009, 693.684998]
        } if discriminative_indices is not None else None
    }
    
    metadata_path = os.path.join(save_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Saved metadata.pkl")
    print(f"💾 Total dataset size: {total_size:.1f} MB")
    print(f"📁 Saved in: {os.path.abspath(save_dir)}")

def main():
    """
    Hàm chính thực hiện toàn bộ quy trình Bước 3 & 4
    """
    start_time = time.time()
    
    # Load data
    hsi_array, gt_array, hsi_img, wavelengths = load_data()
    if hsi_array is None:
        return
    
    # Extract discriminative bands information
    discriminative_indices, discriminative_wavelengths = get_discriminative_bands(wavelengths)
    
    # Bước 3.1: Preprocessing HSI
    hsi_normalized, global_min, global_max = step3_preprocessing(
        hsi_array, apply_smoothing=False
    )
    
    # Bước 3.2: Label processing  
    gt_processed, training_mask = step3_label_processing(gt_array)
    
    # Bước 4.1: Extract 3D patches
    patches, labels = step4_extract_patches(
        hsi_normalized, gt_processed, training_mask, 
        patch_size=9, max_samples_per_class=50000
    )
    
    # Bước 4.2: Train/Val/Test split
    X_train, X_val, X_test, y_train, y_val, y_test = step4_train_val_test_split(
        patches, labels, test_size=0.15, val_size=0.15
    )
    
    # Bước 4.3: Save dataset
    save_dataset(X_train, X_val, X_test, y_train, y_val, y_test,
                 global_min, global_max, discriminative_indices, discriminative_wavelengths)
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH BƯỚC 3 & 4")
    print("=" * 60)
    print("✅ Tiền xử lý dữ liệu hoàn thành")
    print("✅ 3D patches đã được tạo")
    print("✅ Dataset đã được chia train/val/test")
    print("✅ Dữ liệu đã được lưu")
    print(f"⏱️ Thời gian thực hiện: {elapsed_time/60:.1f} phút")
    print("➡️ Sẵn sàng cho Bước 5: Xây dựng 3D-ResNet")
    print("=" * 60)

if __name__ == "__main__":
    main()
