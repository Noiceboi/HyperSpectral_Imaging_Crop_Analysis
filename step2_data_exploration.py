"""
Bước 2: Tải và Khám phá Dữ liệu HSI Eggplant
Mục tiêu: Hiểu rõ cấu trúc và đặc điểm của dữ liệu trước khi xử lý
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import spectral.io.envi as envi
import os

# Cấu hình matplotlib để hiển thị đồ họa
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_hyperspectral_data():
    """
    Tải dữ liệu HSI và Ground Truth từ file ENVI
    """
    print("=" * 60)
    print("🔍 BƯỚC 2: KHÁM PHÁ DỮ LIỆU HYPERSPECTRAL EGGPLANT")
    print("=" * 60)
    
    # Đường dẫn đến các file dữ liệu
    data_path = r"d:\HyperSpectral_Imaging_Crop_Analysis\Eggplant_Crop\Eggplant_Crop"
    
    # File paths
    hsi_header = os.path.join(data_path, "Eggplant_Reflectance_Data.hdr")
    hsi_data = os.path.join(data_path, "Eggplant_Reflectance_Data")
    
    gt_header = os.path.join(data_path, "Eggplant_N2_Concentration_GT.hdr")
    gt_data = os.path.join(data_path, "Eggplant_N2_Concentration_GT")
    
    print("📁 Đường dẫn dữ liệu:")
    print(f"   HSI Header: {hsi_header}")
    print(f"   HSI Data: {hsi_data}")
    print(f"   GT Header: {gt_header}")
    print(f"   GT Data: {gt_data}")
    
    # Kiểm tra sự tồn tại của các file
    files_exist = True
    for file_path in [hsi_header, hsi_data, gt_header, gt_data]:
        if not os.path.exists(file_path):
            print(f"❌ File không tồn tại: {file_path}")
            files_exist = False
        else:
            print(f"✅ File tồn tại: {os.path.basename(file_path)}")
    
    if not files_exist:
        print("❌ Một số file không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
        return None, None
    
    try:
        print("\n🔄 Đang tải dữ liệu HSI...")
        # Tải dữ liệu HSI Reflectance
        hsi_img = envi.open(hsi_header, hsi_data)
        hsi_array = hsi_img.load()
        
        print("🔄 Đang tải Ground Truth...")
        # Tải Ground Truth
        gt_img = envi.open(gt_header, gt_data)
        gt_array = gt_img.load()
        
        print("✅ Tải dữ liệu thành công!")
        return hsi_array, gt_array, hsi_img, gt_img
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        return None, None, None, None

def analyze_data_structure(hsi_array, gt_array, hsi_img, gt_img):
    """
    Phân tích cấu trúc và thông tin chi tiết của dữ liệu
    """
    print("\n" + "=" * 60)
    print("📊 PHÂN TÍCH CẤU TRÚC DỮ LIỆU")
    print("=" * 60)
    
    # Thông tin HSI data
    print("🌈 HYPERSPECTRAL REFLECTANCE DATA:")
    print(f"   📐 Kích thước: {hsi_array.shape}")
    print(f"   📊 Kiểu dữ liệu: {hsi_array.dtype}")
    print(f"   📈 Giá trị min: {np.min(hsi_array):.4f}")
    print(f"   📈 Giá trị max: {np.max(hsi_array):.4f}")
    print(f"   📈 Giá trị trung bình: {np.mean(hsi_array):.4f}")
    
    # Thông tin về các dải phổ
    if hasattr(hsi_img, 'metadata') and 'wavelength' in hsi_img.metadata:
        wavelengths = [float(w) for w in hsi_img.metadata['wavelength']]
        print(f"   🌊 Số dải phổ: {len(wavelengths)}")
        print(f"   🌊 Dải bước sóng: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
    
    # Thông tin Ground Truth
    print("\n🎯 GROUND TRUTH DATA:")
    print(f"   📐 Kích thước: {gt_array.shape}")
    print(f"   📊 Kiểu dữ liệu: {gt_array.dtype}")
    
    # Xử lý shape của GT array - loại bỏ dimension cuối nếu là 1
    if len(gt_array.shape) == 3 and gt_array.shape[2] == 1:
        gt_array = gt_array[:, :, 0]
        print(f"   📐 Kích thước sau khi xử lý: {gt_array.shape}")
    
    # Chuyển đổi sang int để đảm bảo indices chính xác
    gt_array_int = gt_array.astype(int)
    unique_values = np.unique(gt_array_int)
    print(f"   🏷️ Các giá trị nhãn: {unique_values}")
    
    # Thống kê các lớp
    unique_labels, counts = np.unique(gt_array_int, return_counts=True)
    print(f"\n📈 THỐNG KÊ CÁC LỚP:")
    class_names = ['Unclassified', 'Low N2', 'Medium N2', 'High N2']
    total_pixels = gt_array.shape[0] * gt_array.shape[1]
    
    for label, count in zip(unique_labels, counts):
        if int(label) < len(class_names):
            percentage = (count / total_pixels) * 100
            print(f"   {class_names[int(label)]:12} (Lớp {int(label)}): {count:8,} pixels ({percentage:5.1f}%)")
    
    return wavelengths if 'wavelengths' in locals() else None, gt_array_int

def visualize_experimental_design():
    """
    Trực quan hóa thiết kế thí nghiệm dựa trên mô tả từ paper Munipalle & Nidamanuri (2024)
    """
    print("\n" + "=" * 60)
    print("🔬 THIẾT KẾ THÍ NGHIỆM VÀ PHƯƠNG PHÁP")
    print("=" * 60)
    
    # Tạo figure với 2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === Subplot 1: Experimental Layout ===
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 15)
    ax1.set_aspect('equal')
    ax1.set_title('🌾 Experimental Plot Layout\n(University of Agricultural Sciences, Bengaluru)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Vẽ main plot (12m x 18m)
    main_plot = patches.Rectangle((2, 2), 18, 12, linewidth=3, edgecolor='black', 
                                  facecolor='lightgray', alpha=0.3)
    ax1.add_patch(main_plot)
    ax1.text(11, 0.5, '18m', ha='center', fontsize=12, fontweight='bold')
    ax1.text(0.5, 8, '12m', ha='center', va='center', rotation=90, fontsize=12, fontweight='bold')
    
    # Vẽ 3 subplots (6m x 12m each)
    colors = ['#FFB6C1', '#98FB98', '#87CEEB']  # Light colors for Low, Medium, High N
    n_levels = ['Low N2\n(25 kg N/ha)', 'Medium N2\n(50 kg N/ha)', 'High N2\n(75 kg N/ha)']
    
    for i, (color, n_level) in enumerate(zip(colors, n_levels)):
        subplot = patches.Rectangle((2 + i*6, 2), 6, 12, linewidth=2, 
                                   edgecolor='darkblue', facecolor=color, alpha=0.7)
        ax1.add_patch(subplot)
        ax1.text(5 + i*6, 8, n_level, ha='center', va='center', fontsize=10, 
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax1.text(5 + i*6, 1, '6m × 12m', ha='center', fontsize=9, style='italic')
    
    # Thêm thông tin thêm
    ax1.text(11, 15.5, 'Drip Irrigation System', ha='center', fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    ax1.text(11, 14.7, '4 Replications per Treatment', ha='center', fontsize=10, style='italic')
    
    # Additional fertilizer info
    fertilizer_text = """Fertilizer Application:
• P: 41.5 kg/ha (blanket)
• K: 16.6 kg/ha (blanket)
• N: Variable by treatment"""
    ax1.text(21, 8, fertilizer_text, ha='left', va='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
    
    ax1.set_xlabel('Distance (meters)', fontsize=12)
    ax1.set_ylabel('Distance (meters)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # === Subplot 2: Nitrogen Treatment Levels ===
    ax2.set_title('💧 Nitrogen Treatment Levels\n(Eggplant Crop Specifications)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Data for bar chart
    treatments = ['Low N2', 'Medium N2', 'High N2']
    n_rates = [25, 50, 75]  # kg N/ha for eggplant
    colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax2.bar(treatments, n_rates, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, rate in zip(bars, n_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate} kg/ha', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_ylabel('Nitrogen Application Rate (kg N/ha)', fontsize=12)
    ax2.set_xlabel('Treatment Levels', fontsize=12)
    ax2.set_ylim(0, 85)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add reference line for medium level
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(1, 52, 'Regional Standard\n(50 kg N/ha)', ha='center', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add experimental period info
    experiment_info = """Experimental Period:
February - June 2022

Crop: Eggplant (Solanum melongena)
Location: Bengaluru, India
Field Size: 12m × 18m plots
Irrigation: Drip system"""
    
    ax2.text(0.02, 0.98, experiment_info, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Experimental design visualization completed!")
    print("\n📋 KEY EXPERIMENTAL DETAILS:")
    print("   🌱 Crop: Eggplant (Solanum melongena)")
    print("   📍 Location: University of Agricultural Sciences, Bengaluru, India")
    print("   📅 Period: February - June 2022")
    print("   📏 Plot Size: 12m × 18m (subdivided into 3 subplots of 6m × 12m)")
    print("   🔄 Replications: 4 per treatment")
    print("   💧 Irrigation: Drip irrigation system")
    print("   🧪 N Treatments: Low (25 kg/ha), Medium (50 kg/ha), High (75 kg/ha)")
    print("   ⚗️  Other Nutrients: P (41.5 kg/ha), K (16.6 kg/ha) - blanket application")
    
def visualize_data(hsi_array, gt_array, wavelengths=None):
    """
    Trực quan hóa dữ liệu để hiểu rõ hơn
    """
    print("\n" + "=" * 60)
    print("📊 TRỰC QUAN HÓA DỮ LIỆU")
    print("=" * 60)
    
    # Xử lý shape của GT array nếu cần
    if len(gt_array.shape) == 3 and gt_array.shape[2] == 1:
        gt_array = gt_array[:, :, 0]
    
    # Chuyển sang int
    gt_array = gt_array.astype(int)
    
    # Tạo figure với nhiều subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Khám phá Dữ liệu Hyperspectral Eggplant N2 Classification', fontsize=16)
    
    # 1. Hiển thị RGB composite (sử dụng 3 dải phổ làm RGB)
    if hsi_array.shape[2] >= 100:  # Đảm bảo có đủ dải phổ
        # Chọn các dải phổ tương ứng với RGB
        red_band = hsi_array[:, :, 50]    # ~630nm
        green_band = hsi_array[:, :, 30]  # ~550nm  
        blue_band = hsi_array[:, :, 10]   # ~450nm
        
        # Chuẩn hóa về [0,1] - sử dụng cách tính khác để tránh warning NumPy 2.0
        red_min, red_max = float(np.min(red_band)), float(np.max(red_band))
        green_min, green_max = float(np.min(green_band)), float(np.max(green_band))
        blue_min, blue_max = float(np.min(blue_band)), float(np.max(blue_band))
        
        red_norm = np.clip((red_band - red_min) / (red_max - red_min), 0, 1)
        green_norm = np.clip((green_band - green_min) / (green_max - green_min), 0, 1)
        blue_norm = np.clip((blue_band - blue_min) / (blue_max - blue_min), 0, 1)
        
        rgb_composite = np.dstack([red_norm, green_norm, blue_norm])
        
        axes[0, 0].imshow(rgb_composite)
        axes[0, 0].set_title('RGB Composite Image')
        axes[0, 0].axis('off')
    
    # 2. Hiển thị Ground Truth với màu sắc phù hợp
    colors = ['black', 'red', 'green', 'blue']  # Unclassified, Low, Medium, High
    gt_colored = np.zeros((gt_array.shape[0], gt_array.shape[1], 3))
    
    for i, color in enumerate(colors):
        mask = (gt_array == i)
        if color == 'black':
            gt_colored[mask] = [0, 0, 0]
        elif color == 'red':
            gt_colored[mask] = [1, 0, 0]
        elif color == 'green':
            gt_colored[mask] = [0, 1, 0]
        elif color == 'blue':
            gt_colored[mask] = [0, 0, 1]
    
    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title('Ground Truth N2 Classification\n(Red: Low, Green: Medium, Blue: High)')
    axes[0, 1].axis('off')
    
    # 3. Histogram của Ground Truth
    unique_labels, counts = np.unique(gt_array, return_counts=True)
    class_names = ['Unclassified', 'Low N2', 'Medium N2', 'High N2']
    colors_bar = ['black', 'red', 'green', 'blue']
    
    bars = axes[0, 2].bar([class_names[i] for i in unique_labels], 
                         counts, color=[colors_bar[i] for i in unique_labels])
    axes[0, 2].set_title('Distribution of N2 Classes')
    axes[0, 2].set_ylabel('Number of Pixels')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Thêm số lượng lên các cột
    for i, (bar, count) in enumerate(zip(bars, counts)):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                       f'{count:,}', ha='center', va='bottom')
    
    # 4. Spectral signature trung bình của mỗi lớp
    if wavelengths is not None:
        class_names_short = ['Low N2', 'Medium N2', 'High N2']
        colors_line = ['red', 'green', 'blue']
        
        for class_idx in range(1, 4):  # Bỏ qua Unclassified
            mask = (gt_array == class_idx)
            if np.sum(mask) > 0:
                # Lấy tất cả pixel thuộc lớp này - sử dụng fancy indexing cho spectral array
                mask_indices = np.where(mask)
                class_spectra_list = []
                
                # Giới hạn số pixel để tránh quá tải (lấy mẫu 1000 pixels)
                sample_size = min(1000, len(mask_indices[0]))
                sample_indices = np.random.choice(len(mask_indices[0]), sample_size, replace=False)
                
                for idx in sample_indices:
                    i, j = mask_indices[0][idx], mask_indices[1][idx]
                    spectrum = np.array(hsi_array[i, j, :])  # Lấy spectrum của pixel (i,j)
                    # Đảm bảo spectrum là 1D array
                    if spectrum.ndim > 1:
                        spectrum = spectrum.flatten()
                    class_spectra_list.append(spectrum)
                
                # Chuyển thành numpy array và tính mean
                class_spectra = np.array(class_spectra_list)
                mean_spectrum = np.mean(class_spectra, axis=0)
                
                # Đảm bảo mean_spectrum là 1D
                if mean_spectrum.ndim > 1:
                    mean_spectrum = mean_spectrum.flatten()
                
                axes[1, 0].plot(wavelengths, mean_spectrum, 
                               color=colors_line[class_idx-1], 
                               label=class_names_short[class_idx-1], linewidth=2)
        
        axes[1, 0].set_xlabel('Wavelength (nm)')
        axes[1, 0].set_ylabel('Reflectance')
        axes[1, 0].set_title('Mean Spectral Signatures by N2 Class')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Hiển thị một dải phổ cụ thể (ví dụ band 50)
    if hsi_array.shape[2] > 50:
        band_50 = hsi_array[:, :, 50]
        im = axes[1, 1].imshow(band_50, cmap='viridis')
        axes[1, 1].set_title(f'Band 50 Reflectance\n(~{wavelengths[50]:.0f} nm)' if wavelengths else 'Band 50 Reflectance')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # 6. Statistics summary
    axes[1, 2].axis('off')
    stats_text = f"""DATA SUMMARY
    
HSI Shape: {hsi_array.shape}
GT Shape: {gt_array.shape}
Spectral Bands: {hsi_array.shape[2]}
Wavelength Range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm

Class Distribution:
• Unclassified: {counts[0]:,} pixels
• Low N2: {counts[1]:,} pixels  
• Medium N2: {counts[2]:,} pixels
• High N2: {counts[3]:,} pixels

Total Pixels: {gt_array.shape[0] * gt_array.shape[1]:,}
"""
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Trực quan hóa hoàn thành!")

def main():
    """
    Hàm chính thực hiện toàn bộ quy trình khám phá dữ liệu
    """
    # Bước 2.0: Visualize experimental design first
    print("🔬 Hiểu về thiết kế thí nghiệm...")
    visualize_experimental_design()
    
    # Bước 2.1: Tải dữ liệu
    hsi_array, gt_array, hsi_img, gt_img = load_hyperspectral_data()
    
    if hsi_array is None:
        print("❌ Không thể tải dữ liệu. Dừng chương trình.")
        return
    
    # Bước 2.2: Phân tích cấu trúc
    wavelengths, gt_array_processed = analyze_data_structure(hsi_array, gt_array, hsi_img, gt_img)
    
    # Bước 2.3: Trực quan hóa
    print("\n🎨 Bắt đầu trực quan hóa...")
    visualize_data(hsi_array, gt_array_processed, wavelengths)
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH BƯỚC 2: KHÁM PHÁ DỮ LIỆU")
    print("✅ Thiết kế thí nghiệm đã được mô tả")
    print("✅ Dữ liệu đã được tải và phân tích thành công!")
    print("📊 Các biểu đồ trực quan hóa đã được tạo.")
    print("➡️  Sẵn sàng cho Bước 3: Tiền xử lý dữ liệu")
    print("=" * 60)

if __name__ == "__main__":
    main()
