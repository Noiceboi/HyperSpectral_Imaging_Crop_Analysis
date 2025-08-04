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

def analyze_spectral_differences(hsi_array, gt_array, wavelengths=None):
    """
    Phân tích chi tiết sự khác biệt spectral giữa các class N2
    Tìm hiểu tại sao Low N2 và High N2 có phổ tương tự nhau
    """
    print("\n" + "=" * 60)
    print("🔬 PHÂN TÍCH CHI TIẾT SỰ KHÁC BIỆT SPECTRAL")
    print("=" * 60)
    
    if wavelengths is None:
        print("❌ Không có thông tin wavelength để phân tích")
        return
    
    # Xử lý GT array
    if len(gt_array.shape) == 3 and gt_array.shape[2] == 1:
        gt_array = gt_array[:, :, 0]
    gt_array = gt_array.astype(int)
    
    # Tạo figure với 3 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔬 Deep Spectral Analysis: Why Low N2 ≈ High N2?', fontsize=16, fontweight='bold')
    
    # Thu thập spectral data cho từng class
    class_spectra = {}
    class_names = ['Low N2', 'Medium N2', 'High N2']
    colors = ['red', 'green', 'blue']
    
    print("� Thu thập spectral signatures...")
    for class_idx in range(1, 4):
        mask = (gt_array == class_idx)
        if np.sum(mask) > 0:
            mask_indices = np.where(mask)
            sample_size = min(2000, len(mask_indices[0]))  # Lấy nhiều sample hơn
            sample_indices = np.random.choice(len(mask_indices[0]), sample_size, replace=False)
            
            class_spectra_list = []
            for idx in sample_indices:
                i, j = mask_indices[0][idx], mask_indices[1][idx]
                spectrum = np.array(hsi_array[i, j, :]).flatten()
                class_spectra_list.append(spectrum)
            
            class_spectra[class_idx] = np.array(class_spectra_list)
            print(f"   ✅ {class_names[class_idx-1]}: {len(class_spectra_list)} spectra")
    
    # === Subplot 1: Mean Spectra với Standard Deviation ===
    ax1 = axes[0, 0]
    for class_idx in range(1, 4):
        if class_idx in class_spectra:
            spectra = class_spectra[class_idx]
            mean_spectrum = np.mean(spectra, axis=0)
            std_spectrum = np.std(spectra, axis=0)
            
            ax1.plot(wavelengths, mean_spectrum, color=colors[class_idx-1], 
                    label=f'{class_names[class_idx-1]} (mean)', linewidth=2)
            ax1.fill_between(wavelengths, mean_spectrum - std_spectrum, 
                           mean_spectrum + std_spectrum, 
                           color=colors[class_idx-1], alpha=0.2)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title('Mean ± Std Spectral Signatures')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Subplot 2: Difference Analysis ===
    ax2 = axes[0, 1]
    if 1 in class_spectra and 2 in class_spectra and 3 in class_spectra:
        mean_low = np.mean(class_spectra[1], axis=0)
        mean_medium = np.mean(class_spectra[2], axis=0)
        mean_high = np.mean(class_spectra[3], axis=0)
        
        # Tính các differences
        diff_low_medium = mean_low - mean_medium
        diff_high_medium = mean_high - mean_medium
        diff_low_high = mean_low - mean_high
        
        ax2.plot(wavelengths, diff_low_medium, 'r-', label='Low - Medium', linewidth=2)
        ax2.plot(wavelengths, diff_high_medium, 'b-', label='High - Medium', linewidth=2)
        ax2.plot(wavelengths, diff_low_high, 'purple', label='Low - High', linewidth=2, linestyle='--')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Reflectance Difference')
        ax2.set_title('Spectral Differences Between Classes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Highlight important regions
        ax2.axvspan(700, 800, alpha=0.1, color='red', label='Red Edge')
        ax2.axvspan(800, 900, alpha=0.1, color='orange', label='NIR')
    
    # === Subplot 3: Statistical Analysis ===
    ax3 = axes[1, 0]
    
    # Tính correlation matrix giữa các class means
    if len(class_spectra) == 3:
        means = np.array([np.mean(class_spectra[i], axis=0) for i in range(1, 4)])
        correlations = np.corrcoef(means)
        
        im = ax3.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(3))
        ax3.set_yticks(range(3))
        ax3.set_xticklabels(class_names)
        ax3.set_yticklabels(class_names)
        ax3.set_title('Spectral Correlation Matrix')
        
        # Thêm correlation values
        for i in range(3):
            for j in range(3):
                ax3.text(j, i, f'{correlations[i, j]:.3f}', 
                        ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=ax3, shrink=0.8)
    
    # === Subplot 4: Critical Wavelength Analysis ===
    ax4 = axes[1, 1]
    
    if len(class_spectra) >= 3:
        # Tìm wavelengths có sự khác biệt lớn nhất
        diff_abs = np.abs(diff_low_high)
        critical_indices = np.argsort(diff_abs)[-20:]  # Top 20 critical wavelengths
        critical_wavelengths = np.array(wavelengths)[critical_indices]
        critical_diffs = diff_abs[critical_indices]
        
        bars = ax4.bar(range(len(critical_wavelengths)), critical_diffs, 
                      color='orange', alpha=0.7)
        ax4.set_xlabel('Critical Wavelengths (nm)')
        ax4.set_ylabel('|Low N2 - High N2| Difference')
        ax4.set_title('Top 20 Discriminative Wavelengths')
        ax4.set_xticks(range(0, len(critical_wavelengths), 3))
        ax4.set_xticklabels([f'{w:.0f}' for w in critical_wavelengths[::3]], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Highlight max difference
        max_idx = np.argmax(critical_diffs)
        bars[max_idx].set_color('red')
        ax4.text(max_idx, critical_diffs[max_idx], 
                f'{critical_wavelengths[max_idx]:.0f}nm', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # === STATISTICAL ANALYSIS OUTPUT ===
    print("\n📊 PHÂN TÍCH THỐNG KÊ:")
    
    if len(class_spectra) >= 3:
        # Correlation analysis
        corr_low_medium = correlations[0, 1]
        corr_high_medium = correlations[2, 1]
        corr_low_high = correlations[0, 2]
        
        print(f"🔗 SPECTRAL CORRELATIONS:")
        print(f"   Low N2  ↔ Medium N2: {corr_low_medium:.4f}")
        print(f"   High N2 ↔ Medium N2: {corr_high_medium:.4f}")
        print(f"   Low N2  ↔ High N2:   {corr_low_high:.4f} ⚠️")
        
        # Variance analysis
        var_low = np.mean(np.var(class_spectra[1], axis=0))
        var_medium = np.mean(np.var(class_spectra[2], axis=0))
        var_high = np.mean(np.var(class_spectra[3], axis=0))
        
        print(f"\n� WITHIN-CLASS VARIANCE:")
        print(f"   Low N2:    {var_low:.6f}")
        print(f"   Medium N2: {var_medium:.6f}")
        print(f"   High N2:   {var_high:.6f}")
        
        # Critical wavelength analysis
        max_diff_idx = np.argmax(diff_abs)
        max_diff_wavelength = wavelengths[max_diff_idx]
        max_diff_value = diff_abs[max_diff_idx]
        
        print(f"\n🎯 MOST DISCRIMINATIVE WAVELENGTH:")
        print(f"   Wavelength: {max_diff_wavelength:.1f} nm")
        print(f"   |Low - High| difference: {max_diff_value:.6f}")
        
        # Spectral regions analysis
        visible = (np.array(wavelengths) >= 400) & (np.array(wavelengths) <= 700)
        red_edge = (np.array(wavelengths) >= 700) & (np.array(wavelengths) <= 800)
        nir = (np.array(wavelengths) >= 800) & (np.array(wavelengths) <= 1000)
        
        print(f"\n� SPECTRAL REGION ANALYSIS:")
        if np.any(visible):
            vis_diff = np.mean(np.abs(diff_low_high[visible]))
            print(f"   Visible (400-700nm):   {vis_diff:.6f}")
        if np.any(red_edge):
            red_diff = np.mean(np.abs(diff_low_high[red_edge]))
            print(f"   Red Edge (700-800nm):  {red_diff:.6f}")
        if np.any(nir):
            nir_diff = np.mean(np.abs(diff_low_high[nir]))
            print(f"   NIR (800-1000nm):      {nir_diff:.6f}")
    
    # === BIOLOGICAL INTERPRETATION ===
    print(f"\n🌱 SINH LÝ THỰC VẬT - GIẢI THÍCH:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if corr_low_high > 0.95:
        print(f"⚠️  HIGH CORRELATION (r={corr_low_high:.3f}) giữa Low và High N2 cho thấy:")
        print(f"   • Cả thiếu và thừa N2 đều gây STRESS tương tự")
        print(f"   • Leaf structure changes tương đồng")
        print(f"   • Chlorophyll degradation patterns giống nhau")
        
    if var_medium < var_low and var_medium < var_high:
        print(f"📊 MEDIUM N2 có variance thấp nhất → Trạng thái ổn định nhất")
        print(f"   • Optimal nutrition → uniform plant response")
        print(f"   • Low/High N2 → variable stress responses")
    
    print(f"\n💡 IMPLICATIONS FOR MACHINE LEARNING:")
    print(f"   🔸 Binary classification (Optimal vs Non-optimal) có thể hiệu quả hơn")
    print(f"   🔸 Spatial features (3D-CNN) quan trọng để phân biệt Low/High")
    print(f"   🔸 Ensemble methods có thể cải thiện discrimination")
    print(f"   🔸 Feature selection tập trung vào critical wavelengths")
    
    print("✅ Phân tích spectral differences hoàn thành!")
    
def visualize_data(hsi_array, gt_array, wavelengths=None):
    """
    Phân tích chi tiết sự khác biệt spectral giữa các class N2
    Tìm hiểu tại sao Low N2 và High N2 có phổ tương tự nhau
    """
    print("\n" + "=" * 60)
    print("🔬 PHÂN TÍCH CHI TIẾT SỰ KHÁC BIỆT SPECTRAL")
    print("=" * 60)
    
    if wavelengths is None:
        print("❌ Không có thông tin wavelength để phân tích")
        return
    
    # Xử lý GT array
    if len(gt_array.shape) == 3 and gt_array.shape[2] == 1:
        gt_array = gt_array[:, :, 0]
    gt_array = gt_array.astype(int)
    
    # Tạo figure với 3 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🔬 Deep Spectral Analysis: Why Low N2 ≈ High N2?', fontsize=16, fontweight='bold')
    
    # Thu thập spectral data cho từng class
    class_spectra = {}
    class_names = ['Low N2', 'Medium N2', 'High N2']
    colors = ['red', 'green', 'blue']
    
    print("🔄 Thu thập spectral signatures...")
    for class_idx in range(1, 4):
        mask = (gt_array == class_idx)
        if np.sum(mask) > 0:
            mask_indices = np.where(mask)
            sample_size = min(2000, len(mask_indices[0]))  # Lấy nhiều sample hơn
            sample_indices = np.random.choice(len(mask_indices[0]), sample_size, replace=False)
            
            class_spectra_list = []
            for idx in sample_indices:
                i, j = mask_indices[0][idx], mask_indices[1][idx]
                spectrum = np.array(hsi_array[i, j, :]).flatten()
                class_spectra_list.append(spectrum)
            
            class_spectra[class_idx] = np.array(class_spectra_list)
            print(f"   ✅ {class_names[class_idx-1]}: {len(class_spectra_list)} spectra")
    
    # === Subplot 1: Mean Spectra với Standard Deviation ===
    ax1 = axes[0, 0]
    for class_idx in range(1, 4):
        if class_idx in class_spectra:
            spectra = class_spectra[class_idx]
            mean_spectrum = np.mean(spectra, axis=0)
            std_spectrum = np.std(spectra, axis=0)
            
            ax1.plot(wavelengths, mean_spectrum, color=colors[class_idx-1], 
                    label=f'{class_names[class_idx-1]} (mean)', linewidth=2)
            ax1.fill_between(wavelengths, mean_spectrum - std_spectrum, 
                           mean_spectrum + std_spectrum, 
                           color=colors[class_idx-1], alpha=0.2)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title('Mean ± Std Spectral Signatures')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Subplot 2: Difference Analysis ===
    ax2 = axes[0, 1]
    if 1 in class_spectra and 2 in class_spectra and 3 in class_spectra:
        mean_low = np.mean(class_spectra[1], axis=0)
        mean_medium = np.mean(class_spectra[2], axis=0)
        mean_high = np.mean(class_spectra[3], axis=0)
        
        # Tính các differences
        diff_low_medium = mean_low - mean_medium
        diff_high_medium = mean_high - mean_medium
        diff_low_high = mean_low - mean_high
        
        ax2.plot(wavelengths, diff_low_medium, 'r-', label='Low - Medium', linewidth=2)
        ax2.plot(wavelengths, diff_high_medium, 'b-', label='High - Medium', linewidth=2)
        ax2.plot(wavelengths, diff_low_high, 'purple', label='Low - High', linewidth=2, linestyle='--')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Reflectance Difference')
        ax2.set_title('Spectral Differences Between Classes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Highlight important regions
        ax2.axvspan(700, 800, alpha=0.1, color='red', label='Red Edge')
        ax2.axvspan(800, 900, alpha=0.1, color='orange', label='NIR')
    
    # === Subplot 3: Statistical Analysis ===
    ax3 = axes[1, 0]
    
    # Tính correlation matrix giữa các class means
    if len(class_spectra) == 3:
        means = np.array([np.mean(class_spectra[i], axis=0) for i in range(1, 4)])
        correlations = np.corrcoef(means)
        
        im = ax3.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(3))
        ax3.set_yticks(range(3))
        ax3.set_xticklabels(class_names)
        ax3.set_yticklabels(class_names)
        ax3.set_title('Spectral Correlation Matrix')
        
        # Thêm correlation values
        for i in range(3):
            for j in range(3):
                ax3.text(j, i, f'{correlations[i, j]:.3f}', 
                        ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im, ax=ax3, shrink=0.8)
    
    # === Subplot 4: Critical Wavelength Analysis ===
    ax4 = axes[1, 1]
    
    if len(class_spectra) >= 3:
        # Tìm wavelengths có sự khác biệt lớn nhất
        diff_abs = np.abs(diff_low_high)
        critical_indices = np.argsort(diff_abs)[-20:]  # Top 20 critical wavelengths
        critical_wavelengths = np.array(wavelengths)[critical_indices]
        critical_diffs = diff_abs[critical_indices]
        
        bars = ax4.bar(range(len(critical_wavelengths)), critical_diffs, 
                      color='orange', alpha=0.7)
        ax4.set_xlabel('Critical Wavelengths (nm)')
        ax4.set_ylabel('|Low N2 - High N2| Difference')
        ax4.set_title('Top 20 Discriminative Wavelengths')
        ax4.set_xticks(range(0, len(critical_wavelengths), 3))
        ax4.set_xticklabels([f'{w:.0f}' for w in critical_wavelengths[::3]], rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Highlight max difference
        max_idx = np.argmax(critical_diffs)
        bars[max_idx].set_color('red')
        ax4.text(max_idx, critical_diffs[max_idx], 
                f'{critical_wavelengths[max_idx]:.0f}nm', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # === STATISTICAL ANALYSIS OUTPUT ===
    print("\n📊 PHÂN TÍCH THỐNG KÊ:")
    
    if len(class_spectra) >= 3:
        # Correlation analysis
        corr_low_medium = correlations[0, 1]
        corr_high_medium = correlations[2, 1]
        corr_low_high = correlations[0, 2]
        
        print(f"🔗 SPECTRAL CORRELATIONS:")
        print(f"   Low N2  ↔ Medium N2: {corr_low_medium:.4f}")
        print(f"   High N2 ↔ Medium N2: {corr_high_medium:.4f}")
        print(f"   Low N2  ↔ High N2:   {corr_low_high:.4f} ⚠️")
        
        # Variance analysis
        var_low = np.mean(np.var(class_spectra[1], axis=0))
        var_medium = np.mean(np.var(class_spectra[2], axis=0))
        var_high = np.mean(np.var(class_spectra[3], axis=0))
        
        print(f"\n📊 WITHIN-CLASS VARIANCE:")
        print(f"   Low N2:    {var_low:.6f}")
        print(f"   Medium N2: {var_medium:.6f}")
        print(f"   High N2:   {var_high:.6f}")
        
        # Critical wavelength analysis
        max_diff_idx = np.argmax(diff_abs)
        max_diff_wavelength = wavelengths[max_diff_idx]
        max_diff_value = diff_abs[max_diff_idx]
        
        print(f"\n🎯 MOST DISCRIMINATIVE WAVELENGTH:")
        print(f"   Wavelength: {max_diff_wavelength:.1f} nm")
        print(f"   |Low - High| difference: {max_diff_value:.6f}")
        
        # Spectral regions analysis
        visible = (np.array(wavelengths) >= 400) & (np.array(wavelengths) <= 700)
        red_edge = (np.array(wavelengths) >= 700) & (np.array(wavelengths) <= 800)
        nir = (np.array(wavelengths) >= 800) & (np.array(wavelengths) <= 1000)
        
        print(f"\n🌈 SPECTRAL REGION ANALYSIS:")
        if np.any(visible):
            vis_diff = np.mean(np.abs(diff_low_high[visible]))
            print(f"   Visible (400-700nm):   {vis_diff:.6f}")
        if np.any(red_edge):
            red_diff = np.mean(np.abs(diff_low_high[red_edge]))
            print(f"   Red Edge (700-800nm):  {red_diff:.6f}")
        if np.any(nir):
            nir_diff = np.mean(np.abs(diff_low_high[nir]))
            print(f"   NIR (800-1000nm):      {nir_diff:.6f}")
    
    # === BIOLOGICAL INTERPRETATION ===
    print(f"\n🌱 SINH LÝ THỰC VẬT - GIẢI THÍCH:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if corr_low_high > 0.95:
        print(f"⚠️  HIGH CORRELATION (r={corr_low_high:.3f}) giữa Low và High N2 cho thấy:")
        print(f"   • Cả thiếu và thừa N2 đều gây STRESS tương tự")
        print(f"   • Leaf structure changes tương đồng")
        print(f"   • Chlorophyll degradation patterns giống nhau")
        
    if var_medium < var_low and var_medium < var_high:
        print(f"📊 MEDIUM N2 có variance thấp nhất → Trạng thái ổn định nhất")
        print(f"   • Optimal nutrition → uniform plant response")
        print(f"   • Low/High N2 → variable stress responses")
    
    print(f"\n💡 IMPLICATIONS FOR MACHINE LEARNING:")
    print(f"   🔸 Binary classification (Optimal vs Non-optimal) có thể hiệu quả hơn")
    print(f"   🔸 Spatial features (3D-CNN) quan trọng để phân biệt Low/High")
    print(f"   🔸 Ensemble methods có thể cải thiện discrimination")
    print(f"   🔸 Feature selection tập trung vào critical wavelengths")
    
    print("✅ Phân tích spectral differences hoàn thành!")
    
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
    
    # Bước 2.4: Phân tích chi tiết spectral differences
    print("\n🔬 Phân tích chi tiết sự khác biệt spectral...")
    separability_scores = analyze_spectral_differences(hsi_array, gt_array_processed, wavelengths)
    
    # Bước 2.5: Deep dive analysis - khám phá bí ẩn
    print("\n�️ Deep dive analysis - Giải mã bí ẩn spectral...")
    mystery_results = deep_dive_spectral_mystery(hsi_array, gt_array_processed, wavelengths)
    
    # Bước 2.6: Advanced discrimination analysis - Tìm cách phân biệt Low vs High N2
    print("\n🎯 Advanced discrimination analysis - Chiến lược phân biệt Low vs High N2...")
    discrimination_results = advanced_discrimination_analysis(hsi_array, gt_array_processed, wavelengths)
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH BƯỚC 2: KHÁM PHÁ DỮ LIỆU")
    print("✅ Thiết kế thí nghiệm đã được mô tả")
    print("✅ Dữ liệu đã được tải và phân tích thành công!")
    print("📊 Các biểu đồ trực quan hóa đã được tạo.")
    print("🔬 Phân tích chi tiết spectral differences đã hoàn thành.")
    print("➡️  Sẵn sàng cho Bước 3: Tiền xử lý dữ liệu")
    print("=" * 60)

def deep_dive_spectral_mystery(hsi_array, gt_array, wavelengths=None):
    """
    🔍 DEEP DIVE: Khám phá bí ẩn tại sao Low N2 và High N2 lại giống nhau đến thế
    Phân tích những "điểm mù" trong dữ liệu hyperspectral
    """
    print("\n" + "🔬" * 30)
    print("🕵️ DEEP DIVE: GIẢI MÃ BÍ ẨN SPECTRAL N2")
    print("🔬" * 30)
    
    if wavelengths is None:
        print("❌ Cần thông tin wavelength để phân tích sâu")
        return
    
    # Xử lý GT array
    if len(gt_array.shape) == 3 and gt_array.shape[2] == 1:
        gt_array = gt_array[:, :, 0]
    gt_array = gt_array.astype(int)
    
    # Thu thập dữ liệu với sample size lớn hơn
    class_spectra = {}
    class_names = ['Low N2', 'Medium N2', 'High N2']
    colors = ['red', 'green', 'blue']
    
    print("📊 Thu thập dữ liệu với sample size lớn...")
    for class_idx in range(1, 4):
        mask = (gt_array == class_idx)
        if np.sum(mask) > 0:
            mask_indices = np.where(mask)
            # Lấy tất cả pixel có thể
            total_pixels = len(mask_indices[0])
            sample_size = min(5000, total_pixels)  # Tăng sample size
            sample_indices = np.random.choice(total_pixels, sample_size, replace=False)
            
            class_spectra_list = []
            for idx in sample_indices:
                i, j = mask_indices[0][idx], mask_indices[1][idx]
                spectrum = np.array(hsi_array[i, j, :]).flatten()
                class_spectra_list.append(spectrum)
            
            class_spectra[class_idx] = np.array(class_spectra_list)
            print(f"   ✅ {class_names[class_idx-1]}: {len(class_spectra_list)} pixels")
    
    # === 1. PHÂN TÍCH CHI TIẾT CORRELATION ===
    print("\n🔍 1. PHÂN TÍCH CHI TIẾT CORRELATION:")
    
    # Tính correlation cho từng wavelength
    wavelength_correlations = []
    if 1 in class_spectra and 3 in class_spectra:
        for band_idx in range(len(wavelengths)):
            low_band = class_spectra[1][:, band_idx]
            high_band = class_spectra[3][:, band_idx]
            
            # Loại bỏ NaN và inf
            mask = np.isfinite(low_band) & np.isfinite(high_band)
            if np.sum(mask) > 10:  # Cần ít nhất 10 điểm để tính correlation
                corr = np.corrcoef(low_band[mask], high_band[mask])[0, 1]
                wavelength_correlations.append(corr if not np.isnan(corr) else 0)
            else:
                wavelength_correlations.append(0)
    
    wavelength_correlations = np.array(wavelength_correlations)
    
    # Tìm bands có correlation thấp nhất
    low_corr_indices = np.argsort(wavelength_correlations)[:20]
    low_corr_wavelengths = np.array(wavelengths)[low_corr_indices]
    low_corr_values = wavelength_correlations[low_corr_indices]
    
    print(f"   📊 Overall correlation: {np.mean(wavelength_correlations):.4f}")
    print(f"   📉 Lowest correlation bands:")
    for i in range(min(5, len(low_corr_wavelengths))):
        print(f"      {low_corr_wavelengths[i]:.1f}nm: r={low_corr_values[i]:.4f}")
    
    # === 2. PHÂN TÍCH PHÂN BỐ THỐNG KÊ ===
    print("\n📈 2. PHÂN TÍCH PHÂN BỐ THỐNG KÊ:")
    
    # Kiểm tra distribution shapes
    from scipy import stats
    
    if 1 in class_spectra and 3 in class_spectra:
        # Test normality cho một số bands quan trọng
        test_bands = [10, 50, 100, 150, 200]  # Chọn các bands đại diện
        
        for band_idx in test_bands:
            if band_idx < len(wavelengths):
                low_band = class_spectra[1][:, band_idx]
                high_band = class_spectra[3][:, band_idx]
                
                # Shapiro-Wilk test cho normality (sample nhỏ hơn)
                sample_size = min(1000, len(low_band))
                low_sample = np.random.choice(low_band, sample_size, replace=False)
                high_sample = np.random.choice(high_band, sample_size, replace=False)
                
                _, p_low = stats.shapiro(low_sample)
                _, p_high = stats.shapiro(high_sample)
                
                # Kolmogorov-Smirnov test để so sánh distributions
                ks_stat, ks_p = stats.ks_2samp(low_band, high_band)
                
                print(f"   🌊 Band {wavelengths[band_idx]:.1f}nm:")
                print(f"      Normal? Low: p={p_low:.4f}, High: p={p_high:.4f}")
                print(f"      Same distribution? KS: p={ks_p:.4f}")
    
    # === 3. PHÂN TÍCH SPATIAL PATTERNS ===
    print("\n🗺️ 3. PHÂN TÍCH SPATIAL PATTERNS:")
    
    # Tạo spatial maps để xem phân bố các class
    low_mask = (gt_array == 1)
    high_mask = (gt_array == 3)
    
    # Tính spatial autocorrelation
    def calculate_spatial_autocorrelation(mask):
        """Tính Moran's I đơn giản"""
        if np.sum(mask) < 10:
            return 0
        
        coords = np.array(np.where(mask)).T
        if len(coords) < 2:
            return 0
        
        # Tính khoảng cách trung bình giữa các pixel cùng class
        distances = []
        for i in range(0, min(100, len(coords))):  # Sample để tránh quá tải
            for j in range(i+1, min(100, len(coords))):
                dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                distances.append(dist)
        
        return np.mean(distances) if distances else 0
    
    low_autocorr = calculate_spatial_autocorrelation(low_mask)
    high_autocorr = calculate_spatial_autocorrelation(high_mask)
    
    print(f"   📍 Low N2 spatial clustering: {low_autocorr:.2f}")
    print(f"   📍 High N2 spatial clustering: {high_autocorr:.2f}")
    
    # === 4. TẠO VISUALIZATION COMPREHENSIVE ===
    print("\n📊 4. TẠO VISUALIZATION COMPREHENSIVE...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 4.1 Correlation per wavelength
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(wavelengths, wavelength_correlations, 'purple', linewidth=2)
    ax1.axhline(y=0.95, color='red', linestyle='--', label='High correlation threshold')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Low-High N2 Correlation')
    ax1.set_title('Correlation per Wavelength')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight lowest correlation bands
    for i in range(min(3, len(low_corr_indices))):
        idx = low_corr_indices[i]
        ax1.plot(wavelengths[idx], wavelength_correlations[idx], 'ro', markersize=8)
        ax1.annotate(f'{wavelengths[idx]:.0f}nm', 
                    xy=(wavelengths[idx], wavelength_correlations[idx]),
                    xytext=(10, 10), textcoords='offset points')
    
    # 4.2 Distribution comparison cho band có correlation thấp nhất
    if len(low_corr_indices) > 0:
        ax2 = plt.subplot(3, 4, 2)
        worst_band_idx = low_corr_indices[0]
        if 1 in class_spectra and 3 in class_spectra:
            low_values = class_spectra[1][:, worst_band_idx]
            high_values = class_spectra[3][:, worst_band_idx]
            
            ax2.hist(low_values, bins=50, alpha=0.6, color='red', label='Low N2', density=True)
            ax2.hist(high_values, bins=50, alpha=0.6, color='blue', label='High N2', density=True)
            ax2.set_xlabel('Reflectance')
            ax2.set_ylabel('Density')
            ax2.set_title(f'Distribution at {wavelengths[worst_band_idx]:.0f}nm\n(Lowest correlation band)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # 4.3 Spatial distribution maps
    ax3 = plt.subplot(3, 4, 3)
    spatial_map = np.zeros_like(gt_array, dtype=float)
    spatial_map[low_mask] = 1
    spatial_map[high_mask] = 3
    spatial_map[gt_array == 2] = 2
    
    im = ax3.imshow(spatial_map, cmap='RdYlBu_r')
    ax3.set_title('Spatial Distribution\n(Red: Low, Yellow: Medium, Blue: High)')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, shrink=0.8)
    
    # 4.4 Coefficient of Variation comparison
    ax4 = plt.subplot(3, 4, 4)
    if len(class_spectra) >= 3:
        for class_idx in range(1, 4):
            if class_idx in class_spectra:
                spectra = class_spectra[class_idx]
                cv = np.std(spectra, axis=0) / (np.mean(spectra, axis=0) + 1e-8) * 100
                ax4.plot(wavelengths, cv, color=colors[class_idx-1], 
                        label=f'{class_names[class_idx-1]}', linewidth=2)
        
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Coefficient of Variation (%)')
        ax4.set_title('Spectral Variability within Classes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # === 5. KẾT LUẬN VÀ HYPOTHESIS ===
    print("\n🎯 5. KẾT LUẬN VÀ HYPOTHESIS:")
    print("=" * 60)
    
    mean_corr = np.mean(wavelength_correlations)
    min_corr = np.min(wavelength_correlations)
    
    print(f"📊 PHÁT HIỆN QUAN TRỌNG:")
    print(f"   • Mean correlation: {mean_corr:.4f}")
    print(f"   • Minimum correlation: {min_corr:.4f}")
    print(f"   • Bands with correlation < 0.9: {np.sum(wavelength_correlations < 0.9)}")
    
    # Hypothesis generation
    print(f"\n🧠 HYPOTHESES VỀ SỰ TƯƠNG TỰ:")
    
    if mean_corr > 0.98:
        print("   🔬 HYPOTHESIS 1: SIMILAR STRESS RESPONSE")
        print("      • Cả thiếu và thừa N2 đều trigger similar plant stress pathways")
        print("      • Chlorophyll breakdown patterns tương đồng")
        print("      • Cell structure changes theo cùng một pattern")
    
    if min_corr > 0.9:
        print("   🔬 HYPOTHESIS 2: MEASUREMENT SCALE EFFECT")
        print("      • Có thể range của N2 treatments không đủ extreme")
        print("      • 25kg vs 75kg có thể vẫn trong 'stress zone' tương tự")
        print("      • Cần test với extreme levels (0kg vs 150kg)")
    
    if np.std(wavelength_correlations) < 0.05:
        print("   🔬 HYPOTHESIS 3: SENSOR LIMITATION")
        print("      • Hyperspectral sensor có thể không sensitive enough")
        print("      • Atmospheric correction effects")
        print("      • Soil background interference")
    
    print(f"\n🔍 BLIND SPOTS CẦN KHÁM PHÁ:")
    print("   📅 TEMPORAL ANALYSIS:")
    print("      • Phân tích time-series (multiple dates)")
    print("      • Growth stage differences")
    print("      • Phenological responses")
    
    print("   🌡️ ENVIRONMENTAL FACTORS:")
    print("      • Soil moisture variations")
    print("      • Temperature stress interactions")
    print("      • Light conditions during acquisition")
    
    print("   🧬 PHYSIOLOGICAL MECHANISMS:")
    print("      • Leaf chlorophyll content measurements")
    print("      • Plant height, biomass correlations")
    print("      • Root development differences")
    
    print("   📊 ADVANCED SPECTRAL ANALYSIS:")
    print("      • Derivative spectroscopy (1st, 2nd derivatives)")
    print("      • Continuum removal analysis")
    print("      • Spectral unmixing techniques")
    
    print("\n✅ Deep dive analysis completed!")
    return {
        'wavelength_correlations': wavelength_correlations,
        'low_corr_bands': low_corr_wavelengths,
        'mean_correlation': mean_corr,
        'hypotheses': ['stress_response', 'scale_effect', 'sensor_limitation']
    }

def advanced_discrimination_analysis(hsi_array, gt_array, wavelengths=None):
    """
    🎯 PHÂN TÍCH NÂNG CAO: Tìm cách phân biệt Low N2 và High N2
    Dựa trên 4 hướng nghiên cứu: Attention, Texture, Derivative, Context
    """
    print("\n" + "🎯" * 30)
    print("🔍 ADVANCED DISCRIMINATION ANALYSIS")
    print("🎯" * 30)
    
    if wavelengths is None:
        print("❌ Cần thông tin wavelength để phân tích")
        return
    
    # Xử lý GT array
    if len(gt_array.shape) == 3 and gt_array.shape[2] == 1:
        gt_array = gt_array[:, :, 0]
    gt_array = gt_array.astype(int)
    
    # Thu thập dữ liệu
    class_spectra = {}
    spatial_data = {}
    
    for class_idx in [1, 3]:  # Chỉ focus vào Low và High N2
        mask = (gt_array == class_idx)
        if np.sum(mask) > 0:
            mask_indices = np.where(mask)
            sample_size = min(3000, len(mask_indices[0]))
            sample_indices = np.random.choice(len(mask_indices[0]), sample_size, replace=False)
            
            spectra_list = []
            spatial_coords = []
            
            for idx in sample_indices:
                i, j = mask_indices[0][idx], mask_indices[1][idx]
                spectrum = np.array(hsi_array[i, j, :]).flatten()
                spectra_list.append(spectrum)
                spatial_coords.append([i, j])
            
            class_spectra[class_idx] = np.array(spectra_list)
            spatial_data[class_idx] = np.array(spatial_coords)
            
    print(f"📊 Collected data: Low N2: {len(class_spectra[1])}, High N2: {len(class_spectra[3])} samples")
    
    # === HƯỚNG 1: SENSITIVE BANDS & ATTENTION WEIGHTS ===
    print("\n🎯 HƯỚNG 1: TÌM CÁC BANDS NHẠY CẢM NHẤT")
    
    # Tính discrimination power cho từng band
    discrimination_scores = []
    band_importance = []
    
    for band_idx in range(len(wavelengths)):
        low_band = class_spectra[1][:, band_idx]
        high_band = class_spectra[3][:, band_idx]
        
        # Method 1: Fisher Discriminant Ratio
        mean_low = np.mean(low_band)
        mean_high = np.mean(high_band)
        var_low = np.var(low_band)
        var_high = np.var(high_band)
        
        between_class_var = (mean_low - mean_high) ** 2
        within_class_var = (var_low + var_high) / 2
        
        fisher_ratio = between_class_var / (within_class_var + 1e-8)
        discrimination_scores.append(fisher_ratio)
        
        # Method 2: Effect Size (Cohen's d)
        pooled_std = np.sqrt((var_low + var_high) / 2)
        cohens_d = abs(mean_low - mean_high) / (pooled_std + 1e-8)
        band_importance.append(cohens_d)
    
    # Tìm top discriminative bands
    top_bands_fisher = np.argsort(discrimination_scores)[-10:]
    top_bands_cohen = np.argsort(band_importance)[-10:]
    
    print(f"🏆 TOP 5 DISCRIMINATIVE BANDS (Fisher Ratio):")
    for i in range(-5, 0):
        idx = top_bands_fisher[i]
        print(f"   {wavelengths[idx]:.1f}nm: Fisher={discrimination_scores[idx]:.4f}")
    
    print(f"🏆 TOP 5 DISCRIMINATIVE BANDS (Cohen's d):")
    for i in range(-5, 0):
        idx = top_bands_cohen[i]
        print(f"   {wavelengths[idx]:.1f}nm: Cohen's d={band_importance[idx]:.4f}")
    
    # === HƯỚNG 2: TEXTURE ANALYSIS ===
    print("\n🖼️ HƯỚNG 2: PHÂN TÍCH TEXTURE PATTERNS")
    
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import pairwise_distances
    
    # Chọn một vài bands quan trọng cho texture analysis
    selected_bands = top_bands_fisher[-3:]  # Top 3 bands
    
    texture_features = {'low': [], 'high': []}
    
    for class_idx, class_name in [(1, 'low'), (3, 'high')]:
        coords = spatial_data[class_idx]
        
        # Tính Local Binary Pattern style features
        local_variations = []
        local_contrasts = []
        
        # Sample một vài pixel để tính texture
        sample_size = min(500, len(coords))
        sample_indices = np.random.choice(len(coords), sample_size, replace=False)
        
        for idx in sample_indices:
            center_coord = coords[idx]
            
            # Tìm các pixel neighbor trong radius 2
            distances = np.sqrt(np.sum((coords - center_coord)**2, axis=1))
            neighbors = np.where((distances > 0) & (distances <= 2))[0]
            
            if len(neighbors) > 3:  # Đủ neighbors
                for band_idx in selected_bands:
                    center_value = class_spectra[class_idx][idx, band_idx]
                    neighbor_values = class_spectra[class_idx][neighbors, band_idx]
                    
                    # Local variation
                    variation = np.std(neighbor_values)
                    local_variations.append(variation)
                    
                    # Local contrast
                    contrast = np.max(neighbor_values) - np.min(neighbor_values)
                    local_contrasts.append(contrast)
        
        texture_features[class_name] = {
            'mean_variation': np.mean(local_variations),
            'mean_contrast': np.mean(local_contrasts),
            'std_variation': np.std(local_variations),
            'std_contrast': np.std(local_contrasts)
        }
    
    print(f"📊 TEXTURE FEATURES COMPARISON:")
    print(f"   Low N2  - Mean Variation: {texture_features['low']['mean_variation']:.6f}")
    print(f"   High N2 - Mean Variation: {texture_features['high']['mean_variation']:.6f}")
    print(f"   Low N2  - Mean Contrast: {texture_features['low']['mean_contrast']:.6f}")
    print(f"   High N2 - Mean Contrast: {texture_features['high']['mean_contrast']:.6f}")
    
    # === HƯỚNG 3: DERIVATIVE SPECTROSCOPY ===
    print("\n📈 HƯỚNG 3: DERIVATIVE SPECTROSCOPY")
    
    # Tính first và second derivatives
    def calculate_derivatives(spectrum):
        # First derivative
        first_deriv = np.gradient(spectrum)
        # Second derivative
        second_deriv = np.gradient(first_deriv)
        return first_deriv, second_deriv
    
    # Red Edge Position Analysis
    def find_red_edge_position(spectrum, wavelengths):
        # Tìm vùng red edge (700-750nm)
        red_edge_mask = (np.array(wavelengths) >= 700) & (np.array(wavelengths) <= 750)
        if not np.any(red_edge_mask):
            return None
        
        red_edge_spec = spectrum[red_edge_mask]
        red_edge_waves = np.array(wavelengths)[red_edge_mask]
        
        # Tìm điểm có gradient lớn nhất (max của first derivative)
        first_deriv = np.gradient(red_edge_spec)
        max_idx = np.argmax(first_deriv)
        
        return red_edge_waves[max_idx]
    
    # Phân tích derivatives cho mỗi class
    derivative_analysis = {}
    
    for class_idx, class_name in [(1, 'Low N2'), (3, 'High N2')]:
        spectra = class_spectra[class_idx]
        
        red_edge_positions = []
        first_deriv_features = []
        second_deriv_features = []
        
        # Sample để tránh quá tải
        sample_size = min(1000, len(spectra))
        sample_indices = np.random.choice(len(spectra), sample_size, replace=False)
        
        for idx in sample_indices:
            spectrum = spectra[idx]
            
            # Red edge position
            rep = find_red_edge_position(spectrum, wavelengths)
            if rep:
                red_edge_positions.append(rep)
            
            # Derivative features
            first_d, second_d = calculate_derivatives(spectrum)
            
            # Tính một số đặc trưng từ derivatives
            first_deriv_features.append({
                'max': np.max(first_d),
                'min': np.min(first_d),
                'std': np.std(first_d),
                'range': np.max(first_d) - np.min(first_d)
            })
            
            second_deriv_features.append({
                'max': np.max(second_d),
                'min': np.min(second_d), 
                'std': np.std(second_d),
                'range': np.max(second_d) - np.min(second_d)
            })
        
        derivative_analysis[class_name] = {
            'red_edge_pos_mean': np.mean(red_edge_positions),
            'red_edge_pos_std': np.std(red_edge_positions),
            'first_deriv_max_mean': np.mean([f['max'] for f in first_deriv_features]),
            'first_deriv_range_mean': np.mean([f['range'] for f in first_deriv_features]),
            'second_deriv_std_mean': np.mean([f['std'] for f in second_deriv_features])
        }
    
    print(f"📊 DERIVATIVE ANALYSIS:")
    for class_name in ['Low N2', 'High N2']:
        print(f"   {class_name}:")
        print(f"      Red Edge Position: {derivative_analysis[class_name]['red_edge_pos_mean']:.2f} ± {derivative_analysis[class_name]['red_edge_pos_std']:.2f} nm")
        print(f"      1st Deriv Max: {derivative_analysis[class_name]['first_deriv_max_mean']:.6f}")
        print(f"      1st Deriv Range: {derivative_analysis[class_name]['first_deriv_range_mean']:.6f}")
    
    # === HƯỚNG 4: SPATIAL CONTEXT ANALYSIS ===
    print("\n🗺️ HƯỚNG 4: SPATIAL CONTEXT ANALYSIS")
    
    # Phân tích clustering patterns
    def calculate_spatial_clustering(coords):
        if len(coords) < 10:
            return 0
        
        # Tính khoảng cách trung bình đến k nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        
        k = min(5, len(coords)-1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Loại bỏ distance đến chính nó (=0)
        distances = distances[:, 1:]
        
        return np.mean(distances)
    
    # Phân tích patch size requirements
    def analyze_optimal_patch_size(coords, max_patch_size=21):
        patch_analyses = {}
        
        for patch_size in [9, 15, 21]:
            if patch_size > max_patch_size:
                continue
                
            # Đếm số pixel có đủ neighbors trong patch
            valid_patches = 0
            
            for coord in coords[:100]:  # Sample
                i, j = coord
                half_size = patch_size // 2
                
                # Check if patch fits in image bounds
                if (i >= half_size and i < gt_array.shape[0] - half_size and
                    j >= half_size and j < gt_array.shape[1] - half_size):
                    valid_patches += 1
            
            patch_analyses[patch_size] = valid_patches / min(100, len(coords))
        
        return patch_analyses
    
    spatial_analysis = {}
    for class_idx, class_name in [(1, 'Low N2'), (3, 'High N2')]:
        coords = spatial_data[class_idx]
        
        clustering = calculate_spatial_clustering(coords)
        patch_analysis = analyze_optimal_patch_size(coords)
        
        spatial_analysis[class_name] = {
            'clustering_score': clustering,
            'patch_coverage': patch_analysis
        }
    
    print(f"📊 SPATIAL ANALYSIS:")
    for class_name in ['Low N2', 'High N2']:
        print(f"   {class_name}:")
        print(f"      Clustering Score: {spatial_analysis[class_name]['clustering_score']:.2f}")
        print(f"      Patch Coverage: {spatial_analysis[class_name]['patch_coverage']}")
    
    # === VISUALIZATION ===
    print("\n📊 TẠO VISUALIZATION...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Band Importance
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(wavelengths, discrimination_scores, 'purple', linewidth=2, label='Fisher Ratio')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Discrimination Score')
    ax1.set_title('Band Discrimination Power')
    ax1.grid(True, alpha=0.3)
    
    # Highlight top bands
    for i in range(-3, 0):
        idx = top_bands_fisher[i]
        ax1.plot(wavelengths[idx], discrimination_scores[idx], 'ro', markersize=8)
        ax1.annotate(f'{wavelengths[idx]:.0f}nm', 
                    xy=(wavelengths[idx], discrimination_scores[idx]),
                    xytext=(10, 10), textcoords='offset points')
    
    # Plot 2: Derivative Comparison
    ax2 = plt.subplot(2, 4, 2)
    # Tính mean derivatives cho visualization
    mean_spec_low = np.mean(class_spectra[1], axis=0)
    mean_spec_high = np.mean(class_spectra[3], axis=0)
    
    deriv_low = np.gradient(mean_spec_low)
    deriv_high = np.gradient(mean_spec_high)
    
    ax2.plot(wavelengths, deriv_low, 'red', label='Low N2 (1st deriv)', linewidth=2)
    ax2.plot(wavelengths, deriv_high, 'blue', label='High N2 (1st deriv)', linewidth=2)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('First Derivative')
    ax2.set_title('Derivative Spectroscopy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Red Edge Analysis
    ax3 = plt.subplot(2, 4, 3)
    red_edge_data = [derivative_analysis['Low N2']['red_edge_pos_mean'],
                     derivative_analysis['High N2']['red_edge_pos_mean']]
    red_edge_std = [derivative_analysis['Low N2']['red_edge_pos_std'],
                    derivative_analysis['High N2']['red_edge_pos_std']]
    
    bars = ax3.bar(['Low N2', 'High N2'], red_edge_data, 
                   yerr=red_edge_std, capsize=5, 
                   color=['red', 'blue'], alpha=0.7)
    ax3.set_ylabel('Red Edge Position (nm)')
    ax3.set_title('Red Edge Position Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Texture Features
    ax4 = plt.subplot(2, 4, 4)
    texture_comparison = {
        'Variation': [texture_features['low']['mean_variation'], 
                      texture_features['high']['mean_variation']],
        'Contrast': [texture_features['low']['mean_contrast'], 
                     texture_features['high']['mean_contrast']]
    }
    
    x = np.arange(2)
    width = 0.35
    
    ax4.bar(x - width/2, texture_comparison['Variation'], width, 
            label='Variation', color='orange', alpha=0.7)
    ax4.bar(x + width/2, texture_comparison['Contrast'], width, 
            label='Contrast', color='green', alpha=0.7)
    
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Texture Value')
    ax4.set_title('Texture Feature Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Low N2', 'High N2'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # === KẾT QUẢ VÀ KHUYẾN NGHỊ ===
    print("\n🎯 KẾT QUẢ VÀ KHUYẾN NGHỊ:")
    print("=" * 60)
    
    # Tính discrimination potential
    top_fisher_score = np.mean([discrimination_scores[i] for i in top_bands_fisher[-5:]])
    red_edge_diff = abs(derivative_analysis['Low N2']['red_edge_pos_mean'] - 
                       derivative_analysis['High N2']['red_edge_pos_mean'])
    texture_diff = abs(texture_features['low']['mean_variation'] - 
                      texture_features['high']['mean_variation'])
    
    print(f"📊 DISCRIMINATION POTENTIAL:")
    print(f"   🎯 Top 5 bands Fisher score: {top_fisher_score:.4f}")
    print(f"   📈 Red edge position difference: {red_edge_diff:.2f} nm")
    print(f"   🖼️ Texture variation difference: {texture_diff:.6f}")
    
    print(f"\n💡 IMPLEMENTATION RECOMMENDATIONS:")
    
    if top_fisher_score > 0.01:
        print("   ✅ HƯỚNG 1 - ATTENTION MECHANISM: Khả thi cao")
        print("      • Implement channel attention cho top discriminative bands")
        print(f"      • Focus bands: {[wavelengths[i] for i in top_bands_fisher[-3:]]}")
    
    if red_edge_diff > 1.0:
        print("   ✅ HƯỚNG 3 - DERIVATIVE FEATURES: Khả thi cao")
        print("      • Thêm derivative spectra vào input")
        print("      • Đặc biệt focus vào red edge region")
    
    if texture_diff > 0.001:
        print("   ✅ HƯỚNG 2 - TEXTURE ANALYSIS: Khả thi trung bình")
        print("      • Kết hợp texture features với spectral data")
    
    print("   ✅ HƯỚNG 4 - LARGER CONTEXT: Luôn khuyến nghị")
    print("      • Tăng patch size từ 9x9 lên 15x15 hoặc 21x21")
    print("      • Thử nghiệm Multi-scale CNN hoặc Vision Transformer")
    
    return {
        'top_discriminative_bands': [wavelengths[i] for i in top_bands_fisher[-5:]],
        'red_edge_difference': red_edge_diff,
        'texture_features': texture_features,
        'spatial_analysis': spatial_analysis,
        'discrimination_potential': top_fisher_score
    }

if __name__ == "__main__":
    main()
