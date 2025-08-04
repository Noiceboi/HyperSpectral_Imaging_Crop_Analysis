"""
Bước 5: Xây dựng và Huấn luyện Mô hình 3D-ResNet với PyTorch
Mục tiêu: Xây dựng kiến trúc 3D-ResNet để phân loại tình trạng dinh dưỡng N2 của cây cà tím
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm

# Cấu hình GPU
print("=" * 60)
print("🚀 BƯỚC 5: XÂY DỰNG VÀ HUẤN LUYỆN 3D-RESNET (PYTORCH) + CHANNEL ATTENTION")
print("=" * 60)

# Kiểm tra GPU
print("🔍 KIỂM TRA GPU:")
print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🏗️ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"📊 CUDA version: {torch.version.cuda}")
    print(f"🎯 GPU device: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = torch.device('cuda')
    print("✅ Sử dụng GPU cho training!")
else:
    device = torch.device('cpu')
    print("⚠️ Không tìm thấy GPU, sử dụng CPU")

def load_processed_data():
    """
    5.1. Tải dữ liệu đã xử lý từ Bước 4
    """
    print("\n" + "=" * 60)
    print("📂 BƯỚC 5.1: TẢI DỮ LIỆU ĐÃ XỬ LÝ")
    print("=" * 60)
    
    DATA_DIR = 'processed_data'
    
    # Kiểm tra sự tồn tại của thư mục
    if not os.path.exists(DATA_DIR):
        print(f"❌ Không tìm thấy thư mục {DATA_DIR}")
        print("Vui lòng chạy Bước 3 & 4 trước!")
        return None, None, None, None, None
    
    try:
        print("🔄 Đang tải dataset...")
        
        # Tải dữ liệu training và validation
        X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
        
        # Tải metadata
        with open(os.path.join(DATA_DIR, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Extract discriminative bands information
        discriminative_info = metadata.get('discriminative_bands', None)
        if discriminative_info:
            discriminative_indices = discriminative_info.get('indices', None)
            discriminative_wavelengths = discriminative_info.get('wavelengths', None)
            target_wavelengths = discriminative_info.get('target_wavelengths', None)
            
            print(f"🎯 Discriminative Bands Info:")
            if target_wavelengths:
                for i, (target, actual, idx) in enumerate(zip(target_wavelengths, discriminative_wavelengths, discriminative_indices)):
                    print(f"   Band {i+1}: {target:.1f}nm → {actual:.1f}nm (Index {idx})")
        else:
            discriminative_indices = None
            discriminative_wavelengths = None
            print("⚠️ No discriminative bands information found")
        
        print("✅ Tải dữ liệu thành công!")
        print(f"📊 Training set: {X_train.shape} | Labels: {y_train.shape}")
        print(f"📊 Validation set: {X_val.shape} | Labels: {y_val.shape}")
        print(f"📊 Patch size: {metadata['patch_size']}x{metadata['patch_size']}x{metadata['n_bands']}")
        print(f"📊 Number of classes: {metadata['n_classes']}")
        print(f"📊 Class names: {metadata['class_names']}")
        
        return X_train, y_train, X_val, y_val, metadata, discriminative_indices
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        return None, None, None, None, None, None

def prepare_data_for_pytorch(X_train, y_train, X_val, y_val):
    """
    5.2. Chuẩn bị dữ liệu cho PyTorch
    """
    print("\n🔄 CHUẨN BỊ DỮ LIỆU CHO PYTORCH:")
    
    # Kiểm tra phân bố nhãn
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    
    print("📊 Phân bố nhãn Training:")
    class_names = ['Low N2', 'Medium N2', 'High N2']
    for label, count in zip(unique_train, counts_train):
        if label < len(class_names):
            print(f"   {class_names[label]} (Class {label}): {count:,} samples")
    
    # Chuyển đổi sang tensor và thêm channel dimension
    # PyTorch format: (N, C, D, H, W) = (batch, channels, depth, height, width)
    print("\n🔄 Chuyển đổi sang PyTorch tensors...")
    
    # Thêm channel dimension và permute để có format (N, C, H, W, D)
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # (N, 1, 9, 9, 277)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)      # (N, 1, 9, 9, 277)
    
    # Permute để có format chuẩn của PyTorch 3D: (N, C, D, H, W)
    X_train_tensor = X_train_tensor.permute(0, 1, 4, 2, 3)  # (N, 1, 277, 9, 9)
    X_val_tensor = X_val_tensor.permute(0, 1, 4, 2, 3)      # (N, 1, 277, 9, 9)
    
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    
    print(f"✅ Tensor conversion hoàn thành!")
    print(f"📊 X_train tensor shape: {X_train_tensor.shape}")
    print(f"📊 X_val tensor shape: {X_val_tensor.shape}")
    print(f"📊 y_train tensor shape: {y_train_tensor.shape}")
    print(f"📊 y_val tensor shape: {y_val_tensor.shape}")
    
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor

class ChannelAttention3D(nn.Module):
    """
    Channel Attention Module dành riêng cho Hyperspectral data
    Tập trung vào các bands phân biệt được xác định từ discriminative analysis
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
            # Tạo learned weights cho discriminative bands
            self.disc_weight = nn.Parameter(torch.ones(len(discriminative_bands)) * 2.0)
            print(f"🎯 Channel Attention: Emphasizing {len(discriminative_bands)} discriminative bands")
        else:
            self.disc_weight = None
            
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (N, C, D, H, W) where D is spectral dimension (277 bands)
        
        # Global pooling
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine average and max pooling
        attention_weights = self.sigmoid(avg_out + max_out)
        
        # Apply discriminative bands emphasis if available
        if self.disc_weight is not None and self.discriminative_bands is not None:
            # Create a mask for discriminative bands
            band_emphasis = torch.ones_like(attention_weights)
            
            # Emphasize discriminative bands (assuming spectral dimension is D)
            for i, band_idx in enumerate(self.discriminative_bands):
                if band_idx < attention_weights.shape[2]:  # Check bounds
                    band_emphasis[:, :, band_idx, :, :] *= self.disc_weight[i]
            
            attention_weights = attention_weights * band_emphasis
        
        # Apply attention weights
        return x * attention_weights

class SpectralAttention3D(nn.Module):
    """
    Spectral Attention Module - tập trung vào dimension phổ
    Đặc biệt hữu ích cho hyperspectral data
    """
    def __init__(self, spectral_size, reduction=8):
        super(SpectralAttention3D, self).__init__()
        
        # Spectral-wise attention
        self.spectral_fc = nn.Sequential(
            nn.Linear(spectral_size, spectral_size // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(spectral_size // reduction, spectral_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (N, C, D, H, W)
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
    5.3. Khối Residual 3D cho PyTorch
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
        # Main path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Shortcut connection
        out += self.shortcut(x)
        out = self.relu(out)
        
        return out

class ResNet3D_Eggplant_WithAttention(nn.Module):
    """
    Enhanced 3D-ResNet với Channel và Spectral Attention
    Tối ưu cho Hyperspectral Eggplant N2 Classification
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
        
        # Channel Attention after initial conv
        self.channel_attention1 = ChannelAttention3D(32, reduction=8, 
                                                   discriminative_bands=discriminative_bands)
        
        # Spectral Attention - ước tính spectral size sau initial conv và pooling
        # Initial: 277 -> conv stride 2 -> 139 -> pool stride 2 -> ~69
        estimated_spectral_size = 69  # Sẽ được điều chỉnh runtime nếu cần
        self.spectral_attention1 = SpectralAttention3D(estimated_spectral_size, reduction=8)
        
        # Giai đoạn 2: Các khối residual đầu tiên
        self.res_block1_1 = ResidualBlock3D(32, 32)
        self.res_block1_2 = ResidualBlock3D(32, 32)
        
        # Channel Attention after res blocks
        self.channel_attention2 = ChannelAttention3D(32, reduction=8,
                                                   discriminative_bands=discriminative_bands)
        
        # Giai đoạn 3: Giảm kích thước và tăng độ sâu
        self.transition_conv = nn.Conv3d(32, 64, kernel_size=1, stride=(2, 1, 1))
        self.transition_bn = nn.BatchNorm3d(64)
        self.transition_relu = nn.ReLU(inplace=True)
        self.transition_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Channel Attention after transition
        self.channel_attention3 = ChannelAttention3D(64, reduction=8,
                                                   discriminative_bands=discriminative_bands)
        
        self.res_block2_1 = ResidualBlock3D(64, 64)
        self.res_block2_2 = ResidualBlock3D(64, 64)
        
        # Final attention
        self.channel_attention4 = ChannelAttention3D(64, reduction=8,
                                                   discriminative_bands=discriminative_bands)
        
        # Giai đoạn 4: Phân loại
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Enhanced classifier với discriminative features
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        print(f"🎯 Enhanced ResNet3D with Attention Mechanisms:")
        if discriminative_bands is not None:
            print(f"   📊 Discriminative bands: {len(discriminative_bands)} bands")
            print(f"   🎯 Target wavelengths: [691.5, 695.9, 693.7] nm")
        
    def forward(self, x):
        # Giai đoạn 1: Initial feature extraction
        x = self.initial_relu(self.initial_bn(self.initial_conv(x)))
        x = self.initial_pool(x)
        
        # Apply channel attention
        x = self.channel_attention1(x)
        
        # Apply spectral attention (with dynamic spectral size handling)
        try:
            if x.shape[2] != self.spectral_attention1.spectral_fc[0].in_features:
                # Update spectral attention size if needed
                actual_spectral_size = x.shape[2]
                self.spectral_attention1 = SpectralAttention3D(actual_spectral_size, reduction=8).to(x.device)
            x = self.spectral_attention1(x)
        except:
            # Skip spectral attention if size mismatch
            pass
        
        # Giai đoạn 2: Residual blocks
        x = self.res_block1_1(x)
        x = self.res_block1_2(x)
        x = self.channel_attention2(x)
        
        # Giai đoạn 3: Transition and more residuals
        x = self.transition_relu(self.transition_bn(self.transition_conv(x)))
        x = self.transition_pool(x)
        x = self.channel_attention3(x)
        
        x = self.res_block2_1(x)
        x = self.res_block2_2(x)
        x = self.channel_attention4(x)
        
        # Giai đoạn 4: Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        
        return x

class ResNet3D_Eggplant(nn.Module):
    """
    5.4. Mô hình 3D-ResNet hoàn chỉnh
    """
    def __init__(self, num_classes=3, input_channels=1):
        super(ResNet3D_Eggplant, self).__init__()
        
        # Giai đoạn 1: Trích xuất đặc trưng ban đầu
        self.initial_conv = nn.Conv3d(input_channels, 32, kernel_size=(7, 3, 3), 
                                     stride=(2, 1, 1), padding=(3, 1, 1))
        self.initial_bn = nn.BatchNorm3d(32)
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # Giai đoạn 2: Các khối residual đầu tiên
        self.res_block1_1 = ResidualBlock3D(32, 32)
        self.res_block1_2 = ResidualBlock3D(32, 32)
        
        # Giai đoạn 3: Giảm kích thước và tăng độ sâu
        self.transition_conv = nn.Conv3d(32, 64, kernel_size=1, stride=(2, 1, 1))
        self.transition_bn = nn.BatchNorm3d(64)
        self.transition_relu = nn.ReLU(inplace=True)
        self.transition_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.res_block2_1 = ResidualBlock3D(64, 64)
        self.res_block2_2 = ResidualBlock3D(64, 64)
        
        # Giai đoạn 4: Phân loại
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
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def train_model_pytorch(model, train_loader, val_loader, model_name="best_3d_resnet_pytorch.pth", num_epochs=100):
    """
    5.5. Huấn luyện mô hình PyTorch
    """
    print("\n" + "=" * 60)
    print("🚀 BƯỚC 5.5: HUẤN LUYỆN MÔ HÌNH PYTORCH")
    print("=" * 60)
    
    # Setup optimizer và loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=10, verbose=True)
    
    # Move model to device
    model = model.to(device)
    
    # Lưu trữ lịch sử training
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 15
    
    print(f"📊 Cấu hình huấn luyện:")
    print(f"   🔢 Epochs: {num_epochs}")
    print(f"   🏋️ Training batches: {len(train_loader)}")
    print(f"   ✅ Validation batches: {len(val_loader)}")
    print(f"   🎯 Device: {device}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Tính toán metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # In kết quả epoch
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}:")
        print(f"   🏋️ Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"   ✅ Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Early stopping and model saving
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            
            # Lưu model tốt nhất
            if not os.path.exists('models'):
                os.makedirs('models')
            model_path = os.path.join('models', model_name)
            torch.save(model.state_dict(), model_path)
            print(f"   💾 Lưu model tốt nhất! Val Acc: {best_val_acc:.2f}% → {model_name}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"\n⏹️ Early stopping sau {patience_limit} epochs không cải thiện!")
            break
    
    training_time = time.time() - start_time
    print(f"\n🎉 Huấn luyện hoàn tất!")
    print(f"⏱️ Thời gian huấn luyện: {training_time/60:.1f} phút")
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def visualize_training_results_pytorch(history):
    """
    5.6. Trực quan hóa kết quả huấn luyện PyTorch
    """
    print("\n" + "=" * 60)
    print("📊 BƯỚC 5.6: TRỰC QUAN HÓA KẾT QUẢ")
    print("=" * 60)
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Đồ thị Accuracy
    ax1.plot(epochs, history['train_accuracies'], label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history['val_accuracies'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Thêm annotation cho accuracy cao nhất
    max_val_acc = max(history['val_accuracies'])
    max_val_epoch = history['val_accuracies'].index(max_val_acc) + 1
    ax1.annotate(f'Best Val Acc: {max_val_acc:.2f}%',
                xy=(max_val_epoch, max_val_acc),
                xytext=(max_val_epoch+2, max_val_acc-2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # Đồ thị Loss
    ax2.plot(epochs, history['train_losses'], label='Training Loss', linewidth=2)
    ax2.plot(epochs, history['val_losses'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Thêm annotation cho loss thấp nhất
    min_val_loss = min(history['val_losses'])
    min_val_epoch = history['val_losses'].index(min_val_loss) + 1
    ax2.annotate(f'Best Val Loss: {min_val_loss:.4f}',
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch+2, min_val_loss+0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig('models/training_history_pytorch.png', dpi=300, bbox_inches='tight')
    print("📊 Biểu đồ đã được lưu: models/training_history_pytorch.png")
    
    plt.show()
    
    # In thống kê chi tiết
    print("\n📈 THỐNG KÊ CHI TIẾT:")
    print(f"   🎯 Best Validation Accuracy: {max_val_acc:.2f}% (Epoch {max_val_epoch})")
    print(f"   📉 Best Validation Loss: {min_val_loss:.4f} (Epoch {min_val_epoch})")
    print(f"   📊 Total Epochs: {len(history['train_losses'])}")
    
    # Kiểm tra overfitting
    final_train_acc = history['train_accuracies'][-1]
    final_val_acc = history['val_accuracies'][-1]
    acc_gap = final_train_acc - final_val_acc
    
    if acc_gap > 10:
        print(f"⚠️ Có dấu hiệu overfitting (Train-Val gap: {acc_gap:.2f}%)")
    else:
        print(f"✅ Mô hình ổn định (Train-Val gap: {acc_gap:.2f}%)")

def main():
    """
    Hàm chính thực hiện toàn bộ quy trình Bước 5 với PyTorch và Channel Attention
    """
    start_time = time.time()
    
    # Bước 5.1: Tải dữ liệu
    X_train, y_train, X_val, y_val, metadata, discriminative_indices = load_processed_data()
    if X_train is None:
        return
    
    # Bước 5.2: Chuẩn bị dữ liệu cho PyTorch
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = prepare_data_for_pytorch(
        X_train, y_train, X_val, y_val)
    
    # Tạo DataLoaders
    BATCH_SIZE = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n📦 DataLoaders created:")
    print(f"   🏋️ Train batches: {len(train_loader)}")
    print(f"   ✅ Val batches: {len(val_loader)}")
    
    # Bước 5.3 & 5.4: Xây dựng mô hình với Attention
    NUM_CLASSES = metadata['n_classes']
    
    # Choose model architecture based on discriminative bands availability
    if discriminative_indices is not None:
        print(f"\n🎯 USING ENHANCED MODEL WITH CHANNEL ATTENTION")
        model = ResNet3D_Eggplant_WithAttention(
            num_classes=NUM_CLASSES, 
            input_channels=1,
            discriminative_bands=discriminative_indices
        )
        model_name = "best_3d_resnet_pytorch_attention.pth"
    else:
        print(f"\n📊 USING STANDARD MODEL (No discriminative bands found)")
        model = ResNet3D_Eggplant(num_classes=NUM_CLASSES, input_channels=1)
        model_name = "best_3d_resnet_pytorch.pth"
    
    print(f"\n📋 THÔNG TIN MÔ HÌNH:")
    print(f"📊 Input shape: (batch_size, 1, 277, 9, 9)")
    print(f"🎯 Output classes: {NUM_CLASSES}")
    if discriminative_indices is not None:
        print(f"🎯 Discriminative bands: {len(discriminative_indices)} bands")
    
    # Đếm số parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🏋️ Trainable parameters: {trainable_params:,}")
    
    # Bước 5.5: Huấn luyện
    history = train_model_pytorch(model, train_loader, val_loader, model_name, num_epochs=100)
    
    # Bước 5.6: Trực quan hóa kết quả
    visualize_training_results_pytorch(history)
    
    # Tổng kết
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH BƯỚC 5 (PYTORCH)")
    print("=" * 60)
    print("✅ Mô hình 3D-ResNet PyTorch đã được xây dựng")
    print("✅ Huấn luyện hoàn tất")
    print("✅ Kết quả đã được trực quan hóa")
    print("✅ Mô hình tốt nhất đã được lưu")
    print(f"⏱️ Tổng thời gian: {total_time/60:.1f} phút")
    print("➡️ Sẵn sàng cho Bước 6: Đánh giá và Test")
    print("=" * 60)

if __name__ == "__main__":
    main()
