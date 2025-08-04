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
print("🚀 BƯỚC 5: XÂY DỰNG VÀ HUẤN LUYỆN 3D-RESNET (PYTORCH)")
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
        
        print("✅ Tải dữ liệu thành công!")
        print(f"📊 Training set: {X_train.shape} | Labels: {y_train.shape}")
        print(f"📊 Validation set: {X_val.shape} | Labels: {y_val.shape}")
        print(f"📊 Patch size: {metadata['patch_size']}x{metadata['patch_size']}x{metadata['n_bands']}")
        print(f"📊 Number of classes: {metadata['n_classes']}")
        print(f"📊 Class names: {metadata['class_names']}")
        
        return X_train, y_train, X_val, y_val, metadata
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        return None, None, None, None, None

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

def train_model_pytorch(model, train_loader, val_loader, num_epochs=100):
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
            torch.save(model.state_dict(), 'models/best_3d_resnet_pytorch.pth')
            print(f"   💾 Lưu model tốt nhất! Val Acc: {best_val_acc:.2f}%")
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
    Hàm chính thực hiện toàn bộ quy trình Bước 5 với PyTorch
    """
    start_time = time.time()
    
    # Bước 5.1: Tải dữ liệu
    X_train, y_train, X_val, y_val, metadata = load_processed_data()
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
    
    # Bước 5.3 & 5.4: Xây dựng mô hình
    NUM_CLASSES = metadata['n_classes']
    model = ResNet3D_Eggplant(num_classes=NUM_CLASSES, input_channels=1)
    
    print(f"\n📋 THÔNG TIN MÔ HÌNH:")
    print(f"📊 Input shape: (batch_size, 1, 277, 9, 9)")
    print(f"🎯 Output classes: {NUM_CLASSES}")
    
    # Đếm số parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🏋️ Trainable parameters: {trainable_params:,}")
    
    # Bước 5.5: Huấn luyện
    history = train_model_pytorch(model, train_loader, val_loader, num_epochs=100)
    
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
