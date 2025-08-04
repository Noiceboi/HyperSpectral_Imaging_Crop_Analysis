"""
Bước 5: Xây dựng và Huấn luyện Mô hình 3D-ResNet
Mục tiêu: Xây dựng kiến trúc 3D-ResNet để phân loại tình trạng dinh dưỡng N2 của cây cà tím
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv3D, BatchNormalization, ReLU, Add, 
                                     GlobalAveragePooling3D, Dense, MaxPooling3D)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import time

# Cấu hình GPU
print("=" * 60)
print("🚀 BƯỚC 5: XÂY DỰNG VÀ HUẤN LUYỆN 3D-RESNET")
print("=" * 60)

# Kiểm tra GPU
print("🔍 KIỂM TRA GPU:")
print(f"🔧 TensorFlow version: {tf.__version__}")
print(f"🏗️ Built with CUDA: {tf.test.is_built_with_cuda()}")

# Kiểm tra GPU bằng cả 2 phương pháp
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"📊 Physical GPUs found: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"   GPU {i}: {gpu}")

# Kiểm tra GPU availability (deprecated nhưng hữu ích để debug)
try:
    gpu_available = tf.test.is_gpu_available()
    print(f"🎯 GPU available (legacy method): {gpu_available}")
except:
    print("🎯 GPU availability check failed")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Tìm thấy {len(gpus)} GPU(s) và đã cấu hình memory growth")
        print("🚀 Sẵn sàng sử dụng GPU cho training!")
    except RuntimeError as e:
        print(f"❌ Lỗi cấu hình GPU: {e}")
        print("⚠️ Sẽ tiếp tục với CPU")
else:
    print("⚠️ Không tìm thấy GPU, sử dụng CPU")
    print("💡 Kiểm tra: NVIDIA driver, CUDA, cuDNN installation")

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

def prepare_labels_for_training(y_train, y_val, num_classes):
    """
    5.1. Chuẩn bị Nhãn cho Training (One-Hot Encoding)
    """
    print("\n🏷️ CHUẨN BỊ NHÃN CHO TRAINING:")
    
    # Kiểm tra phân bố nhãn trước khi chuyển đổi
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    
    print("📊 Phân bố nhãn Training:")
    class_names = ['Low N2', 'Medium N2', 'High N2']
    for label, count in zip(unique_train, counts_train):
        if label < len(class_names):
            print(f"   {class_names[label]} (Class {label}): {count:,} samples")
    
    print("📊 Phân bố nhãn Validation:")
    for label, count in zip(unique_val, counts_val):
        if label < len(class_names):
            print(f"   {class_names[label]} (Class {label}): {count:,} samples")
    
    # Chuyển đổi sang One-Hot Encoding
    print(f"\n🔄 Chuyển đổi nhãn sang One-Hot Encoding...")
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    
    print(f"✅ One-Hot Encoding hoàn thành!")
    print(f"📊 y_train_cat shape: {y_train_cat.shape}")
    print(f"📊 y_val_cat shape: {y_val_cat.shape}")
    print(f"📊 Ví dụ: nhãn {y_train[0]} → {y_train_cat[0]}")
    
    return y_train_cat, y_val_cat

def residual_block(input_tensor, filters, kernel_size=(3, 3, 3)):
    """
    5.2. Định nghĩa Khối dư 3D (3D Residual Block) 
    Kiến trúc: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    x = input_tensor
    
    # Nhánh chính (main path)
    # Lớp tích chập thứ nhất
    main_path = Conv3D(filters, kernel_size=kernel_size, padding='same')(x)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)
    
    # Lớp tích chập thứ hai
    main_path = Conv3D(filters, kernel_size=kernel_size, padding='same')(main_path)
    main_path = BatchNormalization()(main_path)
    
    # Nhánh tắt (shortcut connection)
    # Nếu số lượng bộ lọc đầu vào và đầu ra khác nhau, dùng 1x1x1 conv để khớp kích thước
    if x.shape[-1] != filters:
        shortcut = Conv3D(filters, kernel_size=(1, 1, 1), padding='same')(x)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = x
        
    # Cộng nhánh chính và nhánh tắt
    added = Add()([main_path, shortcut])
    output = ReLU()(added)
    
    return output

def build_detailed_3d_resnet(input_shape, num_classes):
    """
    5.3. Xây dựng Kiến trúc 3D-ResNet Hoàn chỉnh
    """
    print("\n" + "=" * 60)
    print("🏗️ BƯỚC 5.3: XÂY DỰNG KIẾN TRÚC 3D-RESNET")
    print("=" * 60)
    
    print(f"📐 Input shape: {input_shape}")
    print(f"🎯 Number of classes: {num_classes}")
    
    # Lớp đầu vào, thêm 1 chiều kênh cho Conv3D
    inputs = Input(shape=input_shape + (1,))
    print(f"📊 Input layer shape: {inputs.shape}")
    
    # --- Giai đoạn 1: Trích xuất đặc trưng ban đầu ---
    print("🔧 Giai đoạn 1: Trích xuất đặc trưng ban đầu...")
    # Sử dụng kernel lớn hơn ở chiều quang phổ để nắm bắt các mẫu rộng
    x = Conv3D(filters=32, kernel_size=(3, 3, 7), padding='same', name='initial_conv')(inputs)
    x = BatchNormalization(name='initial_bn')(x)
    x = ReLU(name='initial_relu')(x)
    x = MaxPooling3D(pool_size=(1, 1, 2), name='initial_pool')(x)  # Giảm chiều quang phổ
    print(f"   Sau giai đoạn 1: {x.shape}")
    
    # --- Giai đoạn 2: Các Khối dư đầu tiên ---
    print("🔧 Giai đoạn 2: Khối residual đầu tiên...")
    # Chồng 2 khối dư với 32 bộ lọc
    x = residual_block(x, filters=32)
    x = residual_block(x, filters=32)
    print(f"   Sau giai đoạn 2: {x.shape}")
    
    # --- Giai đoạn 3: Giảm kích thước và tăng chiều sâu ---
    print("🔧 Giai đoạn 3: Giảm kích thước và tăng độ sâu...")
    # Giảm kích thước không gian và quang phổ, đồng thời tăng số bộ lọc
    x = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 2), name='transition_conv')(x)
    x = BatchNormalization(name='transition_bn')(x)
    x = ReLU(name='transition_relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), name='transition_pool')(x)
    print(f"   Sau transition: {x.shape}")
    
    # Chồng 2 khối dư với 64 bộ lọc
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    print(f"   Sau các khối residual: {x.shape}")
    
    # --- Giai đoạn 4: Phân loại ---
    print("🔧 Giai đoạn 4: Lớp phân loại...")
    # Lớp Gộp Toàn cục để giảm số lượng tham số
    x = GlobalAveragePooling3D(name='global_avg_pool')(x)
    print(f"   Sau Global Average Pooling: {x.shape}")
    
    # Lớp đầu ra với hàm kích hoạt 'softmax' cho bài toán đa lớp
    outputs = Dense(num_classes, activation='softmax', name='classification_output')(x)
    print(f"   Output shape: {outputs.shape}")
        
    model = Model(inputs=inputs, outputs=outputs, name="3D_ResNet_Eggplant_N2")
    
    print("✅ Xây dựng mô hình hoàn thành!")
    return model

def compile_and_setup_callbacks(model):
    """
    5.4. Biên dịch mô hình và thiết lập Callbacks
    """
    print("\n" + "=" * 60)
    print("⚙️ BƯỚC 5.4: BIÊN DỊCH MÔ HÌNH VÀ THIẾT LẬP CALLBACKS")
    print("=" * 60)
    
    # 1. Biên dịch mô hình
    print("🔧 Biên dịch mô hình...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    print("✅ Biên dịch hoàn thành!")
    
    # 2. Thiết lập Callbacks
    print("📋 Thiết lập Callbacks...")
    
    # Tạo thư mục để lưu model
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # ModelCheckpoint: Lưu lại model có val_accuracy tốt nhất
    checkpoint = ModelCheckpoint(
        'models/best_3d_resnet_model.keras', 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1,
        save_format='keras'
    )
    
    # EarlyStopping: Dừng sớm nếu val_loss không cải thiện
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15,  # Tăng patience cho deep learning
        restore_best_weights=True,
        verbose=1
    )
    
    # ReduceLROnPlateau: Giảm learning rate khi val_loss không cải thiện
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    print("✅ Callbacks đã được thiết lập:")
    print("   📂 ModelCheckpoint: Lưu model tốt nhất")
    print("   ⏹️ EarlyStopping: Dừng sớm sau 15 epochs không cải thiện")
    print("   📉 ReduceLROnPlateau: Giảm learning rate khi cần")
    
    return callbacks

def train_model(model, X_train, y_train_cat, X_val, y_val_cat, callbacks):
    """
    5.5. Huấn luyện mô hình
    """
    print("\n" + "=" * 60)
    print("🚀 BƯỚC 5.5: HUẤN LUYỆN MÔ HÌNH")
    print("=" * 60)
    
    # Cấu hình training
    EPOCHS = 100
    BATCH_SIZE = 64  # Giảm batch size để fit với GPU memory
    
    print(f"📊 Cấu hình huấn luyện:")
    print(f"   🔢 Epochs: {EPOCHS}")
    print(f"   📦 Batch size: {BATCH_SIZE}")
    print(f"   🏋️ Training samples: {len(X_train):,}")
    print(f"   ✅ Validation samples: {len(X_val):,}")
    
    # Thêm chiều channel cho Conv3D input
    print("\n🔄 Chuẩn bị dữ liệu đầu vào...")
    X_train_reshaped = np.expand_dims(X_train, axis=-1)  # (N, 9, 9, 277, 1)
    X_val_reshaped = np.expand_dims(X_val, axis=-1)      # (N, 9, 9, 277, 1)
    
    print(f"📊 Input shapes sau khi reshape:")
    print(f"   X_train: {X_train_reshaped.shape}")
    print(f"   X_val: {X_val_reshaped.shape}")
    
    print("\n🚀 Bắt đầu huấn luyện...")
    start_time = time.time()
    
    try:
        # Huấn luyện mô hình
        history = model.fit(
            X_train_reshaped, y_train_cat,
            validation_data=(X_val_reshaped, y_val_cat),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\n🎉 Huấn luyện hoàn tất!")
        print(f"⏱️ Thời gian huấn luyện: {training_time/60:.1f} phút")
        print(f"📈 Số epochs đã chạy: {len(history.history['loss'])}")
        
        # Hiển thị kết quả cuối cùng
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"\n📊 KẾT QUẢ CUỐI CÙNG:")
        print(f"   🏋️ Training Accuracy: {final_train_acc:.4f}")
        print(f"   ✅ Validation Accuracy: {final_val_acc:.4f}")
        print(f"   📉 Training Loss: {final_train_loss:.4f}")
        print(f"   📉 Validation Loss: {final_val_loss:.4f}")
        
        return history
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình huấn luyện: {str(e)}")
        return None

def visualize_training_results(history):
    """
    5.6. Trực quan hóa Kết quả Huấn luyện
    """
    print("\n" + "=" * 60)
    print("📊 BƯỚC 5.6: TRỰC QUAN HÓA KẾT QUẢ")
    print("=" * 60)
    
    if history is None:
        print("❌ Không có lịch sử huấn luyện để trực quan hóa")
        return
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Đồ thị Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Thêm annotation cho accuracy cao nhất
    max_val_acc = max(history.history['val_accuracy'])
    max_val_epoch = history.history['val_accuracy'].index(max_val_acc)
    ax1.annotate(f'Best Val Acc: {max_val_acc:.4f}',
                xy=(max_val_epoch, max_val_acc),
                xytext=(max_val_epoch+5, max_val_acc-0.02),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    # Đồ thị Loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Thêm annotation cho loss thấp nhất
    min_val_loss = min(history.history['val_loss'])
    min_val_epoch = history.history['val_loss'].index(min_val_loss)
    ax2.annotate(f'Best Val Loss: {min_val_loss:.4f}',
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch+5, min_val_loss+0.05),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("📊 Biểu đồ đã được lưu: models/training_history.png")
    
    plt.show()
    
    # In thống kê chi tiết
    print("\n📈 THỐNG KÊ CHI TIẾT:")
    print(f"   🎯 Best Validation Accuracy: {max_val_acc:.4f} (Epoch {max_val_epoch+1})")
    print(f"   📉 Best Validation Loss: {min_val_loss:.4f} (Epoch {min_val_epoch+1})")
    print(f"   📊 Total Epochs: {len(history.history['loss'])}")
    
    # Kiểm tra overfitting
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    acc_gap = final_train_acc - final_val_acc
    
    if acc_gap > 0.1:
        print(f"⚠️ Có dấu hiệu overfitting (Train-Val gap: {acc_gap:.4f})")
    else:
        print(f"✅ Mô hình ổn định (Train-Val gap: {acc_gap:.4f})")

def main():
    """
    Hàm chính thực hiện toàn bộ quy trình Bước 5
    """
    start_time = time.time()
    
    # Bước 5.1: Tải dữ liệu
    X_train, y_train, X_val, y_val, metadata = load_processed_data()
    if X_train is None:
        return
    
    # Bước 5.1: Chuẩn bị nhãn
    NUM_CLASSES = metadata['n_classes']
    y_train_cat, y_val_cat = prepare_labels_for_training(y_train, y_val, NUM_CLASSES)
    
    # Bước 5.2 & 5.3: Xây dựng mô hình
    INPUT_SHAPE = (metadata['patch_size'], metadata['patch_size'], metadata['n_bands'])
    model = build_detailed_3d_resnet(INPUT_SHAPE, NUM_CLASSES)
    
    # Hiển thị thông tin mô hình
    print(f"\n📋 THÔNG TIN MÔ HÌNH:")
    model.summary()
    
    # Bước 5.4: Biên dịch và thiết lập callbacks
    callbacks = compile_and_setup_callbacks(model)
    
    # Bước 5.5: Huấn luyện
    history = train_model(model, X_train, y_train_cat, X_val, y_val_cat, callbacks)
    
    # Bước 5.6: Trực quan hóa kết quả
    visualize_training_results(history)
    
    # Tổng kết
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH BƯỚC 5")
    print("=" * 60)
    print("✅ Mô hình 3D-ResNet đã được xây dựng")
    print("✅ Huấn luyện hoàn tất")
    print("✅ Kết quả đã được trực quan hóa")
    print("✅ Mô hình tốt nhất đã được lưu")
    print(f"⏱️ Tổng thời gian: {total_time/60:.1f} phút")
    print("➡️ Sẵn sàng cho Bước 6: Đánh giá và Test")
    print("=" * 60)

if __name__ == "__main__":
    main()
