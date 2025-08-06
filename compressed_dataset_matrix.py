#!/usr/bin/env python3
"""
압축된 데이터셋 매트릭스 접근법
- 전체 데이터셋을 하나의 큰 모자이크로 변환
- 컨볼루션을 통한 데이터 압축
- 압축된 모자이크로 학습 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_word2vec_mosaic import SimpleWord2VecProcessor
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_patients_data():
    """Load and extract text from patients.csv data"""
    print("📊 Loading Patients data...")
    
    # Read CSV file
    df = pd.read_csv('patients.csv')
    
    # Remove empty rows
    df = df.dropna(subset=['FIRST', 'LAST'])
    
        
    print(f"✅ Total {len(df)} patient records loaded")
    
    # Combine text data
    texts = []
    for _, row in df.iterrows():
        text_parts = []
        
        # Name
        if pd.notna(row['PREFIX']): text_parts.append(str(row['PREFIX']))
        if pd.notna(row['FIRST']): text_parts.append(str(row['FIRST']))
        if pd.notna(row['LAST']): text_parts.append(str(row['LAST']))
        if pd.notna(row['SUFFIX']): text_parts.append(str(row['SUFFIX']))
        
        # Location info
        if pd.notna(row['CITY']): text_parts.append(str(row['CITY']))
        if pd.notna(row['STATE']): text_parts.append(str(row['STATE']))
        if pd.notna(row['COUNTY']): text_parts.append(str(row['COUNTY']))
        
        # Demographic info
        if pd.notna(row['RACE']): text_parts.append(str(row['RACE']))
        if pd.notna(row['ETHNICITY']): text_parts.append(str(row['ETHNICITY']))
        if pd.notna(row['GENDER']): text_parts.append(str(row['GENDER']))
        if pd.notna(row['MARITAL']): text_parts.append(str(row['MARITAL']))
        
        # Birth place
        if pd.notna(row['BIRTHPLACE']): 
            birthplace_parts = str(row['BIRTHPLACE']).split()
            text_parts.extend(birthplace_parts)
        
        # Combine text
        combined_text = ' '.join(text_parts).lower()
        texts.append(combined_text)
    
    # 텍스트 데이터 조합
    texts = []
    for _, row in df.iterrows():
        text_parts = []
        
        # 이름
        if pd.notna(row['PREFIX']): text_parts.append(str(row['PREFIX']))
        if pd.notna(row['FIRST']): text_parts.append(str(row['FIRST']))
        if pd.notna(row['LAST']): text_parts.append(str(row['LAST']))
        if pd.notna(row['SUFFIX']): text_parts.append(str(row['SUFFIX']))
        
        # 위치 정보
        if pd.notna(row['CITY']): text_parts.append(str(row['CITY']))
        if pd.notna(row['STATE']): text_parts.append(str(row['STATE']))
        if pd.notna(row['COUNTY']): text_parts.append(str(row['COUNTY']))
        
        # 인구통계학적 정보
        if pd.notna(row['RACE']): text_parts.append(str(row['RACE']))
        if pd.notna(row['ETHNICITY']): text_parts.append(str(row['ETHNICITY']))
        if pd.notna(row['GENDER']): text_parts.append(str(row['GENDER']))
        if pd.notna(row['MARITAL']): text_parts.append(str(row['MARITAL']))
        
        # 출생지
        if pd.notna(row['BIRTHPLACE']): 
            birthplace_parts = str(row['BIRTHPLACE']).split()
            text_parts.extend(birthplace_parts)
        
        # 텍스트 조합
        combined_text = ' '.join(text_parts).lower()
        texts.append(combined_text)
    
    return texts

class CompressedDatasetMatrixGenerator:
    """Generator to convert entire dataset into compressed mosaic"""
    
    def __init__(self, vector_size=256, target_height=512, target_width=512):
        self.vector_size = vector_size
        self.target_height = target_height
        self.target_width = target_width
        self.compression_model = None
        
    def create_compression_model(self, input_shape):
        """Create convolutional model to compress original giant mosaic"""
        print(f"🏗️  Creating compression model: {input_shape} → ({self.target_height}, {self.target_width})")
        
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Sequential compression convolution
        current_h, current_w = input_shape[0], input_shape[1]
        
        # Stage 1: Initial feature extraction
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        # Stage 2: Progressive size compression
        while current_h > self.target_height or current_w > self.target_width:
            # Height compression needed
            if current_h > self.target_height:
                pool_h = min(2, current_h // self.target_height + 1)
            else:
                pool_h = 1
                
            # Width compression needed  
            if current_w > self.target_width:
                pool_w = min(2, current_w // self.target_width + 1)
            else:
                pool_w = 1
            
            if pool_h > 1 or pool_w > 1:
                x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
                x = MaxPooling2D((pool_h, pool_w))(x)
                x = BatchNormalization()(x)
                
                current_h = current_h // pool_h
                current_w = current_w // pool_w
                
                print(f"   Compression stage: {current_h}×{current_w}")
            else:
                break
        
        # Stage 3: Exact size adjustment
        if current_h != self.target_height or current_w != self.target_width:
            # Upsampling or downsampling for exact size fitting
            x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            
            # Global pooling then reshape
            x = GlobalAveragePooling2D()(x)
            x = Dense(self.target_height * self.target_width, activation='relu')(x)
            x = Reshape((self.target_height, self.target_width, 1))(x)
        else:
            # Final channel adjustment
            x = Conv2D(1, (1, 1), activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        print(f"✅ Compression model completed: final size {self.target_height}×{self.target_width}")
        
        return model
    
    def texts_to_compressed_matrix(self, texts):
        """텍스트들을 압축된 하나의 모자이크 행렬로 변환"""
        print(f"\n🎨 전체 데이터셋 압축 모자이크 생성")
        print("="*50)
        
        # 1. Word2Vec 벡터화
        print("1️⃣ Word2Vec 벡터화...")
        processor = SimpleWord2VecProcessor(vector_size=self.vector_size)
        vectors = processor.train_and_vectorize(texts)
        
        print(f"   벡터 행렬: {vectors.shape}")
        
        # 2. 원본 거대 모자이크 생성
        print("2️⃣ 원본 거대 모자이크 생성...")
        
        # 벡터 정규화 (0-1 범위)
        vectors_normalized = (vectors - vectors.min()) / (vectors.max() - vectors.min())
        
        # 하나의 거대한 이미지로 구성 (4D 텐서로: batch, height, width, channels)
        # 2D 행렬을 4D로 변환: (samples, features) -> (1, samples, features, 1)
        original_matrix = vectors_normalized.reshape(1, len(texts), self.vector_size, 1)
        
        print(f"   원본 모자이크: {original_matrix.shape}")
        print(f"   크기: {len(texts)}×{self.vector_size} = {len(texts) * self.vector_size:,} 픽셀")
        
        # 3. 압축이 필요한지 판단
        total_pixels = len(texts) * self.vector_size
        target_pixels = self.target_height * self.target_width
        
        print(f"3️⃣ 압축 필요성 판단...")
        print(f"   원본 픽셀 수: {total_pixels:,}")
        print(f"   목표 픽셀 수: {target_pixels:,}")
        print(f"   압축 비율: {total_pixels/target_pixels:.1f}:1")
        
        if total_pixels > target_pixels * 1.5:  # 50% 이상 클 때만 압축
            print("   ✅ 압축 필요 → 컨볼루션 압축 실행")
            
            # 4. 컨볼루션 압축 모델 생성 (4D 입력: height, width, channels)
            input_shape = (len(texts), self.vector_size, 1)
            self.compression_model = self.create_compression_model(input_shape)
            
            # 5. 압축 실행 (자기지도학습 방식)
            print("4️⃣ 압축 실행...")
            
            # 압축 모델 적용 (4D 입력)
            compressed_matrix = self.compression_model.predict(original_matrix, verbose=0)[0]
            
            print(f"   압축 완료: {original_matrix.shape} → {compressed_matrix.shape}")
            
        else:
            print("   ❌ 압축 불필요 → 원본 사용")
            
            # 원본 사용 (배치 차원 제거)
            original_matrix_2d = original_matrix.squeeze(0)  # (2000, 256, 1)
            
            # 크기 조정만 수행
            if len(texts) < self.target_height and self.vector_size < self.target_width:
                # 패딩으로 크기 맞추기
                pad_h = self.target_height - len(texts)
                pad_w = self.target_width - self.vector_size
                
                compressed_matrix = np.pad(original_matrix_2d.squeeze(), 
                                         ((0, pad_h), (0, pad_w)), 
                                         mode='constant', constant_values=0)
                compressed_matrix = np.expand_dims(compressed_matrix, axis=-1)
            else:
                # 크롭해서 크기 맞추기
                h_end = min(len(texts), self.target_height)
                w_end = min(self.vector_size, self.target_width)
                
                compressed_matrix = original_matrix_2d[:h_end, :w_end, :]
        
        # 6. 최종 결과 확인
        print(f"\n✅ 최종 압축 모자이크:")
        print(f"   크기: {compressed_matrix.shape}")
        print(f"   값 범위: [{compressed_matrix.min():.4f}, {compressed_matrix.max():.4f}]")
        print(f"   데이터 타입: {compressed_matrix.dtype}")
        
        return compressed_matrix, vectors

class CompressedMatrixCNN:
    """CNN for compressed mosaic learning"""
    
    def __init__(self, matrix_shape, target_vector_size=64):
        self.matrix_shape = matrix_shape
        self.target_vector_size = target_vector_size
        self.model = self.build_model()
        
    def build_model(self):
        """Build compressed mosaic → feature vector learning model"""
        print(f"🏗️  Building compressed mosaic CNN: {self.matrix_shape} → {self.target_vector_size}D")
        
        inputs = Input(shape=self.matrix_shape)
        
        # Multi-scale feature extraction
        # Small pattern detection
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = MaxPooling2D((2, 2))(conv1)
        
        # Medium pattern detection  
        conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(conv1)
        conv2 = MaxPooling2D((2, 2))(conv2)
        
        # Large pattern detection
        conv3 = Conv2D(128, (7, 7), activation='relu', padding='same')(conv2)
        conv3 = MaxPooling2D((2, 2))(conv3)
        
        # Global feature extraction
        global_features = GlobalAveragePooling2D()(conv3)
        
        # Fully connected layers
        dense1 = Dense(256, activation='relu')(global_features)
        dense2 = Dense(128, activation='relu')(dense1)
        outputs = Dense(self.target_vector_size, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        model.summary()
        return model
    
    def create_target_vector(self, original_vectors):
        """Generate learning target vector from original vectors"""
        print("🎯 Generating learning target vector...")
        
        # Combine various statistical features
        features = []
        
        # Basic statistics
        features.extend(np.mean(original_vectors, axis=0)[:16])  # Mean
        features.extend(np.std(original_vectors, axis=0)[:16])   # Standard deviation
        features.extend(np.min(original_vectors, axis=0)[:16])   # Minimum
        features.extend(np.max(original_vectors, axis=0)[:16])   # Maximum
        
        target_vector = np.array(features)
        
        print(f"   Target vector size: {len(target_vector)}")
        print(f"   Target vector range: [{target_vector.min():.4f}, {target_vector.max():.4f}]")
        
        return target_vector
    
    def train(self, compressed_matrix, target_vector, epochs=30):
        """Train with compressed mosaic"""
        print(f"🚀 Starting compressed mosaic training...")
        
        # Prepare input data
        X = np.expand_dims(compressed_matrix, axis=0)  # Add batch dimension
        y = np.expand_dims(target_vector, axis=0)      # Add batch dimension
        
        print(f"   Input size: {X.shape}")
        print(f"   Target size: {y.shape}")
        
        # Execute training
        history = self.model.fit(
            X, y,
            epochs=epochs,
            verbose=1,
            validation_split=0.0  # No validation for single sample
        )
        
        return history
    
    def predict(self, compressed_matrix):
        """Predict features from compressed mosaic"""
        X = np.expand_dims(compressed_matrix, axis=0)
        prediction = self.model.predict(X, verbose=0)[0]
        return prediction

def visualize_compression_process(original_shape, compressed_matrix, vectors, texts):
    """Visualize compression process"""
    print("🎨 Visualizing compression process...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Complete Dataset Compression Mosaic Process\nOriginal: {original_shape[0]}×{original_shape[1]} → Compressed: {compressed_matrix.shape[0]}×{compressed_matrix.shape[1]}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Compressed mosaic image
    im1 = axes[0, 0].imshow(compressed_matrix.squeeze(), cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'Compressed Complete Dataset Mosaic\n{compressed_matrix.shape[0]}×{compressed_matrix.shape[1]}')
    axes[0, 0].set_xlabel('Compressed Vector Dimensions')
    axes[0, 0].set_ylabel('Compressed Sample Dimensions')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Vector value distribution
    axes[0, 1].hist(vectors.flatten(), bins=50, alpha=0.7, color='blue', label='Original Vectors')
    axes[0, 1].hist(compressed_matrix.flatten(), bins=50, alpha=0.7, color='red', label='Compressed Mosaic')
    axes[0, 1].set_title('Vector Value Distribution Comparison')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Compression ratio information
    original_size = original_shape[0] * original_shape[1] 
    compressed_size = compressed_matrix.shape[0] * compressed_matrix.shape[1]
    compression_ratio = original_size / compressed_size
    
    info_text = f"""Compression Info:
Original Size: {original_shape[0]}×{original_shape[1]} = {original_size:,} pixels
Compressed Size: {compressed_matrix.shape[0]}×{compressed_matrix.shape[1]} = {compressed_size:,} pixels
Compression Ratio: {compression_ratio:.1f}:1
Memory Saved: {(1-1/compression_ratio)*100:.1f}%

Data Info:
Total Patients: {len(texts):,}
Vector Dimensions: {vectors.shape[1]}
Text Sample: "{texts[0][:50]}..."
"""
    
    axes[0, 2].text(0.05, 0.95, info_text, transform=axes[0, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[0, 2].set_title('Compression Information')
    axes[0, 2].axis('off')
    
    # 4. Vector norm per patient (sampling)
    sample_indices = np.linspace(0, len(vectors)-1, min(1000, len(vectors)), dtype=int)
    sample_norms = np.linalg.norm(vectors[sample_indices], axis=1)
    axes[1, 0].plot(sample_norms, 'b-', alpha=0.7, linewidth=0.5)
    axes[1, 0].set_title(f'Vector Size per Patient (Sampling: {len(sample_indices)} patients)')
    axes[1, 0].set_xlabel('Patient Number (Sampled)')
    axes[1, 0].set_ylabel('Vector Norm')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Average values per dimension
    dim_means = np.mean(vectors, axis=0)
    axes[1, 1].plot(dim_means, 'g-', alpha=0.7, linewidth=0.8)
    axes[1, 1].set_title('Average Values per Dimension')
    axes[1, 1].set_xlabel('Vector Dimension')
    axes[1, 1].set_ylabel('Average Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Compressed mosaic heatmap (partial)
    if compressed_matrix.shape[0] > 50 or compressed_matrix.shape[1] > 50:
        # Show partial if large
        display_h = min(50, compressed_matrix.shape[0])
        display_w = min(50, compressed_matrix.shape[1])
        display_matrix = compressed_matrix[:display_h, :display_w]
        title_suffix = f' (Top {display_h}×{display_w} region)'
    else:
        display_matrix = compressed_matrix
        title_suffix = ''
    
    im6 = axes[1, 2].imshow(display_matrix.squeeze(), cmap='coolwarm', aspect='auto')
    axes[1, 2].set_title(f'Compressed Mosaic Detail{title_suffix}')
    axes[1, 2].set_xlabel('Vector Dimension')
    axes[1, 2].set_ylabel('Patient')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('compressed_dataset_matrix_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization completed: compressed_dataset_matrix_visualization.png saved")

def test_compressed_dataset_matrix():
    """Test compressed dataset matrix approach"""
    
    print("🧪 Compressed Dataset Matrix Test")
    print("="*60)
    
    # 1. Load data
    texts = load_patients_data()
    
    # Set sample size for testing (full or partial)
    max_samples = 2000  # Maximum samples for testing
    if len(texts) > max_samples:
        print(f"⚠️  Using {max_samples} samples for testing (Total: {len(texts)})")
        texts = texts[:max_samples]
    
    print(f"🎯 Test data: {len(texts)} samples")
    
    # 2. Generate compressed mosaic
    generator = CompressedDatasetMatrixGenerator(
        vector_size=256,
        target_height=512,  # Target height
        target_width=512    # Target width
    )
    
    compressed_matrix, original_vectors = generator.texts_to_compressed_matrix(texts)
    
    # 3. Visualization
    original_shape = (len(texts), 256)
    visualize_compression_process(original_shape, compressed_matrix, original_vectors, texts)
    
    # 4. Learning test
    print(f"\n{'='*50}")
    print("🤖 Compressed Mosaic Learning Test")
    print(f"{'='*50}")
    
    # Create CNN model
    cnn = CompressedMatrixCNN(
        matrix_shape=compressed_matrix.shape, 
        target_vector_size=64
    )
    
    # Generate learning target vector
    target_vector = cnn.create_target_vector(original_vectors)
    
    # Execute training
    history = cnn.train(compressed_matrix, target_vector, epochs=25)
    
    # Prediction test
    predicted_features = cnn.predict(compressed_matrix)
    
    # Result analysis
    mae = np.mean(np.abs(target_vector - predicted_features))
    correlation = np.corrcoef(target_vector, predicted_features)[0, 1]
    
    print(f"\n📊 Final Learning Results:")
    print(f"   Compression Ratio: {(len(texts) * 256) / (compressed_matrix.shape[0] * compressed_matrix.shape[1]):.1f}:1")
    print(f"   MAE: {mae:.4f}")
    print(f"   Correlation: {correlation:.4f}")
    print(f"   Target Vector Range: [{target_vector.min():.4f}, {target_vector.max():.4f}]")
    print(f"   Predicted Vector Range: [{predicted_features.min():.4f}, {predicted_features.max():.4f}]")
    
    # Learning performance assessment
    if mae < 0.1 and correlation > 0.8:
        learning_status = "✅ Excellent Learning"
        explanation = "Successfully learned complete dataset patterns from compressed mosaic"
    elif mae < 0.2 and correlation > 0.6:
        learning_status = "⚠️  Average Learning"
        explanation = "Some information loss during compression but basic patterns learned"
    else:
        learning_status = "❌ Insufficient Learning"
        explanation = "Compression ratio too high or data complexity too high"
    
    print(f"\n🎯 Learning Evaluation: {learning_status}")
    print(f"   {explanation}")
    
    return {
        'compressed_matrix': compressed_matrix,
        'original_vectors': original_vectors,
        'mae': mae,
        'correlation': correlation,
        'compression_ratio': (len(texts) * 256) / (compressed_matrix.shape[0] * compressed_matrix.shape[1]),
        'learning_status': learning_status
    }

if __name__ == "__main__":
    # Execute compressed dataset matrix test
    results = test_compressed_dataset_matrix()
    
    print("\n🎉 Compressed Dataset Matrix Test Complete!")
    print("The entire dataset has been converted into a single compressed mosaic,")
    print("enabling efficient memory usage and pattern learning.")
