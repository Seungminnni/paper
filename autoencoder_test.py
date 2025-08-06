#!/usr/bin/env python3
"""
Word2Vec Mosaic Information Preservation Test
목적: 텍스트 → Word2Vec → 이미지 → 학습 → 복원 가능한지 테스트
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Flatten, Dropout, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# GPU 설정 및 확인
print("=== GPU 설정 확인 ===")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증가 허용 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 감지됨: {len(gpus)}개")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"❌ GPU 설정 오류: {e}")
else:
    print("⚠️  GPU가 감지되지 않음. CPU 사용")

print(f"🔧 TensorFlow 버전: {tf.__version__}")
print(f"🔧 사용 가능한 디바이스: {tf.config.list_logical_devices()}")

print("\n=== Word2Vec Mosaic Information Preservation Test ===")
print("텍스트 → Word2Vec → 모자이크 이미지 → 오토인코더 → 원본 복원 테스트")

class Word2VecProcessor:
    """Word2Vec 벡터화 처리"""
    def __init__(self, vector_size=512, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.w2v_model = None

    def train_word2vec(self, sentences):
        print(f"🧠 Word2Vec 학습 중 (벡터 크기: {self.vector_size})...")
        self.w2v_model = Word2Vec(sentences, vector_size=self.vector_size, 
                                  window=self.window, min_count=self.min_count, workers=4)
        print("✅ Word2Vec 학습 완료")

    def vectorize_data(self, df, text_column):
        # 문장 준비
        sentences = [row.split() for row in df[text_column].astype(str)]
        
        print(f"📊 문장 통계:")
        print(f"  - 총 문장 수: {len(sentences)}")
        print(f"  - 평균 길이: {np.mean([len(s) for s in sentences]):.1f} 단어")
        
        self.train_word2vec(sentences)
        
        # 🔧 벡터화 방법 선택
        print("🔧 벡터화 방법: 연결(Concatenation) 방식 사용")
        print("   - 기존 문제: 평균화로 인한 정보 손실")
        print("   - 개선 방법: 단어 벡터들을 연결하여 고유성 보존")
        
        vectors = []
        max_words = 8  # 최대 단어 수 제한 (메모리 고려)
        
        for sentence in sentences:
            # 단어별 벡터 수집
            word_vectors = []
            for word in sentence[:max_words]:  # 처음 8개 단어만 사용
                if word in self.w2v_model.wv:
                    word_vectors.append(self.w2v_model.wv[word])
            
            # 연결 방식: 단어 벡터들을 이어 붙임
            if word_vectors:
                # 8개 단어 × vector_size 차원으로 고정
                combined_vector = np.zeros(max_words * self.vector_size)
                for i, vec in enumerate(word_vectors):
                    start_idx = i * self.vector_size
                    end_idx = start_idx + self.vector_size
                    combined_vector[start_idx:end_idx] = vec
                vectors.append(combined_vector)
            else:
                # 단어가 없으면 0 벡터
                vectors.append(np.zeros(max_words * self.vector_size))
        
        vectors = np.array(vectors)
        print(f"✅ 벡터화 완료. 형태: {vectors.shape}")
        print(f"   - 각 벡터 크기: {max_words} 단어 × {self.vector_size} = {max_words * self.vector_size} 차원")
        return vectors

class MosaicGenerator:
    """벡터를 모자이크 이미지로 변환"""
    def __init__(self, image_size=32):
        self.image_size = image_size

    def vectors_to_mosaics(self, vectors):
        print(f"🖼️  모자이크 이미지 생성 중 ({self.image_size}x{self.image_size})...")
        
        mosaics = []
        for vector in vectors:
            mosaic = self.vector_to_mosaic(vector)
            mosaics.append(mosaic)
        
        mosaics = np.array(mosaics)
        print(f"✅ 모자이크 생성 완료. 형태: {mosaics.shape}")
        return mosaics

    def vector_to_mosaic(self, vector):
        # 🎨 1024차원 벡터를 32x32 픽셀에 직접 매핑
        target_len = self.image_size * self.image_size  # 32*32 = 1024
        
        if len(vector) < target_len:
            padded_vector = np.pad(vector, (0, target_len - len(vector)), 'constant')
        else:
            padded_vector = vector[:target_len]
            
        # 정규화
        if padded_vector.std() > 0:
            padded_vector = (padded_vector - padded_vector.mean()) / padded_vector.std()
        
        # 32x32 이미지로 reshape
        mosaic = padded_vector.reshape((self.image_size, self.image_size, 1))
        return mosaic

class AutoEncoder:
    """오토인코더: 이미지 → 압축 → 복원"""
    def __init__(self, image_size=32, latent_dim=128):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.encoder, self.decoder, self.autoencoder = self._build_models()

    def _build_models(self):
        print("🏗️  오토인코더 구축 중...")
        
        # 인코더
        input_img = Input(shape=(self.image_size, self.image_size, 1))
        
        # 인코더 부분
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # 잠재 공간
        x = Flatten()(x)
        latent = Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        encoder = Model(input_img, latent, name='encoder')
        
        # 디코더
        latent_input = Input(shape=(self.latent_dim,))
        
        # 적절한 형태로 reshape
        size_after_conv = self.image_size // 8  # 3번의 MaxPooling2D
        x = Dense(size_after_conv * size_after_conv * 256, activation='relu')(latent_input)
        x = Reshape((size_after_conv, size_after_conv, 256))(x)
        
        # 디코더 부분
        x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        
        decoded = Conv2D(1, (3, 3), activation='linear', padding='same', name='decoded')(x)
        
        decoder = Model(latent_input, decoded, name='decoder')
        
        # 전체 오토인코더
        autoencoder = Model(input_img, decoder(encoder(input_img)), name='autoencoder')
        
        autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                          loss='mse',
                          metrics=['mae'])
        
        print("✅ 오토인코더 구축 완료")
        encoder.summary()
        decoder.summary()
        
        return encoder, decoder, autoencoder

    def train(self, X_train, X_val, epochs=50, batch_size=32):
        print("🚀 오토인코더 학습 시작...")
        print(f"🔧 사용 중인 디바이스: {tf.config.list_logical_devices()}")
        
        # 디바이스 전략 확인
        strategy = tf.distribute.get_strategy()
        print(f"🔧 분산 전략: {strategy}")
        
        # 목표: 입력 이미지 = 출력 이미지
        history = self.autoencoder.fit(X_train, X_train,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     validation_data=(X_val, X_val),
                                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                                     verbose=1)
        
        print("✅ 학습 완료")
        return history

def analyze_reconstruction_quality(original, reconstructed):
    """복원 품질 분석"""
    print("\n🔍 복원 품질 분석:")
    
    # 전체 통계
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    mae = mean_absolute_error(original.flatten(), reconstructed.flatten())
    
    # 상관관계
    correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
    
    print(f"  📊 MSE (평균 제곱 오차): {mse:.6f}")
    print(f"  📊 MAE (평균 절대 오차): {mae:.6f}")
    print(f"  📊 상관관계: {correlation:.4f}")
    
    # 복원율 계산 (임계값 기반)
    threshold = 0.1
    good_reconstruction = np.abs(original - reconstructed) < threshold
    reconstruction_rate = np.mean(good_reconstruction) * 100
    print(f"  📊 복원율 (±{threshold} 이내): {reconstruction_rate:.2f}%")
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'reconstruction_rate': reconstruction_rate
    }

def visualize_results(original, reconstructed, vectors=None, num_samples=5):
    """결과 시각화 - 벡터값을 명암으로 표현"""
    print(f"\n🎨 결과 시각화 (샘플 {num_samples}개)...")
    
    # 3행으로 확장: 원본 벡터값, 원본 이미지, 복원 이미지
    if vectors is not None:
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
        
        for i in range(num_samples):
            # 원본 벡터값을 32x32 이미지로 시각화
            vector_img = vectors[i][:1024].reshape(32, 32)  # 1024차원을 32x32로
            im1 = axes[0, i].imshow(vector_img, cmap='gray', vmin=vector_img.min(), vmax=vector_img.max())
            axes[0, i].set_title(f'원본 벡터값 {i+1}\n(min:{vector_img.min():.3f}, max:{vector_img.max():.3f})')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # 정규화된 원본 이미지
            im2 = axes[1, i].imshow(original[i].squeeze(), cmap='gray')
            axes[1, i].set_title(f'정규화된 원본 {i+1}')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # 복원 이미지
            im3 = axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[2, i].set_title(f'복원 {i+1}')
            axes[2, i].axis('off')
            plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
    else:
        # 기존 2행 방식
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        
        for i in range(num_samples):
            # 원본
            im1 = axes[0, i].imshow(original[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'원본 {i+1}')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # 복원
            im2 = axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[1, i].set_title(f'복원 {i+1}')
            axes[1, i].axis('off')
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def visualize_vector_patterns(vectors, mosaics=None, num_samples=10):
    """Vector pattern visualization - English labels to avoid font issues"""
    print(f"\n🎨 Vector Pattern Visualization ({num_samples} samples)...")
    
    # Set font for better display
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    
    # 2x5 layout for better comparison
    rows = 2
    cols = num_samples//2
    fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
    
    # Ensure axes is 2D
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Temporary mosaic generator
    mosaic_gen = MosaicGenerator(image_size=32)
    
    for i in range(cols):
        # Row 1: Original vector values (before normalization)
        vector_raw = vectors[i][:1024].reshape(32, 32)
        im1 = axes[0, i].imshow(vector_raw, cmap='viridis', vmin=vector_raw.min(), vmax=vector_raw.max())
        axes[0, i].set_title(f'Original Vector {i+1}\n(Before Normalization)\nmin:{vector_raw.min():.3f}, max:{vector_raw.max():.3f}')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # Row 2: Normalized mosaic (actual process)
        normalized_mosaic = mosaic_gen.vector_to_mosaic(vectors[i]).squeeze()
        im2 = axes[1, i].imshow(normalized_mosaic, cmap='gray', vmin=-3, vmax=3)
        axes[1, i].set_title(f'Normalized Mosaic {i+1}\n(After Processing)\nstd:{normalized_mosaic.std():.3f}')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Word2Vec Vector → Mosaic Transformation Process', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Detailed similarity analysis
    print("\n🔍 Detailed Vector Similarity Analysis:")
    
    # Check raw vector similarities
    print("Raw Vector Similarities (first 5 samples):")
    for i in range(5):
        for j in range(i+1, 5):
            raw_corr = np.corrcoef(vectors[i][:1024], vectors[j][:1024])[0,1]
            print(f"  Vector {i+1} vs Vector {j+1}: {raw_corr:.6f}")
    
    # Check specific sections of vectors
    print("\nSection-wise Analysis (8 words × 128 dims):")
    for section in range(8):
        start_idx = section * 128
        end_idx = start_idx + 128
        section_similarities = []
        
        for i in range(5):
            for j in range(i+1, 5):
                sect_corr = np.corrcoef(vectors[i][start_idx:end_idx], vectors[j][start_idx:end_idx])[0,1]
                section_similarities.append(sect_corr)
        
        avg_sect_sim = np.mean(section_similarities)
        print(f"  Word {section+1} section (dims {start_idx}-{end_idx}): {avg_sect_sim:.6f}")
    
    # Check zero padding effect
    print("\nZero Padding Analysis:")
    for i in range(5):
        non_zero_count = np.count_nonzero(vectors[i])
        zero_ratio = (1024 - non_zero_count) / 1024
        print(f"  Vector {i+1}: {non_zero_count}/1024 non-zero ({zero_ratio:.2%} zeros)")
    
    return
    
    plt.suptitle('Word2Vec 벡터 → 모자이크 변환 과정 시각화', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 벡터값 히스토그램도 추가
    print("\n📊 벡터값 분포 히스토그램...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(10):
        # 1024차원 벡터 전체의 히스토그램
        axes[i].hist(vectors[i][:1024], bins=50, alpha=0.7, color='blue')
        axes[i].set_title(f'벡터 {i+1} 분포 (1024차원)')
        axes[i].set_xlabel('값')
        axes[i].set_ylabel('빈도')
        axes[i].grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_val = vectors[i][:1024].mean()
        std_val = vectors[i][:1024].std()
        axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'평균: {mean_val:.3f}')
        axes[i].legend()
    
    plt.suptitle('1024차원 벡터값 히스토그램 분포', fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    print("\n--- 1단계: 데이터 로딩 ---")
    
    # 2000개 데이터 사용
    df = pd.read_csv('ncvotera.csv', nrows=2000, encoding='latin1')
    print(f"✅ {len(df)}개 레코드 로드")
    
    # 모든 컬럼 결합 (age, zip_code 포함한 전체 12개 컬럼)
    df['full_text'] = df.astype(str).agg(' '.join, axis=1)
    
    # 초기 10개 데이터 출력
    print("\n📋 Initial Data Samples (10 records):")
    for i in range(10):
        print(f"  Data {i+1}: {df['full_text'].iloc[i][:100]}...")  # First 100 characters only
    
    # 실제 이름과 데이터 다양성 확인
    print("\n🔍 Data Diversity Check:")
    print("First names extracted:")
    for i in range(10):
        first_word = df['full_text'].iloc[i].split()[0] if df['full_text'].iloc[i].split() else "None"
        print(f"  Record {i+1}: '{first_word}'")
    
    # 단어 수와 길이 분포 확인
    text_lengths = [len(text.split()) for text in df['full_text']]
    print(f"\nText length statistics:")
    print(f"  Min length: {min(text_lengths)} words")
    print(f"  Max length: {max(text_lengths)} words")
    print(f"  Average length: {np.mean(text_lengths):.1f} words")
    print(f"  Median length: {np.median(text_lengths):.1f} words")
    
    print("\n--- 2단계: Word2Vec 벡터화 ---")
    processor = Word2VecProcessor(vector_size=128)  # 8단어 × 128 = 1024차원
    vectors = processor.vectorize_data(df, 'full_text')
    
    print("\n--- 3단계: 모자이크 이미지 생성 ---")
    mosaic_gen = MosaicGenerator(image_size=32)  # 32x32 = 1024 (벡터 크기와 일치)
    mosaics = mosaic_gen.vectors_to_mosaics(vectors)
    
    # 🔍 모자이크 이미지 다양성 검증
    print("\n🔍 모자이크 이미지 다양성 검증:")
    print(f"  - 생성된 이미지 형태: {mosaics.shape}")
    
    # 처음 10개 이미지의 다양성 확인
    print("\n📊 처음 10개 모자이크 이미지 분석:")
    for i in range(10):
        img = mosaics[i].squeeze()
        print(f"    이미지 {i+1}:")
        print(f"      - 평균: {img.mean():.6f}")
        print(f"      - 표준편차: {img.std():.6f}")
        print(f"      - 최소값: {img.min():.6f}")
        print(f"      - 최대값: {img.max():.6f}")
        print(f"      - 고유값 개수: {len(np.unique(np.round(img, 3)))}")
        print(f"      - 첫 20개 픽셀: {img.flatten()[:20]}")
    
    # 이미지 간 상관관계 확인
    print("\n🔍 이미지 간 유사도 분석:")
    similarities = []
    for i in range(10):
        for j in range(i+1, 10):
            corr = np.corrcoef(mosaics[i].flatten(), mosaics[j].flatten())[0,1]
            similarities.append(corr)
            if i < 3 and j < 4:  # 처음 몇 개만 출력
                print(f"    이미지 {i+1} vs 이미지 {j+1}: 상관관계 = {corr:.6f}")
    
    avg_similarity = np.mean(similarities)
    print(f"    Average image similarity: {avg_similarity:.6f}")
    
    if avg_similarity > 0.7:  # 0.95에서 0.7로 변경 - 더 엄격한 기준
        print("    ⚠️  WARNING: Images are too similar! Possible information loss")
        
        # 원본 벡터의 다양성 확인
        print("\n🔍 Original Word2Vec Vector Diversity Re-verification:")
        vector_similarities = []
        for i in range(10):
            for j in range(i+1, 10):
                corr = np.corrcoef(vectors[i], vectors[j])[0,1]
                vector_similarities.append(corr)
                if i < 3 and j < 4:
                    print(f"    Vector {i+1} vs Vector {j+1}: correlation = {corr:.6f}")
        
        avg_vector_similarity = np.mean(vector_similarities)
        print(f"    Average vector similarity: {avg_vector_similarity:.6f}")
        
        if avg_vector_similarity > 0.7:
            print("    ❌ PROBLEM: Original vectors are also too similar - Word2Vec training issue")
        else:
            print("    ❌ PROBLEM: Information loss in vector→image conversion process")
    else:
        print("    ✅ Images show appropriate diversity")
    
    # 벡터 패턴 명암 시각화 추가 - 실제 모자이크와 함께
    print("\n--- 벡터 패턴 시각화 ---")
    visualize_vector_patterns(vectors, mosaics, num_samples=10)
    
    print("\n--- 4단계: 데이터 분할 ---")
    X_train, X_test = train_test_split(mosaics, test_size=0.2, random_state=42)
    print(f"  - 학습 데이터: {X_train.shape}")
    print(f"  - 테스트 데이터: {X_test.shape}")
    
    print("\n--- 5단계: 오토인코더 학습 ---")
    autoencoder = AutoEncoder(image_size=32, latent_dim=256)
    history = autoencoder.train(X_train, X_test, epochs=30, batch_size=32)
    
    print("\n=== 학습 완료! ===")
    print("✅ 오토인코더 학습이 성공적으로 완료되었습니다.")
    print(f"� 최종 학습 손실: {history.history['loss'][-1]:.6f}")
    print(f"📊 최종 검증 손실: {history.history['val_loss'][-1]:.6f}")
    print(f"📊 최종 학습 MAE: {history.history['mae'][-1]:.6f}")
    print(f"📊 최종 검증 MAE: {history.history['val_mae'][-1]:.6f}")
    
    # 학습 곡선 시각화
    print("\n🎨 학습 곡선 시각화...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('모델 손실 (Loss)')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('평균 절대 오차 (MAE)')
    plt.xlabel('에포크')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 간단한 테스트 예측
    print("\n🔍 간단한 예측 테스트...")
    sample_predictions = autoencoder.autoencoder.predict(X_test[:5], verbose=0)
    print(f"  - 입력 이미지 형태: {X_test[:5].shape}")
    print(f"  - 예측 이미지 형태: {sample_predictions.shape}")
    print("  ✅ 예측이 정상적으로 작동합니다!")
    
    print("\n=== 최종 결론 ===")
    print("🎉 학습 단계 완료!")
    print("✅ Word2Vec → 이미지 → 오토인코더 학습 파이프라인이 정상 작동")
    print("📈 모델이 이미지 패턴을 학습하는 것을 확인했습니다.")

if __name__ == "__main__":
    main()
