#!/usr/bin/env python3
"""
Simple Word2Vec to Mosaic Learning Test
목적: Word2Vec 벡터 → 픽셀별 직접 매핑 → 학습 가능성 확인
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Flatten, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# GPU 설정
print("=== GPU 설정 확인 ===")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 감지: {len(gpus)}개")
    except RuntimeError as e:
        print(f"❌ GPU 설정 오류: {e}")
else:
    print("⚠️  CPU 사용")

class SentenceMatrixProcessor:
    """Yoon Kim 논문 스타일: 문장을 Word2Vec 행렬로 처리"""
    
    def __init__(self, vector_size=128, max_words=32):
        self.vector_size = vector_size
        self.max_words = max_words  # 최대 단어 수 (패딩 기준)
        self.model = None
        print(f"📄 문장 행렬 처리: {max_words} 단어 × {vector_size} 차원 = [{max_words}x{vector_size}] 행렬")
    
    def train_and_vectorize(self, texts):
        """텍스트를 Word2Vec 학습 후 행렬 형태로 변환"""
        print(f"🧠 Word2Vec 학습 시작 (행렬 방식, 벡터 크기: {self.vector_size})...")
        
        # 문장을 단어로 분할
        sentences = [text.split() for text in texts]
        
        # Word2Vec 학습
        self.model = Word2Vec(sentences, 
                             vector_size=self.vector_size, 
                             window=5, 
                             min_count=1, 
                             workers=4)
        
        print(f"✅ Word2Vec 학습 완료 - 어휘 수: {len(self.model.wv.key_to_index)}")
        
        # 각 문장을 행렬로 변환
        sentence_matrices = []
        for sentence in sentences:
            matrix = self._sentence_to_matrix(sentence)
            sentence_matrices.append(matrix)
        
        sentence_matrices = np.array(sentence_matrices)
        print(f"✅ 문장 행렬 변환 완료: {sentence_matrices.shape}")
        return sentence_matrices
    
    def _sentence_to_matrix(self, sentence):
        """개별 문장을 [max_words x vector_size] 행렬로 변환"""
        # 빈 행렬 초기화
        matrix = np.zeros((self.max_words, self.vector_size))
        
        # 문장의 각 단어를 벡터로 변환하여 행렬에 채움
        for i, word in enumerate(sentence[:self.max_words]):  # max_words 제한
            if word in self.model.wv:
                matrix[i] = self.model.wv[word]
            # 없는 단어는 0 벡터로 유지 (패딩)
        
        return matrix

class MatrixMosaicGenerator:
    """문장 행렬을 모자이크 이미지로 변환"""
    
    def __init__(self, matrix_shape=(32, 128)):
        self.matrix_shape = matrix_shape  # (max_words, vector_size)
        self.max_words, self.vector_size = matrix_shape
        print(f"🖼️  행렬→모자이크: {self.max_words}x{self.vector_size} 행렬 → {self.max_words}x{self.vector_size} 이미지")
    
    def matrices_to_mosaics(self, matrices):
        """문장 행렬들을 모자이크 이미지로 변환"""
        print(f"🎨 행렬 → 모자이크 변환 중...")
        
        mosaics = []
        for i, matrix in enumerate(matrices):
            mosaic = self._matrix_to_mosaic(matrix)
            mosaics.append(mosaic)
            
            # 처음 5개 샘플 정보 출력
            if i < 5:
                non_zero_words = np.count_nonzero(np.sum(matrix, axis=1))  # 실제 단어 수
                print(f"  샘플 {i+1}: {non_zero_words}개 단어, 행렬 범위 [{matrix.min():.3f}, {matrix.max():.3f}] → 모자이크 범위 [{mosaic.min():.3f}, {mosaic.max():.3f}]")
        
        mosaics = np.array(mosaics)
        print(f"✅ 행렬 모자이크 생성 완료: {mosaics.shape}")
        return mosaics
    
    def _matrix_to_mosaic(self, matrix):
        """개별 행렬을 모자이크 이미지로 변환"""
        # 정규화
        if matrix.std() > 0:
            normalized_matrix = (matrix - matrix.mean()) / matrix.std()
            # -3~3 범위로 클리핑 후 0~1로 스케일링
            normalized_matrix = np.clip(normalized_matrix, -3, 3)
            normalized_matrix = (normalized_matrix + 3) / 6
        else:
            normalized_matrix = matrix
        
        # 채널 차원 추가 (32, 128, 1)
        mosaic = np.expand_dims(normalized_matrix, axis=-1)
        return mosaic

class SimpleWord2VecProcessor:
    """간단한 Word2Vec 처리기 (평균 방식)"""
    
    def __init__(self, vector_size=256):
        self.vector_size = vector_size
        self.model = None
    
    def train_and_vectorize(self, texts):
        """텍스트 학습 및 벡터화"""
        print(f"🧠 Word2Vec 학습 시작 (벡터 크기: {self.vector_size})...")
        
        # 문장을 단어로 분할
        sentences = [text.split() for text in texts]
        
        # Word2Vec 학습
        self.model = Word2Vec(sentences, 
                             vector_size=self.vector_size, 
                             window=5, 
                             min_count=1, 
                             workers=4)
        
        print(f"✅ Word2Vec 학습 완료 - 어휘 수: {len(self.model.wv.key_to_index)}")
        
        # 각 문장을 벡터로 변환 (평균 방식)
        vectors = []
        for sentence in sentences:
            word_vectors = [self.model.wv[word] for word in sentence if word in self.model.wv]
            if word_vectors:
                # 문장의 평균 벡터
                sentence_vector = np.mean(word_vectors, axis=0)
            else:
                # 빈 벡터
                sentence_vector = np.zeros(self.vector_size)
            vectors.append(sentence_vector)
        
        vectors = np.array(vectors)
        print(f"✅ 벡터화 완료: {vectors.shape}")
        return vectors

class ConvolutionalMosaicGenerator:
    """컨볼루션 레이어를 사용한 단계적 벡터→이미지 매핑"""
    
    def __init__(self, vector_size=256, final_image_size=32):
        self.vector_size = vector_size
        self.final_image_size = final_image_size
        self.model = self._build_vector_to_image_model()
        print(f"🖼️  컨볼루션 매핑: {vector_size}차원 → {final_image_size}x{final_image_size} 이미지")
    
    def _build_vector_to_image_model(self):
        """벡터를 이미지로 변환하는 컨볼루션 네트워크 구축"""
        print("🏗️  벡터→이미지 변환 네트워크 구축...")
        
        # 입력: 256차원 벡터
        inputs = Input(shape=(self.vector_size,))
        
        # 1단계: Dense → 2D 구조 생성 (4x4x16)
        x = Dense(256, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)  # 중간 Dense 레이어
        x = Reshape((4, 4, 16))(x)  # 4x4 이미지, 16채널
        
        # 2단계: Conv2DTranspose로 점진적 확장
        # 4x4x16 → 8x8x32
        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        
        # 8x8x32 → 16x16x16
        x = Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        
        # 16x16x16 → 32x32x8
        x = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        
        # 최종: 32x32x1 (그레이스케일 이미지)
        outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        model = Model(inputs, outputs, name='vector_to_image')
        print("✅ 벡터→이미지 변환 네트워크 구축 완료")
        model.summary()
        
        return model
    
    def vectors_to_mosaics(self, vectors):
        """벡터를 컨볼루션을 통해 모자이크 이미지로 변환"""
        print(f"🎨 컨볼루션 벡터 → 모자이크 변환 중...")
        
        # 벡터 정규화
        normalized_vectors = []
        for vector in vectors:
            # 벡터를 -1~1 범위로 정규화
            if vector.std() > 0:
                norm_vector = (vector - vector.mean()) / vector.std()
                norm_vector = np.tanh(norm_vector)  # -1~1 범위로 제한
            else:
                norm_vector = vector
            normalized_vectors.append(norm_vector)
        
        normalized_vectors = np.array(normalized_vectors)
        
        # 컨볼루션 네트워크를 통해 이미지 생성
        mosaics = self.model.predict(normalized_vectors, verbose=0)
        
        # 처음 5개 샘플 정보 출력
        for i in range(min(5, len(vectors))):
            print(f"  샘플 {i+1}: 벡터 범위 [{vectors[i].min():.3f}, {vectors[i].max():.3f}] → 모자이크 범위 [{mosaics[i].min():.3f}, {mosaics[i].max():.3f}]")
        
        print(f"✅ 컨볼루션 모자이크 생성 완료: {mosaics.shape}")
        return mosaics

class DirectMosaicGenerator:
    """벡터를 픽셀에 직접 매핑하는 모자이크 생성기 (비교용)"""
    
    def __init__(self, vector_size=256):
        self.vector_size = vector_size
        # 256차원 → 16x16 이미지로 매핑
        self.image_size = int(np.sqrt(vector_size))
        print(f"🖼️  직접 매핑 크기: {self.image_size}x{self.image_size} = {self.image_size**2}차원")
    
    def vectors_to_mosaics(self, vectors):
        """벡터를 모자이크 이미지로 직접 변환"""
        print(f"🎨 직접 벡터 → 모자이크 변환 중...")
        
        mosaics = []
        for i, vector in enumerate(vectors):
            mosaic = self.vector_to_mosaic(vector)
            mosaics.append(mosaic)
            
            # 처음 5개 샘플 정보 출력
            if i < 5:
                print(f"  샘플 {i+1}: 벡터 범위 [{vector.min():.3f}, {vector.max():.3f}] → 모자이크 범위 [{mosaic.min():.3f}, {mosaic.max():.3f}]")
        
        mosaics = np.array(mosaics)
        print(f"✅ 직접 모자이크 생성 완료: {mosaics.shape}")
        return mosaics
    
    def vector_to_mosaic(self, vector):
        """벡터 한 개를 모자이크 이미지로 변환"""
        # 벡터를 이미지 크기에 맞게 조정
        if len(vector) != self.image_size**2:
            if len(vector) > self.image_size**2:
                # 벡터가 더 크면 자르기
                vector = vector[:self.image_size**2]
            else:
                # 벡터가 더 작으면 패딩
                vector = np.pad(vector, (0, self.image_size**2 - len(vector)), 'constant')
        
        # 정규화: 0~1 범위로 변환 (이미지 픽셀값처럼)
        vector_min = vector.min()
        vector_max = vector.max()
        if vector_max > vector_min:
            normalized_vector = (vector - vector_min) / (vector_max - vector_min)
        else:
            normalized_vector = vector
        
        # 이미지 형태로 reshape
        mosaic = normalized_vector.reshape(self.image_size, self.image_size, 1)
        return mosaic

class MatrixCNN:
    """행렬 입력을 위한 CNN (Yoon Kim 스타일)"""
    
    def __init__(self, input_shape=(32, 128, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        """Yoon Kim 논문 스타일의 CNN 구축"""
        print("🏗️  행렬 CNN 오토인코더 구축 (Yoon Kim 스타일)...")
        
        inputs = Input(shape=self.input_shape)  # (32, 128, 1)
        
        # 인코더: 여러 필터 크기로 특징 추출
        # 필터 크기별 컨볼루션 (3, 4, 5 단어 그룹)
        conv_3 = Conv2D(64, (3, self.input_shape[1]), activation='relu', padding='valid')(inputs)  # (30, 1, 64)
        conv_4 = Conv2D(64, (4, self.input_shape[1]), activation='relu', padding='valid')(inputs)  # (29, 1, 64)  
        conv_5 = Conv2D(64, (5, self.input_shape[1]), activation='relu', padding='valid')(inputs)  # (28, 1, 64)
        
        # GlobalMaxPooling으로 가장 중요한 특징 추출
        pool_3 = tf.keras.layers.GlobalMaxPooling2D()(conv_3)  # (64,)
        pool_4 = tf.keras.layers.GlobalMaxPooling2D()(conv_4)  # (64,)
        pool_5 = tf.keras.layers.GlobalMaxPooling2D()(conv_5)  # (64,)
        
        # 특징 연결
        merged = tf.keras.layers.concatenate([pool_3, pool_4, pool_5])  # (192,)
        
        # 잠재 공간
        encoded = Dense(256, activation='relu')(merged)
        
        # 디코더: 잠재 공간에서 원본 행렬 복원
        x = Dense(512, activation='relu')(encoded)
        x = Dense(1024, activation='relu')(x)
        x = Dense(self.input_shape[0] * self.input_shape[1], activation='linear')(x)  # 32*128
        decoded = Reshape(self.input_shape)(x)
        
        # 모델 구성
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("✅ 행렬 CNN 구축 완료")
        autoencoder.summary()
        return autoencoder
    
    def train(self, X_train, X_val, epochs=20):
        """학습 실행"""
        print("🚀 행렬 CNN 학습 시작...")
        
        history = self.model.fit(X_train, X_train,
                                epochs=epochs,
                                batch_size=32,
                                validation_data=(X_val, X_val),
                                verbose=1)
        
        print("✅ 학습 완료")
        return history

class SimpleCNN:
    """간단한 CNN 학습기 (기존 방식)"""
    
    def __init__(self, image_size=32):  # 32x32로 변경
        self.image_size = image_size
        self.model = self._build_model()
    
    def _build_model(self):
        """간단한 오토인코더 CNN 구축"""
        print("🏗️  간단한 CNN 오토인코더 구축...")
        
        # 인코더 (32x32 → 4x4)
        inputs = Input(shape=(self.image_size, self.image_size, 1))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 32x32 → 16x16
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 16x16 → 8x8
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 8x8 → 4x4
        
        # 디코더 (4x4 → 32x32)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 4x4 → 8x8
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 8x8 → 16x16
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 16x16 → 32x32
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        # 모델 구성
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("✅ CNN 구축 완료")
        autoencoder.summary()
        return autoencoder
    
    def train(self, X_train, X_val, epochs=20):
        """학습 실행"""
        print("🚀 CNN 학습 시작...")
        
        history = self.model.fit(X_train, X_train,  # 오토인코더: 입력=출력
                                epochs=epochs,
                                batch_size=32,
                                validation_data=(X_val, X_val),
                                verbose=1)
        
        print("✅ 학습 완료")
        return history

def compare_all_methods(vectors_avg, matrices, mosaics_conv, mosaics_direct, mosaics_matrix, num_samples=3):
    """모든 매핑 방법 비교"""
    print(f"🔍 전체 매핑 방법 비교 시각화 ({num_samples}개 샘플)...")
    
    fig, axes = plt.subplots(5, num_samples, figsize=(15, 20))
    
    for i in range(num_samples):
        # 1. 원본 벡터 (평균 방식)
        axes[0, i].hist(vectors_avg[i], bins=30, alpha=0.7, color='blue')
        axes[0, i].set_title(f'Avg Vector {i+1}\n(min:{vectors_avg[i].min():.3f}, max:{vectors_avg[i].max():.3f})')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Count')
        
        # 2. 문장 행렬 (Yoon Kim 방식)
        im1 = axes[1, i].imshow(matrices[i].squeeze(), cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'Sentence Matrix {i+1}\n32 words × 128 dims')
        axes[1, i].set_xlabel('Vector Dimensions')
        axes[1, i].set_ylabel('Word Position')
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # 3. 컨볼루션 매핑 (32x32)
        im2 = axes[2, i].imshow(mosaics_conv[i].squeeze(), cmap='gray')
        axes[2, i].set_title(f'Conv Mapping {i+1}\n32x32')
        axes[2, i].axis('off')
        plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)
        
        # 4. 직접 매핑 (16x16)
        im3 = axes[3, i].imshow(mosaics_direct[i].squeeze(), cmap='gray')
        axes[3, i].set_title(f'Direct Mapping {i+1}\n16x16')
        axes[3, i].axis('off')
        plt.colorbar(im3, ax=axes[3, i], fraction=0.046, pad=0.04)
        
        # 5. 행렬 모자이크 (32x128)
        im4 = axes[4, i].imshow(mosaics_matrix[i].squeeze(), cmap='gray', aspect='auto')
        axes[4, i].set_title(f'Matrix Mosaic {i+1}\n32x128')
        axes[4, i].set_xlabel('Vector Dimensions')
        axes[4, i].set_ylabel('Word Position')
        plt.colorbar(im4, ax=axes[4, i], fraction=0.046, pad=0.04)
    
    plt.suptitle('All Vector → Mosaic Mapping Methods Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()

def compare_mosaic_methods(vectors, mosaics_conv, mosaics_direct, num_samples=5):
    """컨볼루션 vs 직접 매핑 방법 비교"""
    print(f"🔍 매핑 방법 비교 시각화 ({num_samples}개 샘플)...")
    
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 12))
    
    for i in range(num_samples):
        # 원본 벡터 히스토그램
        axes[0, i].hist(vectors[i], bins=30, alpha=0.7, color='blue')
        axes[0, i].set_title(f'Vector {i+1}\n(min:{vectors[i].min():.3f}, max:{vectors[i].max():.3f})')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Count')
        
        # 컨볼루션 매핑 결과 (32x32)
        im1 = axes[1, i].imshow(mosaics_conv[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Conv Mapping {i+1}\n32x32')
        axes[1, i].axis('off')
        plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # 직접 매핑 결과 (16x16)
        im2 = axes[2, i].imshow(mosaics_direct[i].squeeze(), cmap='gray')
        axes[2, i].set_title(f'Direct Mapping {i+1}\n16x16')
        axes[2, i].axis('off')
        plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Vector → Mosaic Mapping Methods Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_mosaics(vectors, mosaics, num_samples=5):
    """벡터와 모자이크 시각화"""
    print(f"🎨 벡터 → 모자이크 시각화 ({num_samples}개 샘플)...")
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # 원본 벡터 히스토그램
        axes[0, i].hist(vectors[i], bins=30, alpha=0.7, color='blue')
        axes[0, i].set_title(f'Vector {i+1}\n(min:{vectors[i].min():.3f}, max:{vectors[i].max():.3f})')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Count')
        
        # 모자이크 이미지
        im = axes[1, i].imshow(mosaics[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Mosaic {i+1}')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def visualize_reconstruction(original, reconstructed, num_samples=5):
    """복원 결과 시각화"""
    print(f"🎨 복원 결과 시각화 ({num_samples}개 샘플)...")
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # 원본
        im1 = axes[0, i].imshow(original[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # 복원
        im2 = axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def analyze_learning_quality(original, reconstructed):
    """학습 품질 분석"""
    print("\n🔍 학습 품질 분석:")
    
    # MSE, MAE 계산
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    
    # 상관관계
    corr = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
    
    print(f"  📊 MSE (평균 제곱 오차): {mse:.6f}")
    print(f"  📊 MAE (평균 절대 오차): {mae:.6f}")
    print(f"  📊 상관관계: {corr:.4f}")
    
    # 학습 가능성 판정
    if mae < 0.1 and corr > 0.7:
        print("  ✅ 학습 가능: 모델이 패턴을 성공적으로 학습했습니다!")
    elif mae < 0.2 and corr > 0.5:
        print("  🔄 부분 학습: 모델이 일부 패턴을 학습했습니다.")
    else:
        print("  ❌ 학습 부족: 더 많은 학습이나 다른 접근이 필요합니다.")

def main():
    print("\n=== Enhanced Word2Vec → Mosaic Learning Test ===")
    print("목적: 다양한 벡터→이미지 매핑 방법 비교 및 학습 가능성 확인\n")
    
    # 1. 데이터 로딩
    print("--- 1단계: 데이터 로딩 ---")
    df = pd.read_csv('ncvotera.csv', nrows=1000, encoding='latin1')
    
    # 모든 컬럼을 텍스트로 결합
    df['combined_text'] = df.astype(str).agg(' '.join, axis=1)
    texts = df['combined_text'].tolist()
    
    print(f"✅ {len(texts)}개 텍스트 로드")
    print(f"📝 샘플 텍스트: {texts[0][:100]}...")
    
    # 2-1. 평균 벡터 방식 (기존)
    print("\n--- 2-1단계: 평균 벡터 방식 ---")
    processor_avg = SimpleWord2VecProcessor(vector_size=256)
    vectors_avg = processor_avg.train_and_vectorize(texts)
    
    # 2-2. 행렬 방식 (Yoon Kim 스타일)
    print("\n--- 2-2단계: 문장 행렬 방식 (Yoon Kim 스타일) ---")
    processor_matrix = SentenceMatrixProcessor(vector_size=128, max_words=32)
    sentence_matrices = processor_matrix.train_and_vectorize(texts)
    
    # 3-1. 컨볼루션 매핑 (256 → 32x32)
    print("\n--- 3-1단계: 컨볼루션 매핑 ---")
    conv_mosaic_gen = ConvolutionalMosaicGenerator(vector_size=256, final_image_size=32)
    mosaics_conv = conv_mosaic_gen.vectors_to_mosaics(vectors_avg)
    
    # 3-2. 직접 매핑 (256 → 16x16)
    print("\n--- 3-2단계: 직접 매핑 ---")
    direct_mosaic_gen = DirectMosaicGenerator(vector_size=256)
    mosaics_direct = direct_mosaic_gen.vectors_to_mosaics(vectors_avg)
    
    # 3-3. 행렬 모자이크 (32x128 행렬 그대로)
    print("\n--- 3-3단계: 행렬 모자이크 ---")
    matrix_mosaic_gen = MatrixMosaicGenerator(matrix_shape=(32, 128))
    mosaics_matrix = matrix_mosaic_gen.matrices_to_mosaics(sentence_matrices)
    
    # 4. 전체 방법 비교 시각화
    print("\n--- 4단계: 전체 방법 비교 시각화 ---")
    compare_all_methods(vectors_avg, sentence_matrices, mosaics_conv, mosaics_direct, mosaics_matrix, num_samples=3)
    
    # 5. 행렬 방식 학습 테스트 (Yoon Kim 스타일)
    print("\n--- 5단계: 행렬 방식 CNN 학습 테스트 ---")
    X_train_matrix, X_test_matrix = train_test_split(mosaics_matrix, test_size=0.2, random_state=42)
    print(f"  - 행렬 학습 데이터: {X_train_matrix.shape}")
    print(f"  - 행렬 테스트 데이터: {X_test_matrix.shape}")
    
    # 행렬 CNN 학습
    matrix_cnn = MatrixCNN(input_shape=(32, 128, 1))
    history_matrix = matrix_cnn.train(X_train_matrix, X_test_matrix, epochs=10)
    
    # 6. 기존 방식과 성능 비교
    print("\n--- 6단계: 직접 매핑 방식 학습 (비교용) ---")
    X_train_direct, X_test_direct = train_test_split(mosaics_direct, test_size=0.2, random_state=42)
    
    direct_cnn = SimpleCNN(image_size=16)
    history_direct = direct_cnn.train(X_train_direct, X_test_direct, epochs=10)
    
    # 7. 결과 비교
    print("\n--- 7단계: 학습 결과 비교 ---")
    
    # 행렬 방식 결과
    predictions_matrix = matrix_cnn.model.predict(X_test_matrix[:5], verbose=0)
    print("\n🔍 행렬 방식 (Yoon Kim 스타일) 학습 품질:")
    analyze_learning_quality(X_test_matrix, matrix_cnn.model.predict(X_test_matrix, verbose=0))
    
    # 직접 매핑 방식 결과
    predictions_direct = direct_cnn.model.predict(X_test_direct[:5], verbose=0)
    print("\n🔍 직접 매핑 방식 학습 품질:")
    analyze_learning_quality(X_test_direct, direct_cnn.model.predict(X_test_direct, verbose=0))
    
    # 8. 복원 결과 시각화
    print("\n--- 8단계: 복원 결과 시각화 ---")
    
    # 행렬 방식 복원 결과
    print("행렬 방식 복원 결과:")
    visualize_reconstruction(X_test_matrix[:3], predictions_matrix[:3], num_samples=3)
    
    # 직접 매핑 방식 복원 결과
    print("직접 매핑 방식 복원 결과:")
    visualize_reconstruction(X_test_direct[:3], predictions_direct[:3], num_samples=3)
    
    # 9. 학습 곡선 비교
    print("\n--- 9단계: 학습 곡선 비교 ---")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history_matrix.history['loss'], label='Matrix Method Training', color='blue')
    plt.plot(history_matrix.history['val_loss'], label='Matrix Method Validation', color='lightblue')
    plt.plot(history_direct.history['loss'], label='Direct Method Training', color='red')
    plt.plot(history_direct.history['val_loss'], label='Direct Method Validation', color='pink')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history_matrix.history['mae'], label='Matrix Method Training', color='blue')
    plt.plot(history_matrix.history['val_mae'], label='Matrix Method Validation', color='lightblue')
    plt.plot(history_direct.history['mae'], label='Direct Method Training', color='red')
    plt.plot(history_direct.history['val_mae'], label='Direct Method Validation', color='pink')
    plt.title('MAE Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    methods = ['Matrix\n(Yoon Kim)', 'Direct\nMapping']
    final_mae = [history_matrix.history['val_mae'][-1], history_direct.history['val_mae'][-1]]
    colors = ['blue', 'red']
    plt.bar(methods, final_mae, color=colors, alpha=0.7)
    plt.title('Final Validation MAE')
    plt.ylabel('MAE')
    for i, v in enumerate(final_mae):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 최종 결론 ===")
    print("🎉 다양한 Word2Vec → 모자이크 매핑 방법 비교 완료!")
    print(f"📊 행렬 방식 (Yoon Kim) 최종 MAE: {history_matrix.history['val_mae'][-1]:.6f}")
    print(f"📊 직접 매핑 방식 최종 MAE: {history_direct.history['val_mae'][-1]:.6f}")
    
    if history_matrix.history['val_mae'][-1] < history_direct.history['val_mae'][-1]:
        print("✅ 결론: 행렬 방식(Yoon Kim 스타일)이 더 우수한 성능을 보입니다!")
        print("   📝 이유: 단어 위치 정보와 문맥 정보가 더 잘 보존됨")
    else:
        print("✅ 결론: 직접 매핑 방식이 더 우수하거나 비슷한 성능을 보입니다!")
        print("   📝 이유: 간단한 구조로도 충분한 정보 보존 가능")

if __name__ == "__main__":
    main()
