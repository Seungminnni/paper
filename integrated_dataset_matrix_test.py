#!/usr/bin/env python3
"""
전체 데이터셋을 하나의 큰 행렬 이미지로 변환하는 Word2Vec 모자이크 테스트
- 개별 텍스트마다 모자이크가 아닌, 전체 데이터셋을 하나의 큰 이미지로 처리
- 1000명 환자 → 1000×256 행렬 → 하나의 큰 모자이크 이미지
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_word2vec_mosaic import *
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_patients_data():
    """patients.csv 데이터 로드 및 텍스트 추출"""
    print("📊 Patients 데이터 로딩...")
    
    # CSV 파일 읽기
    df = pd.read_csv('patients.csv')
    
    # 빈 행 제거
    df = df.dropna(subset=['FIRST', 'LAST'])
    
    print(f"✅ 총 {len(df)}개 환자 데이터 로드")
    print(f"📋 컬럼: {list(df.columns)}")
    
    # 텍스트 데이터 조합 (이름, 주소, 인종, 성별 등)
    texts = []
    for _, row in df.iterrows():
        # 의미있는 텍스트 필드들을 조합
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
            # 출생지 정보 분할 추가
            birthplace_parts = str(row['BIRTHPLACE']).split()
            text_parts.extend(birthplace_parts)
        
        # 텍스트 조합
        combined_text = ' '.join(text_parts).lower()
        texts.append(combined_text)
    
    # 처음 5개 샘플 확인
    print("\n📝 텍스트 샘플:")
    for i, text in enumerate(texts[:5]):
        print(f"  {i+1}: {text}")
    
    # 텍스트 길이 통계
    text_lengths = [len(text.split()) for text in texts]
    print(f"\n📊 텍스트 통계:")
    print(f"  평균 단어 수: {np.mean(text_lengths):.1f}")
    print(f"  최소/최대 단어 수: {min(text_lengths)}/{max(text_lengths)}")
    print(f"  중앙값 단어 수: {np.median(text_lengths):.1f}")
    
    return texts

class DatasetMatrixGenerator:
    """전체 데이터셋을 하나의 큰 행렬 이미지로 변환하는 생성기"""
    
    def __init__(self, vector_size=256):
        self.vector_size = vector_size
        self.processor = SimpleWord2VecProcessor(vector_size=vector_size)
        print(f"🗂️  데이터셋 행렬 생성기: 전체 데이터 → N×{vector_size} 행렬 → 하나의 큰 이미지")
    
    def create_dataset_matrix_image(self, texts):
        """전체 텍스트 데이터셋을 하나의 큰 행렬 이미지로 변환"""
        print(f"\n🔄 전체 데이터셋 처리 과정:")
        print(f"1단계: {len(texts)}개 텍스트 → Word2Vec 벡터화")
        
        # 전체 텍스트를 Word2Vec으로 벡터화
        vectors = self.processor.train_and_vectorize(texts)
        print(f"   결과: {vectors.shape} 벡터 행렬")
        
        print(f"2단계: {vectors.shape} 행렬 → 하나의 큰 이미지로 변환")
        
        # 행렬을 이미지로 변환
        matrix_image = self._vectors_to_matrix_image(vectors)
        print(f"   결과: {matrix_image.shape} 이미지")
        
        return matrix_image, vectors
    
    def _vectors_to_matrix_image(self, vectors):
        """벡터 행렬을 이미지로 변환"""
        # 정규화 (0~1 범위)
        normalized_vectors = self._normalize_vectors(vectors)
        
        # 채널 차원 추가 (N, vector_size, 1)
        matrix_image = np.expand_dims(normalized_vectors, axis=-1)
        
        print(f"   📊 벡터 통계:")
        print(f"      원본 범위: [{vectors.min():.4f}, {vectors.max():.4f}]")
        print(f"      정규화 후: [{matrix_image.min():.4f}, {matrix_image.max():.4f}]")
        
        return matrix_image
    
    def _normalize_vectors(self, vectors):
        """벡터 정규화"""
        # 표준화 후 0~1 범위로 변환
        scaler = StandardScaler()
        normalized = scaler.fit_transform(vectors)
        
        # Min-max 정규화로 0~1 범위
        min_val = normalized.min()
        max_val = normalized.max()
        if max_val > min_val:
            normalized = (normalized - min_val) / (max_val - min_val)
        
        return normalized

class DatasetMatrixCNN:
    """전체 데이터셋 행렬 이미지를 학습하는 CNN"""
    
    def __init__(self, matrix_shape, vector_size):
        self.matrix_shape = matrix_shape  # (N, vector_size, 1)
        self.vector_size = vector_size
        self.n_samples = matrix_shape[0]
        self.model = self._build_model()
    
    def _build_model(self):
        """전체 데이터셋 행렬 학습 모델 구축"""
        print(f"🏗️  데이터셋 행렬 CNN 구축 ({self.matrix_shape})")
        
        # 입력: 전체 데이터셋 행렬 이미지
        inputs = Input(shape=self.matrix_shape)
        
        # 1D 컨볼루션 (벡터 차원에 대해)
        x = Conv2D(32, (1, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (1, 5), activation='relu', padding='same')(x)
        x = Conv2D(128, (1, 7), activation='relu', padding='same')(x)
        
        # 공간적 특징 추출 (샘플 차원에 대해)
        x = Conv2D(64, (3, 1), activation='relu', padding='same')(x)
        x = Conv2D(32, (5, 1), activation='relu', padding='same')(x)
        
        # 글로벌 특징 추출
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # 완전연결층
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        # 출력: 데이터셋 전체의 표현 벡터
        outputs = Dense(64, activation='linear', name='dataset_embedding')(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("✅ 데이터셋 행렬 CNN 구축 완료")
        model.summary()
        
        return model
    
    def create_target_vectors(self, original_vectors):
        """학습 목표가 될 벡터 생성 (예: 데이터셋 통계적 특성)"""
        print("🎯 학습 목표 벡터 생성...")
        
        # 여러 통계적 특성을 결합한 목표 벡터
        mean_vec = np.mean(original_vectors, axis=0)[:16]  # 평균
        std_vec = np.std(original_vectors, axis=0)[:16]    # 표준편차
        min_vec = np.min(original_vectors, axis=0)[:16]    # 최솟값
        max_vec = np.max(original_vectors, axis=0)[:16]    # 최댓값
        
        target_vector = np.concatenate([mean_vec, std_vec, min_vec, max_vec])
        
        print(f"   목표 벡터 크기: {target_vector.shape}")
        print(f"   목표 벡터 범위: [{target_vector.min():.4f}, {target_vector.max():.4f}]")
        
        return target_vector
    
    def train_dataset_learning(self, matrix_image, target_vector, epochs=50):
        """데이터셋 전체를 학습"""
        print(f"🚀 데이터셋 전체 학습 시작...")
        print(f"   입력: {matrix_image.shape} (전체 데이터셋 이미지)")
        print(f"   목표: {target_vector.shape} (데이터셋 특성 벡터)")
        
        # 배치 차원 추가
        X = np.expand_dims(matrix_image, axis=0)  # (1, N, vector_size, 1)
        y = np.expand_dims(target_vector, axis=0)  # (1, 64)
        
        history = self.model.fit(X, y,
                                epochs=epochs,
                                batch_size=1,
                                verbose=1)
        
        print("✅ 데이터셋 학습 완료")
        return history
    
    def predict_dataset_features(self, matrix_image):
        """데이터셋 이미지에서 특성 예측"""
        X = np.expand_dims(matrix_image, axis=0)
        prediction = self.model.predict(X, verbose=0)
        return prediction[0]

def visualize_dataset_matrix(matrix_image, vectors, texts, title="Dataset Matrix Visualization"):
    """전체 데이터셋 행렬 시각화"""
    print(f"🎨 데이터셋 행렬 시각화: {title}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. 전체 데이터셋 행렬 이미지
    im1 = axes[0, 0].imshow(matrix_image.squeeze(), cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'전체 데이터셋 행렬\n{matrix_image.shape[0]}×{matrix_image.shape[1]} 이미지')
    axes[0, 0].set_xlabel('벡터 차원 (256차원)')
    axes[0, 0].set_ylabel('환자 번호 (N명)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. 벡터 차원별 분포
    axes[0, 1].hist(vectors.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 1].set_title('전체 벡터값 분포')
    axes[0, 1].set_xlabel('벡터 값')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 환자별 벡터 노름
    vector_norms = np.linalg.norm(vectors, axis=1)
    axes[0, 2].plot(vector_norms, 'b-', alpha=0.7)
    axes[0, 2].set_title('환자별 벡터 크기')
    axes[0, 2].set_xlabel('환자 번호')
    axes[0, 2].set_ylabel('벡터 노름')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 차원별 평균값
    dim_means = np.mean(vectors, axis=0)
    axes[1, 0].plot(dim_means, 'r-', alpha=0.7)
    axes[1, 0].set_title('차원별 평균값')
    axes[1, 0].set_xlabel('벡터 차원')
    axes[1, 0].set_ylabel('평균값')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. PCA 분석
    if len(vectors) > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(vectors)
        scatter = axes[1, 1].scatter(pca_result[:, 0], pca_result[:, 1], 
                                   c=range(len(pca_result)), cmap='viridis', alpha=0.6)
        axes[1, 1].set_title(f'PCA 분석\n(설명 분산: {pca.explained_variance_ratio_.sum():.3f})')
        axes[1, 1].set_xlabel('PC1')
        axes[1, 1].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[1, 1])
    
    # 6. 텍스트 샘플 정보
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.9, f"데이터셋 정보:", fontsize=12, fontweight='bold', 
                   transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.8, f"• 총 환자 수: {len(texts)}", fontsize=10, 
                   transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.7, f"• 벡터 차원: {vectors.shape[1]}", fontsize=10, 
                   transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.6, f"• 행렬 크기: {matrix_image.shape[0]}×{matrix_image.shape[1]}", fontsize=10, 
                   transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.5, f"• 벡터 범위: [{vectors.min():.3f}, {vectors.max():.3f}]", fontsize=10, 
                   transform=axes[1, 2].transAxes)
    
    # 샘플 텍스트 표시
    axes[1, 2].text(0.1, 0.4, "샘플 텍스트:", fontsize=10, fontweight='bold', 
                   transform=axes[1, 2].transAxes)
    for i, text in enumerate(texts[:3]):
        axes[1, 2].text(0.1, 0.3-i*0.08, f"{i+1}: {text[:40]}...", fontsize=8, 
                       transform=axes[1, 2].transAxes)
    
    plt.tight_layout()
    plt.savefig('dataset_matrix_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 시각화 완료: dataset_matrix_visualization.png 저장")

def test_dataset_matrix_approach():
    """전체 데이터셋을 하나의 행렬 이미지로 처리하는 접근법 테스트"""
    
    print("="*80)
    print("🧪 전체 데이터셋 행렬 이미지 접근법 테스트")
    print("="*80)
    
    # 1. 데이터 로드
    texts = load_patients_data()
    
    print(f"\n🎯 전체 데이터셋: {len(texts)}개 환자")
    print("📋 접근 방식: 개별 모자이크가 아닌 전체 데이터셋을 하나의 큰 이미지로 처리")
    
    # 2. 전체 데이터셋을 하나의 행렬 이미지로 변환
    print(f"\n{'='*50}")
    print("📊 전체 데이터셋 → 하나의 행렬 이미지 변환")
    print(f"{'='*50}")
    
    generator = DatasetMatrixGenerator(vector_size=256)
    matrix_image, vectors = generator.create_dataset_matrix_image(texts)
    
    print(f"\n✅ 변환 완료:")
    print(f"   📊 원본: {len(texts)}개 텍스트")
    print(f"   📊 벡터 행렬: {vectors.shape}")
    print(f"   📊 행렬 이미지: {matrix_image.shape}")
    print(f"   📊 이미지 크기: {matrix_image.shape[0]}×{matrix_image.shape[1]} 픽셀")
    
    # 3. 시각화
    visualize_dataset_matrix(matrix_image, vectors, texts, 
                           title=f"Patients 데이터셋 전체 행렬 ({len(texts)}×{vectors.shape[1]})")
    
    # 4. 전체 데이터셋 학습 테스트
    print(f"\n{'='*50}")
    print("🤖 전체 데이터셋 학습 테스트")
    print(f"{'='*50}")
    
    # CNN 모델 생성 및 학습
    cnn = DatasetMatrixCNN(matrix_shape=matrix_image.shape, vector_size=256)
    
    # 학습 목표 벡터 생성
    target_vector = cnn.create_target_vectors(vectors)
    
    # 학습 실행
    history = cnn.train_dataset_learning(matrix_image, target_vector, epochs=30)
    
    # 예측 테스트
    predicted_features = cnn.predict_dataset_features(matrix_image)
    
    # 결과 분석
    mae = np.mean(np.abs(target_vector - predicted_features))
    correlation = np.corrcoef(target_vector, predicted_features)[0, 1]
    
    print(f"\n📊 학습 결과:")
    print(f"   MAE: {mae:.4f}")
    print(f"   상관관계: {correlation:.4f}")
    print(f"   목표 벡터 범위: [{target_vector.min():.4f}, {target_vector.max():.4f}]")
    print(f"   예측 벡터 범위: [{predicted_features.min():.4f}, {predicted_features.max():.4f}]")
    
    # 학습 가능성 판정
    if mae < 0.1 and correlation > 0.8:
        learning_status = "✅ 우수한 학습"
        explanation = "전체 데이터셋의 패턴을 잘 학습함"
    elif mae < 0.3 and correlation > 0.5:
        learning_status = "🔄 양호한 학습"
        explanation = "전체 데이터셋의 일부 패턴을 학습함"
    else:
        learning_status = "⚠️  기본 학습"
        explanation = "더 많은 학습이나 다른 접근 필요"
    
    print(f"\n🔍 학습 가능성: {learning_status}")
    print(f"   📝 설명: {explanation}")
    
    # 5. 결론
    print(f"\n{'='*80}")
    print("🎉 전체 데이터셋 행렬 이미지 접근법 테스트 완료!")
    print(f"{'='*80}")
    print(f"✅ 성공적으로 {len(texts)}개 환자 데이터를 {matrix_image.shape[0]}×{matrix_image.shape[1]} 이미지로 변환")
    print(f"✅ 전체 데이터셋의 패턴을 하나의 CNN으로 학습 완료")
    print(f"📊 최종 성능: MAE={mae:.4f}, 상관관계={correlation:.4f}")
    
    return {
        'matrix_image': matrix_image,
        'vectors': vectors,
        'texts': texts,
        'mae': mae,
        'correlation': correlation,
        'learning_status': learning_status
    }

if __name__ == "__main__":
    # 전체 데이터셋 행렬 접근법 테스트 실행
    results = test_dataset_matrix_approach()
