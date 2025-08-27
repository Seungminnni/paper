#!/usr/bin/env python3
"""
Patients.csv 데이터로 올바른 Word2Vec 모자이크 학습 테스트
- 1단계: 텍스트 → Word2Vec 벡터 → 모자이크 생성
- 2단계: 모자이크 → 벡터 복원 학습 (올바른 학습)
- 3단계: 복원된 벡터 → 텍스트 유사도 검증
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_word2vec_mosaic import *
import pandas as pd
from sklearn.metrics import cosine_similarity

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

class MosaicToVectorCNN:
    """모자이크 → 벡터 복원 전용 CNN"""
    
    def __init__(self, mosaic_shape, vector_size):
        self.mosaic_shape = mosaic_shape  # (height, width, channels)
        self.vector_size = vector_size
        self.model = self._build_model()
    
    def _build_model(self):
        """모자이크 → 벡터 복원 모델 구축"""
        print(f"🏗️  모자이크→벡터 복원 CNN 구축 ({self.mosaic_shape} → {self.vector_size})")
        
        # 입력: 모자이크 이미지
        inputs = Input(shape=self.mosaic_shape)
        
        if len(self.mosaic_shape) == 3 and self.mosaic_shape[0] == self.mosaic_shape[1]:
            # 정사각형 이미지 (16x16 또는 32x32)
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        else:
            # 직사각형 이미지 (32x128)
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Dense 레이어로 벡터 크기에 맞춤
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        outputs = Dense(self.vector_size, activation='linear')(x)  # 벡터 복원
        
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("✅ 모자이크→벡터 복원 CNN 구축 완료")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        """학습 실행 (X: 모자이크, y: 벡터)"""
        print(f"🚀 모자이크→벡터 복원 학습 시작...")
        print(f"   입력: {X_train.shape} (모자이크)")
        print(f"   출력: {y_train.shape} (벡터)")
        
        history = self.model.fit(X_train, y_train,
                                epochs=epochs,
                                batch_size=32,
                                validation_data=(X_val, y_val),
                                verbose=1)
        
        print("✅ 학습 완료")
        return history
    
    def evaluate(self, X_test, y_test):
        """테스트 평가"""
        predictions = self.model.predict(X_test, verbose=0)
        
        # MAE 계산
        mae = np.mean(np.abs(y_test - predictions))
        
        # 상관관계 계산
        correlation = np.corrcoef(y_test.flatten(), predictions.flatten())[0, 1]
        
        # 코사인 유사도 계산 (벡터별)
        cosine_similarities = []
        for i in range(len(y_test)):
            cos_sim = cosine_similarity([y_test[i]], [predictions[i]])[0, 0]
            cosine_similarities.append(cos_sim)
        avg_cosine_sim = np.mean(cosine_similarities)
        
        return mae, correlation, avg_cosine_sim, predictions

def test_correct_learning_with_patients():
    """환자 데이터로 올바른 학습 테스트"""
    
    # 데이터 로드
    texts = load_patients_data()
    
    # 샘플 수 제한 (빠른 테스트를 위해)
    sample_size = min(500, len(texts))
    texts = texts[:sample_size]
    print(f"\n🎯 테스트 샘플: {len(texts)}개")
    
    print("\n" + "="*70)
    print("🧪 PATIENTS 데이터 올바른 Word2Vec 모자이크 학습 테스트")
    print("="*70)
    
    results = {}
    
    # 1. 직접 매핑 방식 올바른 학습
    print("\n🔹 1. 직접 매핑 방식 (올바른 학습)")
    print("-" * 50)
    try:
        # 1단계: 텍스트 → 벡터 → 모자이크
        print("1단계: 텍스트 → 벡터 → 모자이크 생성")
        processor = SimpleWord2VecProcessor(vector_size=256)
        generator = DirectMosaicGenerator(vector_size=256)
        
        vectors = processor.train_and_vectorize(texts)
        mosaics = generator.vectors_to_mosaics(vectors)
        
        # 2단계: 모자이크 → 벡터 복원 학습
        print("2단계: 모자이크 → 벡터 복원 학습")
        X_train, X_test, y_train, y_test = train_test_split(mosaics, vectors, test_size=0.2, random_state=42)
        
        cnn = MosaicToVectorCNN(mosaic_shape=mosaics.shape[1:], vector_size=256)
        history = cnn.train(X_train, y_train, X_test, y_test, epochs=30)
        
        # 3단계: 평가
        print("3단계: 복원 성능 평가")
        mae, correlation, cosine_sim, predictions = cnn.evaluate(X_test, y_test)
        
        results['direct'] = {
            'mae': mae,
            'correlation': correlation,
            'cosine_similarity': cosine_sim,
            'mosaics': mosaics,
            'vectors': vectors,
            'predictions': predictions,
            'y_test': y_test,
            'history': history
        }
        
        print(f"✅ 직접 매핑 완료:")
        print(f"   📊 MAE: {mae:.4f}")
        print(f"   📊 상관관계: {correlation:.4f}")
        print(f"   📊 평균 코사인 유사도: {cosine_sim:.4f}")
        
    except Exception as e:
        print(f"❌ 직접 매핑 오류: {e}")
        import traceback
        traceback.print_exc()
        results['direct'] = None
    
    # 2. 컨볼루션 방식 올바른 학습
    print("\n🔹 2. 컨볼루션 방식 (올바른 학습)")
    print("-" * 50)
    try:
        # 1단계: 텍스트 → 벡터 → 모자이크
        print("1단계: 텍스트 → 벡터 → 모자이크 생성")
        processor = SimpleWord2VecProcessor(vector_size=256)
        generator = ConvolutionalMosaicGenerator(vector_size=256, final_image_size=32)
        
        vectors = processor.train_and_vectorize(texts)
        mosaics = generator.vectors_to_mosaics(vectors)
        
        # 2단계: 모자이크 → 벡터 복원 학습
        print("2단계: 모자이크 → 벡터 복원 학습")
        X_train, X_test, y_train, y_test = train_test_split(mosaics, vectors, test_size=0.2, random_state=42)
        
        cnn = MosaicToVectorCNN(mosaic_shape=mosaics.shape[1:], vector_size=256)
        history = cnn.train(X_train, y_train, X_test, y_test, epochs=30)
        
        # 3단계: 평가
        print("3단계: 복원 성능 평가")
        mae, correlation, cosine_sim, predictions = cnn.evaluate(X_test, y_test)
        
        results['convolutional'] = {
            'mae': mae,
            'correlation': correlation,
            'cosine_similarity': cosine_sim,
            'mosaics': mosaics,
            'vectors': vectors,
            'predictions': predictions,
            'y_test': y_test,
            'history': history
        }
        
        print(f"✅ 컨볼루션 방식 완료:")
        print(f"   📊 MAE: {mae:.4f}")
        print(f"   📊 상관관계: {correlation:.4f}")
        print(f"   📊 평균 코사인 유사도: {cosine_sim:.4f}")
        
    except Exception as e:
        print(f"❌ 컨볼루션 방식 오류: {e}")
        import traceback
        traceback.print_exc()
        results['convolutional'] = None
    
    # 3. 행렬 방식 올바른 학습
    print("\n🔹 3. 행렬 방식 (Yoon Kim 스타일, 올바른 학습)")
    print("-" * 50)
    try:
        # 1단계: 텍스트 → 행렬 → 모자이크
        print("1단계: 텍스트 → 행렬 → 모자이크 생성")
        processor = SentenceMatrixProcessor(vector_size=128, max_words=32)
        generator = MatrixMosaicGenerator(matrix_shape=(32, 128))
        
        sentence_matrices = processor.train_and_vectorize(texts)
        mosaics = generator.matrices_to_mosaics(sentence_matrices)
        
        # 평균 벡터 계산 (비교용)
        avg_vectors = np.mean(sentence_matrices, axis=1)  # (N, 128)
        
        # 2단계: 모자이크 → 평균벡터 복원 학습
        print("2단계: 모자이크 → 평균벡터 복원 학습")
        X_train, X_test, y_train, y_test = train_test_split(mosaics, avg_vectors, test_size=0.2, random_state=42)
        
        cnn = MosaicToVectorCNN(mosaic_shape=mosaics.shape[1:], vector_size=128)
        history = cnn.train(X_train, y_train, X_test, y_test, epochs=30)
        
        # 3단계: 평가
        print("3단계: 복원 성능 평가")
        mae, correlation, cosine_sim, predictions = cnn.evaluate(X_test, y_test)
        
        results['matrix'] = {
            'mae': mae,
            'correlation': correlation,
            'cosine_similarity': cosine_sim,
            'mosaics': mosaics,
            'vectors': avg_vectors,
            'predictions': predictions,
            'y_test': y_test,
            'history': history
        }
        
        print(f"✅ 행렬 방식 완료:")
        print(f"   📊 MAE: {mae:.4f}")
        print(f"   📊 상관관계: {correlation:.4f}")
        print(f"   📊 평균 코사인 유사도: {cosine_sim:.4f}")
        
    except Exception as e:
        print(f"❌ 행렬 방식 오류: {e}")
        import traceback
        traceback.print_exc()
        results['matrix'] = None
    
    # 결과 비교 및 시각화
    print("\n" + "="*70)
    print("📊 PATIENTS 데이터 올바른 학습 결과 비교")
    print("="*70)
    
    # 성능 비교 테이블
    print("\n🏆 벡터 복원 성능 비교:")
    print(f"{'방법':<15} {'MAE':<10} {'상관관계':<10} {'코사인유사도':<12} {'이미지크기':<12}")
    print("-" * 65)
    
    if results['direct']:
        r = results['direct']
        print(f"{'직접 매핑':<15} {r['mae']:<10.4f} {r['correlation']:<10.4f} {r['cosine_similarity']:<12.4f} {'16x16':<12}")
    
    if results['convolutional']:
        r = results['convolutional']
        print(f"{'컨볼루션':<15} {r['mae']:<10.4f} {r['correlation']:<10.4f} {r['cosine_similarity']:<12.4f} {'32x32':<12}")
    
    if results['matrix']:
        r = results['matrix']
        print(f"{'행렬(Yoon Kim)':<15} {r['mae']:<10.4f} {r['correlation']:<10.4f} {r['cosine_similarity']:<12.4f} {'32x128':<12}")
    
    # 학습 가능성 판정
    print("\n🔍 학습 가능성 분석:")
    for method_name, result in results.items():
        if result is not None:
            mae = result['mae']
            cosine_sim = result['cosine_similarity']
            
            method_display = {'direct': '직접 매핑', 'convolutional': '컨볼루션', 'matrix': '행렬(Yoon Kim)'}[method_name]
            
            if mae < 0.1 and cosine_sim > 0.8:
                status = "✅ 우수한 학습"
                explanation = "모자이크에서 벡터를 정확히 복원"
            elif mae < 0.2 and cosine_sim > 0.6:
                status = "🔄 양호한 학습"
                explanation = "모자이크에서 벡터를 어느정도 복원"
            elif mae < 0.5 and cosine_sim > 0.3:
                status = "⚠️  기본 학습"
                explanation = "기본적인 패턴 학습"
            else:
                status = "❌ 학습 부족"
                explanation = "더 많은 데이터나 다른 접근 필요"
            
            print(f"  {method_display}: {status} ({explanation})")
    
    # 시각화
    visualize_correct_learning_results(results, texts[:3])
    
    return results

def visualize_correct_learning_results(results, sample_texts):
    """올바른 학습 결과 시각화"""
    
    # 유효한 결과만 추출
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ 시각화할 유효한 결과가 없습니다.")
        return
    
    # 그래프 설정
    n_methods = len(valid_results)
    n_samples = min(3, len(sample_texts))
    
    fig, axes = plt.subplots(n_samples * 3, n_methods, figsize=(5*n_methods, 4*n_samples*3))
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    if n_samples == 1:
        axes = axes.reshape(3, -1)
    
    fig.suptitle('Patients 데이터 올바른 Word2Vec 모자이크 학습 결과', fontsize=16, fontweight='bold')
    
    method_names = {'direct': '직접 매핑', 'convolutional': '컨볼루션', 'matrix': '행렬(Yoon Kim)'}
    
    for j, (method_key, result) in enumerate(valid_results.items()):
        method_name = method_names.get(method_key, method_key)
        
        for i in range(n_samples):
            # 1행: 원본 모자이크
            axes[i*3, j].imshow(result['mosaics'][i].squeeze(), cmap='viridis', aspect='auto')
            axes[i*3, j].set_title(f'{method_name}\nMosaic {i+1}', fontsize=10)
            axes[i*3, j].axis('off')
            
            # 2행: 벡터 비교 (원본 vs 복원)
            if i < len(result['y_test']):
                original_vec = result['y_test'][i]
                predicted_vec = result['predictions'][i]
                
                x_pos = np.arange(min(20, len(original_vec)))  # 처음 20개 차원만 표시
                axes[i*3+1, j].plot(x_pos, original_vec[:len(x_pos)], 'b-', label='Original', alpha=0.7)
                axes[i*3+1, j].plot(x_pos, predicted_vec[:len(x_pos)], 'r--', label='Predicted', alpha=0.7)
                axes[i*3+1, j].set_title(f'Vector Comparison {i+1}', fontsize=10)
                axes[i*3+1, j].legend()
                axes[i*3+1, j].grid(True, alpha=0.3)
            
            # 3행: 성능 정보
            axes[i*3+2, j].text(0.1, 0.8, f"Text: {sample_texts[i][:40]}...", 
                               fontsize=8, wrap=True, transform=axes[i*3+2, j].transAxes)
            axes[i*3+2, j].text(0.1, 0.6, f"MAE: {result['mae']:.4f}", 
                               fontsize=10, fontweight='bold', transform=axes[i*3+2, j].transAxes)
            axes[i*3+2, j].text(0.1, 0.4, f"Correlation: {result['correlation']:.4f}", 
                               fontsize=10, fontweight='bold', transform=axes[i*3+2, j].transAxes)
            axes[i*3+2, j].text(0.1, 0.2, f"Cosine Sim: {result['cosine_similarity']:.4f}", 
                               fontsize=10, fontweight='bold', transform=axes[i*3+2, j].transAxes)
            axes[i*3+2, j].set_title(f'Performance {i+1}', fontsize=10)
            axes[i*3+2, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('patients_correct_learning_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 시각화 완료: patients_correct_learning_results.png 저장")

if __name__ == "__main__":
    # 환자 데이터로 올바른 학습 테스트 실행
    results = test_correct_learning_with_patients()
    
    print("\n🎉 Patients 데이터 올바른 Word2Vec 모자이크 학습 테스트 완료!")
    print("이제 모자이크에서 원본 벡터를 정확히 복원하는 능력을 평가했습니다.")
    print("각 방법의 모자이크 → 벡터 복원 성능을 비교할 수 있습니다.")
