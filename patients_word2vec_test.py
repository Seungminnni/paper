#!/usr/bin/env python3
"""
Patients.csv 데이터로 Word2Vec 모자이크 테스트
- 환자 정보 텍스트를 Word2Vec으로 벡터화
- 세 가지 방법으로 모자이크 생성 및 학습 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_word2vec_mosaic import *
import pandas as pd

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

def test_all_methods_with_patients():
    """환자 데이터로 세 가지 방법 모두 테스트"""
    
    # 데이터 로드
    texts = load_patients_data()
    
    # 샘플 수 제한 (빠른 테스트를 위해)
    sample_size = min(500, len(texts))
    texts = texts[:sample_size]
    print(f"\n🎯 테스트 샘플: {len(texts)}개")
    
    print("\n" + "="*60)
    print("🧪 PATIENTS 데이터 Word2Vec 모자이크 비교 테스트")
    print("="*60)
    
    # 세 가지 방법으로 테스트
    results = {}
    
    # 1. 직접 매핑 방식
    print("\n🔹 1. 직접 매핑 방식 테스트")
    print("-" * 40)
    try:
        direct_processor = SimpleWord2VecProcessor(vector_size=256)
        direct_generator = DirectMosaicGenerator(vector_size=256)
        
        vectors = direct_processor.train_and_vectorize(texts)
        mosaics = direct_generator.vectors_to_mosaics(vectors)
        
        # 학습 테스트 (오토인코더 방식)
        X_train, X_test = train_test_split(mosaics, test_size=0.2, random_state=42)
        cnn = SimpleCNN(image_size=16)
        history = cnn.train(X_train, X_test, epochs=20)
        
        # 예측 및 평가
        predictions = cnn.model.predict(X_test, verbose=0)
        mae = np.mean(np.abs(X_test - predictions))
        correlation = np.corrcoef(X_test.flatten(), predictions.flatten())[0, 1]
        
        results['direct'] = {
            'mae': mae,
            'correlation': correlation,
            'mosaics': mosaics,
            'vectors': vectors,
            'history': history
        }
        print(f"✅ 직접 매핑 완료 - MAE: {mae:.4f}, 상관관계: {correlation:.4f}")
        
    except Exception as e:
        print(f"❌ 직접 매핑 오류: {e}")
        results['direct'] = None
    
    # 2. 컨볼루션 방식
    print("\n🔹 2. 컨볼루션 방식 테스트")
    print("-" * 40)
    try:
        conv_processor = SimpleWord2VecProcessor(vector_size=256)
        conv_generator = ConvolutionalMosaicGenerator(vector_size=256, final_image_size=32)
        
        vectors = conv_processor.train_and_vectorize(texts)
        mosaics = conv_generator.vectors_to_mosaics(vectors)
        
        # 학습 테스트 (오토인코더 방식)
        X_train, X_test = train_test_split(mosaics, test_size=0.2, random_state=42)
        cnn = SimpleCNN(image_size=32)
        history = cnn.train(X_train, X_test, epochs=20)
        
        # 예측 및 평가
        predictions = cnn.model.predict(X_test, verbose=0)
        mae = np.mean(np.abs(X_test - predictions))
        correlation = np.corrcoef(X_test.flatten(), predictions.flatten())[0, 1]
        
        results['convolutional'] = {
            'mae': mae,
            'correlation': correlation,
            'mosaics': mosaics,
            'vectors': vectors,
            'history': history
        }
        print(f"✅ 컨볼루션 방식 완료 - MAE: {mae:.4f}, 상관관계: {correlation:.4f}")
        
    except Exception as e:
        print(f"❌ 컨볼루션 방식 오류: {e}")
        results['convolutional'] = None
    
    # 3. 행렬 방식 (Yoon Kim 스타일)
    print("\n🔹 3. 행렬 방식 (Yoon Kim 스타일) 테스트")
    print("-" * 40)
    try:
        matrix_processor = SentenceMatrixProcessor(vector_size=128, max_words=32)
        matrix_generator = MatrixMosaicGenerator(matrix_shape=(32, 128))
        
        sentence_matrices = matrix_processor.train_and_vectorize(texts)
        mosaics = matrix_generator.matrices_to_mosaics(sentence_matrices)
        
        # 벡터화 (평균)
        flattened_matrices = sentence_matrices.reshape(len(sentence_matrices), -1)
        
        # 학습 테스트 (오토인코더 방식)
        X_train, X_test = train_test_split(mosaics, test_size=0.2, random_state=42)
        cnn = MatrixCNN(input_shape=(32, 128, 1))
        history = cnn.train(X_train, X_test, epochs=20)
        
        # 예측 및 평가
        predictions = cnn.model.predict(X_test, verbose=0)
        mae = np.mean(np.abs(X_test - predictions))
        correlation = np.corrcoef(X_test.flatten(), predictions.flatten())[0, 1]
        
        results['matrix'] = {
            'mae': mae,
            'correlation': correlation,
            'mosaics': mosaics,
            'vectors': flattened_matrices,
            'history': history
        }
        print(f"✅ 행렬 방식 완료 - MAE: {mae:.4f}, 상관관계: {correlation:.4f}")
        
    except Exception as e:
        print(f"❌ 행렬 방식 오류: {e}")
        results['matrix'] = None
    
    # 결과 비교 및 시각화
    print("\n" + "="*60)
    print("📊 PATIENTS 데이터 최종 결과 비교")
    print("="*60)
    
    # 성능 비교 테이블
    print("\n🏆 성능 비교:")
    print(f"{'방법':<15} {'MAE':<10} {'상관관계':<10} {'이미지 크기':<15}")
    print("-" * 50)
    
    if results['direct']:
        print(f"{'직접 매핑':<15} {results['direct']['mae']:<10.4f} {results['direct']['correlation']:<10.4f} {'16x16':<15}")
    
    if results['convolutional']:
        print(f"{'컨볼루션':<15} {results['convolutional']['mae']:<10.4f} {results['convolutional']['correlation']:<10.4f} {'32x32':<15}")
    
    if results['matrix']:
        print(f"{'행렬(Yoon Kim)':<15} {results['matrix']['mae']:<10.4f} {results['matrix']['correlation']:<10.4f} {'32x128':<15}")
    
    # 시각화
    visualize_patients_comparison(results, sample_texts=texts[:3])
    
    return results

def visualize_patients_comparison(results, sample_texts):
    """환자 데이터 결과 시각화"""
    
    # 유효한 결과만 추출
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ 시각화할 유효한 결과가 없습니다.")
        return
    
    # 그래프 설정
    n_methods = len(valid_results)
    n_samples = min(3, len(sample_texts))
    
    fig, axes = plt.subplots(n_samples * 2, n_methods, figsize=(4*n_methods, 3*n_samples*2))
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    if n_samples == 1:
        axes = axes.reshape(2, -1)
    
    fig.suptitle('Patients 데이터 Word2Vec 모자이크 비교', fontsize=16, fontweight='bold')
    
    method_names = {'direct': '직접 매핑', 'convolutional': '컨볼루션', 'matrix': '행렬(Yoon Kim)'}
    
    for j, (method_key, result) in enumerate(valid_results.items()):
        method_name = method_names.get(method_key, method_key)
        
        for i in range(n_samples):
            # 원본 모자이크
            axes[i*2, j].imshow(result['mosaics'][i].squeeze(), cmap='viridis', aspect='auto')
            axes[i*2, j].set_title(f'{method_name}\nOriginal {i+1}', fontsize=10)
            axes[i*2, j].axis('off')
            
            # 텍스트 정보 (복원된 이미지 대신)
            axes[i*2+1, j].text(0.1, 0.7, f"Text: {sample_texts[i][:50]}...", 
                               fontsize=8, wrap=True, transform=axes[i*2+1, j].transAxes)
            axes[i*2+1, j].text(0.1, 0.5, f"MAE: {result['mae']:.4f}", 
                               fontsize=10, fontweight='bold', transform=axes[i*2+1, j].transAxes)
            axes[i*2+1, j].text(0.1, 0.3, f"Correlation: {result['correlation']:.4f}", 
                               fontsize=10, fontweight='bold', transform=axes[i*2+1, j].transAxes)
            axes[i*2+1, j].text(0.1, 0.1, f"Shape: {result['mosaics'].shape[1:]}", 
                               fontsize=8, transform=axes[i*2+1, j].transAxes)
            axes[i*2+1, j].set_title(f'Info {i+1}', fontsize=10)
            axes[i*2+1, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('patients_word2vec_mosaic_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 시각화 완료: patients_word2vec_mosaic_comparison.png 저장")

if __name__ == "__main__":
    # 환자 데이터로 테스트 실행
    results = test_all_methods_with_patients()
    
    print("\n🎉 Patients 데이터 Word2Vec 모자이크 테스트 완료!")
    print("각 방법은 환자 정보 텍스트마다 고유한 모자이크를 생성하며,")
    print("컨볼루션을 통한 차원 압축으로 벡터→이미지 매핑을 수행합니다.")
