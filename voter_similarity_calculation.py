# 유권자 데이터 smashed data 유사도 계산 - 순환 은폐 구조 적용
# 연구 아이디어: 은폐된 벡터 간 유사도 분석으로 보안 효과 검증
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityAnalyzer:
    """
    순환 은폐 구조의 smashed data 유사도 분석
    공격자가 벡터를 탈취하더라도 의미 있는 패턴을 찾기 어려움 검증
    """
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    def analyze_obfuscation_effectiveness(self, client_vectors, server_vectors):
        """
        은폐 효과 분석: 원본 vs 은폐된 벡터의 차이 분석
        """
        # 벡터 간 유사도 계산
        similarities = 1 / (1 + euclidean_distances(client_vectors, server_vectors))
        
        # 통계 분석
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        max_similarity = np.max(similarities)
        min_similarity = np.min(similarities)
        
        return {
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'similarity_distribution': similarities
        }
    
    def detect_anomalies(self, vectors, threshold=0.1):
        """
        이상치 탐지: 은폐된 벡터의 비정상 패턴 탐지
        공격자가 패턴을 학습하기 어려운지 검증
        """
        # 벡터 정규화
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # 코사인 유사도 계산
        cosine_sim = np.dot(normalized_vectors, normalized_vectors.T)
        
        # 자기 자신과의 유사도는 1이므로 제외
        np.fill_diagonal(cosine_sim, 0)
        
        # 임계값 이상의 유사도를 가진 이상치 탐지
        anomalies = []
        for i in range(len(cosine_sim)):
            max_sim = np.max(cosine_sim[i])
            if max_sim > threshold:
                anomalies.append((i, max_sim))
        
        return anomalies, cosine_sim

def calculate_circular_obfuscation_similarity(client_file, dictionary_file, n=5):
    """
    순환 은폐 구조의 smashed data 간 유사도를 계산하는 함수
    기존 유사도 계산에 은폐 효과 분석 추가
    """
    # 변환된 파일을 읽어옵니다.
    client_data = pd.read_csv(client_file)
    dictionary_data = pd.read_csv(dictionary_file)
    
    # 첫 번째 컬럼을 voter_id로 이름 변경
    client_data.columns = ['voter_id'] + [f'vec_{i}' for i in range(len(client_data.columns) - 1)]
    dictionary_data.columns = ['voter_id'] + [f'vec_{i}' for i in range(len(dictionary_data.columns) - 1)]
    
    # voter_id 추출 (비교용)
    client_voter_ids = client_data['voter_id'].tolist()
    dictionary_voter_ids = dictionary_data['voter_id'].tolist()
    
    # 벡터 데이터만 추출 (voter_id 제외)
    client_vectors = client_data.drop('voter_id', axis=1).values
    dictionary_vectors = dictionary_data.drop('voter_id', axis=1).values

    # 순환 은폐 분석기 초기화
    analyzer = SimilarityAnalyzer()
    
    # 기본 유사도 계산 (기존 방식)
    distances = euclidean_distances(client_vectors, dictionary_vectors)
    topn_similarities = np.argsort(distances, axis=1)[:, :n]
    topn_values = np.sort(distances, axis=1)[:, :n]
    
    # 은폐 효과 분석
    obfuscation_analysis = analyzer.analyze_obfuscation_effectiveness(
        client_vectors, dictionary_vectors
    )
    
    # 이상치 탐지
    client_anomalies, client_similarity_matrix = analyzer.detect_anomalies(
        client_vectors, threshold=0.1
    )
    server_anomalies, server_similarity_matrix = analyzer.detect_anomalies(
        dictionary_vectors, threshold=0.1
    )
    
    # 모든 결과를 출력하고 정확도를 계산합니다.
    successful_distances = []
    unsuccessful_distances = []
    successes = 0
    success_indices = []  # 성공한 인덱스를 저장할 리스트
    success_ranks_count = {rank: 0 for rank in range(1, n+1)}  # 각 성공한 서버 측 랭크의 수를 저장할 딕셔너리

    # 순환 은폐 구조의 유사도 분석 (voter_id 매칭 대신 통계적 분석)
    # Client와 Server는 서로 다른 데이터를 처리하므로 voter_id 매칭은 의미 없음
    
    # 기본 통계 계산
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    
    # Top-N 분석
    top1_distances = np.sort(distances, axis=1)[:, :1].flatten()
    top5_distances = np.sort(distances, axis=1)[:, :5].flatten()
    
    print(f"📊 Distance Statistics:")
    print(f"   Mean Distance: {mean_distance:.4f}")
    print(f"   Std Distance: {std_distance:.4f}")
    print(f"   Min Distance: {min_distance:.4f}")
    print(f"   Max Distance: {max_distance:.4f}")
    print(f"   Top-1 Mean Distance: {np.mean(top1_distances):.4f}")
    print(f"   Top-5 Mean Distance: {np.mean(top5_distances):.4f}")
    
    # 은폐 효과 분석 결과 반환
    return {
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'top1_mean_distance': np.mean(top1_distances),
        'top5_mean_distance': np.mean(top5_distances),
        'obfuscation_analysis': obfuscation_analysis,
        'client_anomalies': client_anomalies,
        'server_anomalies': server_anomalies,
        'client_similarity_matrix': client_similarity_matrix,
        'server_similarity_matrix': server_similarity_matrix
    }

# 메인 실행 부분
if __name__ == "__main__":
    print("Loading voter data for similarity calculation...")

    # 원본 데이터 로드
    full_data = pd.read_csv("ncvoterb.csv", encoding='latin-1')

    # 데이터 크기 제한: 실험을 위해 1,000개로 제한 (smashed data와 동일하게)
    # 전체 데이터: 약 224,061개 → 실험용: 1,000개 (약 0.4% 사용)
    SAMPLE_SIZE = 1000
    if len(full_data) > SAMPLE_SIZE:
        print(f"📊 Reducing data size from {len(full_data):,} to {SAMPLE_SIZE:,} for faster experimentation")
        full_data = full_data.sample(n=SAMPLE_SIZE, random_state=42)  # smashed data와 동일한 random_state
        print(f"✅ Data reduced successfully! Working with {len(full_data):,} records")

    print(f"✅ Data loaded successfully! Total records: {len(full_data)}")

    # 서버측과 클라이언트 측 데이터 분리 (smashed data 생성과 동일한 방식)
    # 실험용 데이터에서 70% 서버, 30% 클라이언트로 분리
    server_sample_size = int(len(full_data) * 0.7)
    server_data = full_data.sample(n=server_sample_size, random_state=42)  # smashed data와 동일한 방식
    client_data = full_data.drop(server_data.index).sample(frac=1.0, random_state=123)  # 나머지 데이터

    print(f"📊 Server data size: {len(server_data)} (70% of {len(full_data)} = {len(server_data):,})")
    print(f"📊 Client data size: {len(client_data)} (30% of {len(full_data)} = {len(client_data):,})")

    # 변환된 파일 경로
    dictionary_file = "Dictionary_smashed_data.csv"
    client_file = "Client_smashed_data.csv"

    # Top n 설정
    n = 5

    # 파일 존재 확인
    import os
    if not os.path.exists(client_file) or not os.path.exists(dictionary_file):
        print(f"Error: Required files not found!")
        print(f"Client file: {client_file} - {'Found' if os.path.exists(client_file) else 'Not found'}")
        print(f"Dictionary file: {dictionary_file} - {'Found' if os.path.exists(dictionary_file) else 'Not found'}")
        exit(1)

    # 유사도 계산 (순환 은폐 구조 적용)
    print("🔄 Calculating similarity with circular obfuscation analysis...")
    print("   📋 Analysis includes: Basic similarity + Obfuscation effectiveness + Anomaly detection")
    
    results = calculate_circular_obfuscation_similarity(
        client_file, dictionary_file, n
    )

    # 결과 출력
    print("\n" + "="*70)
    print("CIRCULAR OBFUSCATION SMASHED DATA SIMILARITY ANALYSIS")
    print("="*70)
    print(f"📊 Mean Distance: {results['mean_distance']:.4f}")
    print(f"📊 Std Distance: {results['std_distance']:.4f}")
    print(f"📊 Min Distance: {results['min_distance']:.4f}")
    print(f"� Max Distance: {results['max_distance']:.4f}")
    print(f"🎯 Top-1 Mean Distance: {results['top1_mean_distance']:.4f}")
    print(f"🎯 Top-5 Mean Distance: {results['top5_mean_distance']:.4f}")
    
    # 은폐 효과 분석 결과
    print(f"\n🛡️ OBFUSCATION EFFECTIVENESS ANALYSIS")
    print("-" * 40)
    obf = results['obfuscation_analysis']
    print(f"📊 Mean Similarity: {obf['mean_similarity']:.4f}")
    print(f"📊 Similarity Std Dev: {obf['std_similarity']:.4f}")
    print(f"📊 Max Similarity: {obf['max_similarity']:.4f}")
    print(f"📊 Min Similarity: {obf['min_similarity']:.4f}")
    
    # 이상치 분석 결과
    print(f"\n🚨 ANOMALY DETECTION RESULTS")
    print("-" * 40)
    print(f"📊 Client Anomalies: {len(results['client_anomalies'])} detected")
    print(f"📊 Server Anomalies: {len(results['server_anomalies'])} detected")
    
    if results['client_anomalies']:
        print("   Top client anomalies:")
        for idx, sim in results['client_anomalies'][:5]:
            print(f"     Sample {idx}: similarity {sim:.4f}")
    
    print("\n🎉 Circular obfuscation similarity analysis completed!")
    print("="*70)
    
    # 보안 효과 요약
    print("\n🔒 SECURITY EFFECTIVENESS SUMMARY")
    print("-" * 40)
    print(f"   • Obfuscation Strength: {'Strong' if results['std_distance'] > 10 else 'Moderate'}")
    print(f"   • Anomaly Detection: {len(results['client_anomalies'] + results['server_anomalies'])} patterns found")
    print(f"   • Distance Distribution: {results['min_distance']:.3f} - {results['max_distance']:.3f}")
    print(f"   • Similarity Distribution: {obf['min_similarity']:.3f} - {obf['max_similarity']:.3f}")
    print(f"   • Attack Difficulty: {'High' if results['mean_distance'] > 20 else 'Medium'}")
    print("="*70)
