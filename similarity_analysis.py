#!/usr/bin/env python3
"""
Smashed Data Similarity Analysis
클라이언트와 서버 smashed data의 유사도 분석
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def load_smashed_data(file_path):
    """Smashed data CSV 파일 로드"""
    df = pd.read_csv(file_path)
    # 각 행이 하나의 벡터
    vectors = df.values
    print(f"📊 Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]} from {file_path}")
    return vectors

def calculate_similarity_stats(client_vectors, server_vectors):
    """두 데이터셋 간 유사도 통계 계산"""
    print("\n🔍 Calculating similarity statistics...")

    similarities = []

    # 각 클라이언트 벡터에 대해 가장 유사한 서버 벡터 찾기
    for client_vec in client_vectors:
        # 코사인 유사도 계산
        sim_scores = cosine_similarity([client_vec], server_vectors)[0]
        max_sim = np.max(sim_scores)
        similarities.append(max_sim)

    similarities = np.array(similarities)

    print("📈 Similarity Statistics:")
    print(f"   • Mean similarity: {np.mean(similarities):.4f}")
    print(f"   • Median similarity: {np.median(similarities):.4f}")
    print(f"   • Std similarity: {np.std(similarities):.4f}")
    print(f"   • Min similarity: {np.min(similarities):.4f}")
    print(f"   • Max similarity: {np.max(similarities):.4f}")

    return similarities

def plot_similarity_distribution(similarities):
    """유사도 분포 시각화"""
    try:
        plt.figure(figsize=(10, 6))

        # 모든 값이 같으면 특별 처리
        if np.std(similarities) == 0:
            plt.bar([0.5], [len(similarities)], width=0.1, alpha=0.7, color='blue')
            plt.title('All Similarities are Identical (Value = 1.0)')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Count')
            plt.xticks([0.5], ['1.0'])
        else:
            plt.hist(similarities, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Distribution of Cosine Similarities between Client and Server Vectors')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('similarity_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ Plot saved as 'similarity_distribution.png'")
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")
    finally:
        plt.close('all')  # 모든 플롯 닫기

def analyze_vector_clusters(client_vectors, server_vectors):
    """벡터 클러스터 분석"""
    print("\n🎯 Analyzing vector clusters...")

    # 각 데이터셋의 평균 벡터 계산
    client_mean = np.mean(client_vectors, axis=0)
    server_mean = np.mean(server_vectors, axis=0)

    # 평균 벡터 간 유사도
    mean_similarity = cosine_similarity([client_mean], [server_mean])[0][0]
    print(f"📊 Mean vector similarity: {mean_similarity:.4f}")

    # 각 데이터셋 내 분산
    client_variance = np.var(client_vectors, axis=0).mean()
    server_variance = np.var(server_vectors, axis=0).mean()

    print(f"📊 Client data variance: {client_variance:.6f}")
    print(f"📊 Server data variance: {server_variance:.6f}")

    return mean_similarity, client_variance, server_variance

def main():
    print("🔬 Smashed Data Similarity Analysis")
    print("=" * 60)

    # 데이터 로드
    client_file = "Client_smashed_data_layer2.csv"
    server_file = "Dictionary_smashed_data_layer2.csv"

    try:
        client_vectors = load_smashed_data(client_file)
        server_vectors = load_smashed_data(server_file)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 유사도 분석
    similarities = calculate_similarity_stats(client_vectors, server_vectors)

    # 클러스터 분석
    mean_sim, client_var, server_var = analyze_vector_clusters(client_vectors, server_vectors)

    # 시각화 (모든 값이 같으면 생략)
    if np.std(similarities) > 0:
        print("\n📊 Generating similarity distribution plot...")
        plot_similarity_distribution(similarities)
    else:
        print("\n📊 All similarities are identical (1.0) - skipping plot")

    # 결과 요약
    print("\n🎉 Analysis Complete!")
    print("=" * 60)
    print("📋 Summary:")
    print(f"   • Client samples: {client_vectors.shape[0]}")
    print(f"   • Server samples: {server_vectors.shape[0]}")
    print(f"   • Average similarity: {np.mean(similarities):.4f}")
    print(f"   • Mean vector similarity: {mean_sim:.4f}")
    print(f"   • Client variance: {client_var:.6f}")
    print(f"   • Server variance: {server_var:.6f}")

    # 해석
    print("\n💡 Interpretation:")
    if np.mean(similarities) > 0.8:
        print("   • High similarity: Data distributions are very similar")
    elif np.mean(similarities) > 0.6:
        print("   • Moderate similarity: Some overlap in data distributions")
    else:
        print("   • Low similarity: Data distributions are quite different")

    if abs(client_var - server_var) < 0.01:
        print("   • Similar variance: Data spread is comparable")
    else:
        print("   • Different variance: Data spread differs between client and server")

if __name__ == "__main__":
    main()
