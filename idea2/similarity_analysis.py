#!/usr/bin/env python3
"""
Pipeline Information-Loss Analysis
(Before) Original vs (After) Restored Smashed Data
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

def calculate_direct_similarity(original_vectors, restored_vectors):
    """두 데이터셋의 각 벡터를 1:1로 비교하여 유사도 통계 계산"""
    print("\n🔍 Calculating 1-to-1 similarity statistics...")

    # 각 쌍의 코사인 유사도를 직접 계산
    similarities = [
        cosine_similarity([original], [restored])[0][0] 
        for original, restored in zip(original_vectors, restored_vectors)
    ]

    similarities = np.array(similarities)

    print("📈 Similarity Statistics (Original vs. Restored):")
    print(f"   • Mean similarity: {np.mean(similarities):.6f}")
    print(f"   • Median similarity: {np.median(similarities):.6f}")
    print(f"   • Std Dev similarity: {np.std(similarities):.6f}")
    print(f"   • Min similarity: {np.min(similarities):.6f} (Most information loss)")
    print(f"   • Max similarity: {np.max(similarities):.6f} (Least information loss)")

    return similarities

def plot_similarity_distribution(similarities):
    """유사도 분포 시각화"""
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, alpha=0.75, color='green', edgecolor='black')
    plt.title('Distribution of Cosine Similarities (Original vs. Restored Vectors)')
    plt.xlabel('Cosine Similarity (1.0 = Perfect Reconstruction)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('information_loss_distribution.png', dpi=300)
    print("\n✅ Plot saved as 'information_loss_distribution.png'")
    plt.close()

def main():
    print("🔬 Pipeline Information-Loss Analysis")
    print("=" * 60)

    # 데이터 로드
    original_file = "Client_smashed_data_layer2.csv"
    restored_file = "restored_client_vectors.csv"

    try:
        original_vectors = load_smashed_data(original_file)
        restored_vectors = load_smashed_data(restored_file)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run the full pipeline (client_smashed_data_generation.py -> ... -> server_side.py) first.")
        return

    if len(original_vectors) != len(restored_vectors):
        print("❌ Error: The number of vectors in both files do not match.")
        return

    # 1:1 유사도 분석
    similarities = calculate_direct_similarity(original_vectors, restored_vectors)

    if similarities is not None:
        # 시각화
        plot_similarity_distribution(similarities)

        # 결과 요약
        print("\n🎉 Analysis Complete!")
        print("=" * 60)
        print("📋 Summary of Pipeline Information Loss:")
        print(f"   • Compared {len(similarities)} pairs of vectors.")
        print(f"   • Average Cosine Similarity: {np.mean(similarities):.6f}")
        print(f"   • Minimum Similarity (Worst Case): {np.min(similarities):.6f}")

        # 해석
        print("\n💡 Interpretation:")
        mean_sim = np.mean(similarities)
        if mean_sim > 0.9999:
            print("   • Excellent: The image conversion pipeline causes negligible information loss.")
        elif mean_sim > 0.99:
            print("   • Good: The pipeline causes very minor information loss.")
        elif mean_sim > 0.95:
            print("   • Moderate: The pipeline causes some information loss, which might affect performance.")
        else:
            print("   • High: The pipeline significantly distorts the data.")

if __name__ == "__main__":
    main()
