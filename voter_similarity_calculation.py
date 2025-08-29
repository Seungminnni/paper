# 유클리드 거리 유사도
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calculate_accuracy_and_distance(client_file, dictionary_file, original_file_client, original_file_dictionary, n=5):
    # 변환된 파일을 읽어옵니다.
    client_data = pd.read_csv(client_file)
    dictionary_data = pd.read_csv(dictionary_file)
    
    # 원본 파일을 읽어옵니다.
    original_client_data = pd.read_csv(original_file_client, encoding='latin-1')
    original_dictionary_data = pd.read_csv(original_file_dictionary, encoding='latin-1')
    
    # 데이터 포인트 간의 유클리드 거리를 계산합니다.
    distances = euclidean_distances(client_data.values, dictionary_data.values)
    
    # Top@n 유사도를 찾습니다.
    topn_similarities = np.argsort(distances, axis=1)[:, :n]
    topn_values = np.sort(distances, axis=1)[:, :n]
    
    # 모든 결과를 출력하고 정확도를 계산합니다.
    successful_distances = []
    unsuccessful_distances = []
    successes = 0
    success_indices = []  # 성공한 인덱스를 저장할 리스트
    success_ranks_count = {rank: 0 for rank in range(1, n+1)}  # 각 성공한 서버 측 랭크의 수를 저장할 딕셔너리
    
    for i, (indices, scores) in enumerate(zip(topn_similarities, topn_values)):
        """print(f"\nTop {n} inferences for client {i + 1}:")"""
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            """print(f"Server {idx + 1} with distance {score}")"""
            if original_client_data.iloc[i].equals(original_dictionary_data.iloc[idx]):
                successes += 1
                successful_distances.append(score)
                success_indices.append((i + 1, rank))  # 성공한 인덱스를 추가
                success_ranks_count[rank] += 1  # 해당 랭크의 수를 증가시킴
            else:
                unsuccessful_distances.append(score)
        if successes == 0:
            print("No successful match found.")
    
    # 정확도 계산
    accuracy = successes / len(client_data)
    
    # 성공적으로 일치하는 데이터 포인트와 클라이언트 데이터 포인트, 그리고 일치하지 않는 데이터 포인트와 클라이언트 데이터 포인트 간의 평균 거리를 계산합니다.
    successful_mean_distance = np.mean(successful_distances) if successful_distances else 0
    unsuccessful_mean_distance = np.mean(unsuccessful_distances) if unsuccessful_distances else 0
    
    # 평균 거리의 분산 계산
    successful_distance_variance = np.var(successful_distances) if successful_distances else 0
    unsuccessful_distance_variance = np.var(unsuccessful_distances) if unsuccessful_distances else 0
    
    return accuracy, successful_mean_distance, unsuccessful_mean_distance, success_indices, successful_distance_variance, unsuccessful_distance_variance, success_ranks_count

# 변환된 파일 경로
dictionary_file = "Dictionary_smashed_data.csv"
client_file = "Client_smashed_data.csv"

# 원본 파일 경로
original_file_client = "ncvoterb.csv"
original_file_dictionary = "ncvoterb.csv"

# Top n 설정
n = 5

# 파일 존재 확인
import os
if not os.path.exists(client_file) or not os.path.exists(dictionary_file):
    print(f"Error: Required files not found!")
    print(f"Client file: {client_file} - {'Found' if os.path.exists(client_file) else 'Not found'}")
    print(f"Dictionary file: {dictionary_file} - {'Found' if os.path.exists(dictionary_file) else 'Not found'}")
else:
    # 정확도 계산 및 평균 거리 계산
    print("🔄 Calculating voter similarity with accuracy...")
    accuracy, successful_mean_distance, unsuccessful_mean_distance, success_indices, successful_distance_variance, unsuccessful_distance_variance, success_ranks_count = calculate_accuracy_and_distance(client_file, dictionary_file, original_file_client, original_file_dictionary, n)
    
    print("\n" + "="*50)
    print("VOTER SMASHED DATA SIMILARITY ANALYSIS")
    print("="*50)
    print("\nFor file:", client_file)
    print("Accuracy:", accuracy)
    print("Successful Mean Distance:", successful_mean_distance)
    print("Unsuccessful Mean Distance:", unsuccessful_mean_distance)
    
    # 분산 출력
    print("Successful Distance Variance:", successful_distance_variance)
    print("Unsuccessful Distance Variance:", unsuccessful_distance_variance)
    
    # 성공한 인덱스들을 출력합니다.
    print("Success Indices:", success_indices)
    
    # 각 성공한 서버 측 랭크의 수를 출력합니다.
    print("Success Ranks Count:")
    for rank, count in success_ranks_count.items():
        print(f"Rank {rank}: {count} successes")
    
    print("\n🎉 Voter similarity analysis completed!")
    print("="*50)