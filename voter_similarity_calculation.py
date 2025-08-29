# 유권자 데이터 smashed data 유사도 계산
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calculate_voter_similarity(client_file, dictionary_file, original_client_data, original_dictionary_data, n=5):
    """
    유권자 데이터의 smashed data 간 유사도를 계산하는 함수

    Parameters:
    - client_file: 클라이언트 측 smashed data 파일 경로
    - dictionary_file: 서버 측 smashed data 파일 경로
    - original_client_data: 원본 클라이언트 데이터 (DataFrame)
    - original_dictionary_data: 원본 서버 데이터 (DataFrame)
    - n: Top-N 유사도 계산 개수
    """
    # 변환된 파일을 읽어옵니다.
    client_data = pd.read_csv(client_file)
    dictionary_data = pd.read_csv(dictionary_file)

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
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            # 원본 데이터의 voter_id로 비교
            client_voter_id = original_client_data.iloc[i]["voter_id"]
            dictionary_voter_id = original_dictionary_data.iloc[idx]["voter_id"]

            if client_voter_id == dictionary_voter_id:
                successes += 1
                successful_distances.append(score)
                success_indices.append((i + 1, rank))  # 성공한 인덱스를 추가
                success_ranks_count[rank] += 1  # 해당 랭크의 수를 증가시킴
                break  # 같은 voter_id를 찾았으므로 더 이상 확인하지 않음
            else:
                unsuccessful_distances.append(score)

    # 정확도 계산
    accuracy = successes / len(client_data)

    # 성공적으로 일치하는 데이터 포인트와 일치하지 않는 데이터 포인트 간의 평균 거리를 계산합니다.
    successful_mean_distance = np.mean(successful_distances) if successful_distances else 0
    unsuccessful_mean_distance = np.mean(unsuccessful_distances) if unsuccessful_distances else 0

    # 평균 거리의 분산 계산
    successful_distance_variance = np.var(successful_distances) if successful_distances else 0
    unsuccessful_distance_variance = np.var(unsuccessful_distances) if unsuccessful_distances else 0

    return accuracy, successful_mean_distance, unsuccessful_mean_distance, success_indices, successful_distance_variance, unsuccessful_distance_variance, success_ranks_count

# 메인 실행 부분
if __name__ == "__main__":
    print("Loading voter data for similarity calculation...")

    # 원본 데이터 로드
    full_data = pd.read_csv("ncvoterb.csv")

    # 서버측과 클라이언트 측 데이터 분리 (이전과 동일한 방식)
    server_sample_size = int(len(full_data) * 0.7)
    server_data = full_data.sample(n=server_sample_size, random_state=42)
    client_data = full_data.drop(server_data.index).sample(frac=1.0, random_state=123)  # 나머지 데이터 사용

    print(f"Server data size: {len(server_data)}")
    print(f"Client data size: {len(client_data)}")

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

    # 유사도 계산
    print("Calculating similarity between smashed data...")
    accuracy, successful_mean_distance, unsuccessful_mean_distance, success_indices, successful_distance_variance, unsuccessful_distance_variance, success_ranks_count = calculate_voter_similarity(
        client_file, dictionary_file, client_data, server_data, n
    )

    # 결과 출력
    print("\n" + "="*50)
    print("VOTER DATA SMASHED DATA SIMILARITY RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Successful Mean Distance: {successful_mean_distance:.4f}")
    print(f"Unsuccessful Mean Distance: {unsuccessful_mean_distance:.4f}")
    print(f"Successful Distance Variance: {successful_distance_variance:.4f}")
    print(f"Unsuccessful Distance Variance: {unsuccessful_distance_variance:.4f}")

    print(f"\nSuccess Indices (Top {n}): {len(success_indices)} matches found")
    for idx, rank in success_indices[:10]:  # 처음 10개만 출력
        print(f"  Client {idx} matched at rank {rank}")

    print(f"\nSuccess Ranks Count:")
    for rank, count in success_ranks_count.items():
        print(f"  Rank {rank}: {count} successes")

    print("\nSimilarity calculation completed!")
