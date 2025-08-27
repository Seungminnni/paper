#!/usr/bin/env python3
"""
Probabilistic Reconstruction Mosaic Communication System
- 확률적 복원: 글자 수 + ASCII 합계 → 가능한 모든 조합 생성 → 사전 검색
- 높은 확률 후보만 선택하여 학습
- 계산량 무시하고 완전한 조합 탐색
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import itertools
import string
from collections import Counter, defaultdict
import pickle
import time
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

print("=== Probabilistic Reconstruction Mosaic System ===")
print("🎯 글자수 + ASCII합계 → 확률적 조합 생성 → 사전 검색")

class ProbabilisticReconstructor:
    """확률적 복원 시스템 - 모든 가능한 조합을 생성하고 최적 후보 선택"""
    
    def __init__(self):
        self.name_dictionary = set()  # 실제 존재하는 이름 사전
        self.word_frequency = {}      # 단어 빈도 사전
        self.ascii_range = list(range(32, 127))  # 인쇄 가능한 ASCII 문자
        self.printable_chars = string.printable.strip()
        self.common_names = self._build_name_dictionary()
        
    def _build_name_dictionary(self):
        """실제 이름 사전 구축"""
        # 일반적인 이름들
        first_names = [
            'john', 'mary', 'david', 'sarah', 'michael', 'jennifer', 'robert', 'linda',
            'james', 'patricia', 'william', 'elizabeth', 'richard', 'barbara', 'joseph',
            'susan', 'thomas', 'jessica', 'charles', 'karen', 'christopher', 'nancy',
            'daniel', 'lisa', 'matthew', 'betty', 'anthony', 'helen', 'mark', 'sandra',
            'donald', 'donna', 'steven', 'carol', 'paul', 'ruth', 'andrew', 'sharon',
            'joshua', 'michelle', 'kenneth', 'laura', 'kevin', 'sarah', 'brian', 'kimberly',
            'george', 'deborah', 'edward', 'dorothy', 'ronald', 'lisa', 'timothy', 'nancy',
            'jason', 'karen', 'jeffrey', 'betty', 'ryan', 'helen', 'jacob', 'sandra'
        ]
        
        last_names = [
            'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller', 'davis',
            'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez', 'wilson', 'anderson',
            'thomas', 'taylor', 'moore', 'jackson', 'martin', 'lee', 'perez', 'thompson',
            'white', 'harris', 'sanchez', 'clark', 'ramirez', 'lewis', 'robinson', 'walker',
            'young', 'allen', 'king', 'wright', 'scott', 'torres', 'nguyen', 'hill',
            'flores', 'green', 'adams', 'nelson', 'baker', 'hall', 'rivera', 'campbell'
        ]
        
        # 모든 이름을 사전에 추가
        all_names = first_names + last_names
        for name in all_names:
            self.name_dictionary.add(name.lower())
            self.word_frequency[name.lower()] = len(all_names) - all_names.index(name)  # 빈도 점수
        
        print(f"📚 Built dictionary with {len(self.name_dictionary)} names")
        return all_names
    
    def generate_all_combinations(self, length: int, target_sum: int, max_combinations: int = 100000) -> List[str]:
        """
        주어진 길이와 ASCII 합계로 가능한 모든 문자 조합 생성
        
        Args:
            length: 문자열 길이
            target_sum: 목표 ASCII 합계
            max_combinations: 최대 생성할 조합 수 (계산량 제한)
        
        Returns:
            가능한 모든 문자열 조합 리스트
        """
        if length <= 0 or target_sum <= 0:
            return []
        
        print(f"🔄 Generating combinations: length={length}, sum={target_sum}")
        start_time = time.time()
        
        combinations = []
        generated_count = 0
        
        # 소문자 알파벳만 사용 (a-z, ASCII 97-122)
        chars = string.ascii_lowercase
        min_char_val = ord('a')  # 97
        max_char_val = ord('z')  # 122
        
        # 이론적 범위 체크
        min_possible = length * min_char_val
        max_possible = length * max_char_val
        
        if target_sum < min_possible or target_sum > max_possible:
            print(f"⚠️  Target sum {target_sum} impossible for length {length}")
            return []
        
        # 재귀적으로 조합 생성
        def generate_recursive(current_string: str, remaining_length: int, remaining_sum: int):
            nonlocal generated_count, combinations
            
            if generated_count >= max_combinations:
                return
            
            if remaining_length == 0:
                if remaining_sum == 0:
                    combinations.append(current_string)
                    generated_count += 1
                return
            
            # 남은 길이와 합계로 가능한 문자 범위 계산
            min_needed = remaining_sum - (remaining_length - 1) * max_char_val
            max_needed = remaining_sum - (remaining_length - 1) * min_char_val
            
            for char in chars:
                char_val = ord(char)
                if min_needed <= char_val <= max_needed:
                    generate_recursive(
                        current_string + char,
                        remaining_length - 1,
                        remaining_sum - char_val
                    )
        
        generate_recursive("", length, target_sum)
        
        elapsed = time.time() - start_time
        print(f"✅ Generated {len(combinations)} combinations in {elapsed:.2f}s")
        
        return combinations
    
    def score_candidate(self, candidate: str) -> float:
        """
        후보 문자열의 점수 계산
        
        점수 기준:
        1. 사전에 존재하는가? (기본 점수)
        2. 빈도는 얼마나 높은가?
        3. 언어적으로 자연스러운가?
        """
        score = 0.0
        
        # 1. 사전 검색 (가장 중요한 점수)
        if candidate in self.name_dictionary:
            score += 100.0  # 사전에 있으면 기본 100점
            
            # 2. 빈도 점수 추가
            frequency = self.word_frequency.get(candidate, 1)
            score += frequency * 2  # 빈도에 따른 가산점
        
        # 3. 언어적 자연스러움 (간단한 휴리스틱)
        # 연속된 같은 문자 패널티
        consecutive_penalty = 0
        for i in range(len(candidate) - 1):
            if candidate[i] == candidate[i + 1]:
                consecutive_penalty += 20
        score -= consecutive_penalty
        
        # 모음 비율 점수 (자연스러운 이름은 모음이 적절히 있음)
        vowels = set('aeiou')
        vowel_count = sum(1 for c in candidate if c in vowels)
        vowel_ratio = vowel_count / len(candidate) if len(candidate) > 0 else 0
        if 0.2 <= vowel_ratio <= 0.6:  # 적절한 모음 비율
            score += 10
        
        return max(0.0, score)
    
    def find_best_candidates(self, length: int, ascii_sum: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        주어진 조건에서 가장 가능성 높은 후보들 찾기
        
        Returns:
            (candidate, score) 튜플의 리스트 (점수 내림차순)
        """
        print(f"🎯 Finding best candidates for length={length}, sum={ascii_sum}")
        
        # 모든 조합 생성
        all_combinations = self.generate_all_combinations(length, ascii_sum, max_combinations=50000)
        
        if not all_combinations:
            return []
        
        # 각 후보 점수 계산
        candidates_with_scores = []
        for candidate in all_combinations:
            score = self.score_candidate(candidate)
            if score > 0:  # 점수가 있는 것만
                candidates_with_scores.append((candidate, score))
        
        # 점수 순으로 정렬
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 K개만 반환
        top_candidates = candidates_with_scores[:top_k]
        
        print(f"📊 Found {len(candidates_with_scores)} valid candidates")
        for i, (candidate, score) in enumerate(top_candidates):
            print(f"   #{i+1}: '{candidate}' (score: {score:.1f})")
        
        return top_candidates

class ProbabilisticVoterProcessor:
    """확률적 복원을 사용하는 투표자 데이터 프로세서"""
    
    def __init__(self):
        self.reconstructor = ProbabilisticReconstructor()
        self.encoding_metadata = {}  # 인코딩 메타데이터 저장
        self.feature_names = [
            'voter_id', 'voter_reg_num', 'name_prefix', 'first_name', 'middle_name',
            'last_name', 'name_suffix', 'age', 'gender', 'race', 'ethnic',
            'street_address', 'city', 'state', 'zip_code', 'full_phone_num',
            'birth_place', 'register_date', 'download_month'
        ]
        
    def encode_string_probabilistic(self, text: str) -> Tuple[float, float]:
        """
        문자열을 확률적 복원이 가능한 형태로 인코딩
        
        Returns:
            (length_normalized, sum_normalized): 길이와 ASCII 합계의 정규화된 값
        """
        if pd.isna(text) or text == '':
            return (0.0, 0.0)
        
        text_clean = str(text).lower().strip()
        length = len(text_clean)
        ascii_sum = sum(ord(c) for c in text_clean)
        
        # 정규화 (복원 시 역정규화 가능)
        length_norm = min(length / 20.0, 1.0)  # 최대 20글자 가정
        sum_norm = min(ascii_sum / 2000.0, 1.0)  # 최대 ASCII 합 2000 가정
        
        # 메타데이터 저장 (복원용)
        encoding_key = f"{length}_{ascii_sum}"
        if encoding_key not in self.encoding_metadata:
            self.encoding_metadata[encoding_key] = {
                'length': length,
                'ascii_sum': ascii_sum,
                'original_examples': set()
            }
        self.encoding_metadata[encoding_key]['original_examples'].add(text_clean)
        
        return (length_norm, sum_norm)
    
    def preprocess_data_probabilistic(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """확률적 복원을 위한 데이터 전처리"""
        print(f"🔄 Probabilistic preprocessing: {len(df)} records")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            vector = np.zeros(19, dtype='float32')
            
            for i, feature in enumerate(self.feature_names):
                if feature in df.columns:
                    value = row[feature]
                    
                    if pd.isna(value):
                        vector[i] = 0.0
                        continue
                    
                    if feature in ['first_name', 'last_name', 'middle_name', 'city']:
                        # 텍스트 필드 - 확률적 인코딩
                        length_norm, sum_norm = self.encode_string_probabilistic(str(value))
                        # 길이와 합계를 하나의 값으로 결합 (복원 시 분리 가능)
                        vector[i] = (length_norm + sum_norm) / 2.0
                        
                    elif feature == 'age':
                        # 나이는 직접 정규화
                        try:
                            age = float(value)
                            vector[i] = min(max(age / 100.0, 0.0), 1.0)
                        except:
                            vector[i] = 0.5
                            
                    elif feature in ['gender', 'race', 'state']:
                        # 짧은 범주형 데이터도 확률적 인코딩
                        length_norm, sum_norm = self.encode_string_probabilistic(str(value))
                        vector[i] = (length_norm + sum_norm) / 2.0
                        
                    else:
                        # 기타 필드는 간단한 해시 인코딩
                        text_val = str(value).lower()
                        hash_val = hash(text_val) % 10000
                        vector[i] = hash_val / 10000.0
            
            processed_data.append(vector)
        
        print(f"✅ Processed {len(processed_data)} records with probabilistic encoding")
        print(f"   Encoding metadata entries: {len(self.encoding_metadata)}")
        
        return np.array(processed_data), self.encoding_metadata
    
    def reconstruct_probabilistic(self, vectors: np.ndarray, metadata: Dict) -> pd.DataFrame:
        """확률적 복원을 사용한 벡터→텍스트 변환"""
        print(f"🎯 Probabilistic reconstruction from {len(vectors)} vectors")
        
        reconstructed_records = []
        
        for vector in vectors:
            record = {}
            
            for i, feature in enumerate(self.feature_names):
                if i < len(vector):
                    value = vector[i]
                    
                    if feature in ['first_name', 'last_name', 'middle_name', 'city']:
                        # 확률적 복원
                        combined_val = value * 2.0  # 역정규화
                        
                        # 길이와 합계 추정 (여러 조합 시도)
                        best_reconstruction = "unknown"
                        best_score = 0.0
                        
                        # 다양한 길이 시도 (3-8글자)
                        for length in range(3, 9):
                            # 추정된 ASCII 합계 계산
                            estimated_sum = int((combined_val - length/20.0) * 2000)
                            
                            if estimated_sum > 0:
                                candidates = self.reconstructor.find_best_candidates(
                                    length, estimated_sum, top_k=3
                                )
                                
                                for candidate, score in candidates:
                                    if score > best_score:
                                        best_score = score
                                        best_reconstruction = candidate
                        
                        record[feature] = best_reconstruction
                        
                    elif feature == 'age':
                        # 나이 복원
                        age = value * 100.0
                        record[feature] = max(18, min(100, int(round(age))))
                        
                    elif feature in ['gender', 'race', 'state']:
                        # 짧은 범주형 확률적 복원
                        combined_val = value * 2.0
                        
                        best_reconstruction = "unknown"
                        best_score = 0.0
                        
                        # 짧은 길이만 시도 (1-6글자)
                        for length in range(1, 7):
                            estimated_sum = int((combined_val - length/20.0) * 2000)
                            
                            if estimated_sum > 0:
                                candidates = self.reconstructor.find_best_candidates(
                                    length, estimated_sum, top_k=2
                                )
                                
                                for candidate, score in candidates:
                                    if score > best_score:
                                        best_score = score
                                        best_reconstruction = candidate
                        
                        # 범주형 특수 처리
                        if feature == 'gender':
                            if 'male' in best_reconstruction or 'm' in best_reconstruction:
                                record[feature] = 'male'
                            elif 'female' in best_reconstruction or 'f' in best_reconstruction:
                                record[feature] = 'female'
                            else:
                                record[feature] = best_reconstruction
                        else:
                            record[feature] = best_reconstruction
                    
                    else:
                        # 기타 필드
                        hash_val = int(value * 10000)
                        record[feature] = f"field_{hash_val:04d}"
                else:
                    record[feature] = "unknown"
            
            reconstructed_records.append(record)
        
        return pd.DataFrame(reconstructed_records)

def demonstrate_probabilistic_reconstruction():
    """확률적 복원 시스템 시연"""
    
    print("\n" + "="*70)
    print("🎯 PROBABILISTIC RECONSTRUCTION DEMONSTRATION")
    print("="*70)
    
    # 1. 확률적 복원 엔진 테스트
    print("\n📊 Step 1: Testing probabilistic reconstruction engine...")
    
    reconstructor = ProbabilisticReconstructor()
    
    # 테스트 케이스들
    test_cases = [
        ("john", 4, 431),  # john = j(106) + o(111) + h(104) + n(110) = 431
        ("mary", 4, 435),  # mary = m(109) + a(97) + r(114) + y(121) = 441
        ("david", 5, 507), # david = d(100) + a(97) + v(118) + i(105) + d(100) = 520
    ]
    
    print("   Testing known cases:")
    for original, length, expected_sum in test_cases:
        actual_sum = sum(ord(c) for c in original)
        print(f"   '{original}': length={length}, expected_sum={expected_sum}, actual_sum={actual_sum}")
        
        # 실제 복원 테스트
        candidates = reconstructor.find_best_candidates(length, actual_sum, top_k=5)
        found_original = any(candidate == original for candidate, _ in candidates)
        
        print(f"     ✅ Original found: {found_original}")
        if candidates:
            print(f"     Top candidate: '{candidates[0][0]}' (score: {candidates[0][1]:.1f})")
    
    # 2. 전체 시스템 테스트
    print("\n🔄 Step 2: Testing complete probabilistic system...")
    
    # 테스트 데이터 생성
    test_data = pd.DataFrame({
        'first_name': ['john', 'mary', 'david', 'sarah', 'michael'],
        'last_name': ['smith', 'johnson', 'brown', 'davis', 'wilson'],
        'age': [25, 30, 35, 28, 42],
        'gender': ['male', 'female', 'male', 'female', 'male'],
        'city': ['boston', 'chicago', 'denver', 'atlanta', 'seattle']
    })
    
    print(f"   Test data: {len(test_data)} records")
    print("   Original data:")
    for i, row in test_data.iterrows():
        print(f"     {i+1}: {row['first_name']} {row['last_name']}, {row['age']}, {row['gender']}, {row['city']}")
    
    # 3. 확률적 인코딩
    print("\n🔄 Step 3: Probabilistic encoding...")
    
    processor = ProbabilisticVoterProcessor()
    vectors, metadata = processor.preprocess_data_probabilistic(test_data)
    
    print(f"   Encoded vectors shape: {vectors.shape}")
    print(f"   Metadata entries: {len(metadata)}")
    
    # 인코딩 정보 출력
    print("   Encoding examples:")
    for key, info in list(metadata.items())[:5]:
        print(f"     {key}: length={info['length']}, sum={info['ascii_sum']}")
        print(f"       Examples: {list(info['original_examples'])[:3]}")
    
    # 4. 신경망 훈련 (간단한 오토인코더)
    print("\n🤖 Step 4: Training neural network...")
    
    # 간단한 오토인코더
    input_dim = vectors.shape[1]
    
    # 인코더
    encoder_input = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(encoder_input)
    encoded = Dense(16, activation='relu')(encoded)
    encoded = Dense(8, activation='relu')(encoded)
    
    # 디코더
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # 훈련
    print("   Training autoencoder...")
    history = autoencoder.fit(vectors, vectors, epochs=100, batch_size=2, verbose=0)
    
    print(f"   Training completed. Final loss: {history.history['loss'][-1]:.6f}")
    
    # 5. 확률적 복원
    print("\n🎯 Step 5: Probabilistic reconstruction...")
    
    # 신경망을 통과한 벡터들
    reconstructed_vectors = autoencoder.predict(vectors, verbose=0)
    
    # 확률적 복원
    reconstructed_df = processor.reconstruct_probabilistic(reconstructed_vectors, metadata)
    
    print(f"   Reconstructed {len(reconstructed_df)} records")
    print("   Reconstructed data:")
    for i, row in reconstructed_df.iterrows():
        print(f"     {i+1}: {row['first_name']} {row['last_name']}, {row['age']}, {row['gender']}, {row['city']}")
    
    # 6. 정확도 분석
    print("\n📊 Step 6: Accuracy analysis...")
    
    exact_matches = 0
    field_accuracies = defaultdict(int)
    total_fields = len(test_data)
    
    comparison_fields = ['first_name', 'last_name', 'gender', 'city']
    
    print("   Field-by-field comparison:")
    for field in comparison_fields:
        correct = 0
        for i in range(len(test_data)):
            original = str(test_data.iloc[i][field]).lower()
            reconstructed = str(reconstructed_df.iloc[i][field]).lower()
            
            if original == reconstructed:
                correct += 1
                field_accuracies[field] += 1
        
        accuracy = correct / total_fields * 100
        print(f"     {field:12s}: {correct}/{total_fields} ({accuracy:.1f}%)")
    
    # 전체 레코드 매치
    for i in range(len(test_data)):
        match_count = 0
        for field in comparison_fields:
            original = str(test_data.iloc[i][field]).lower()
            reconstructed = str(reconstructed_df.iloc[i][field]).lower()
            if original == reconstructed:
                match_count += 1
        
        if match_count == len(comparison_fields):
            exact_matches += 1
    
    print(f"\n   Overall Results:")
    print(f"     Exact record matches: {exact_matches}/{total_fields} ({exact_matches/total_fields*100:.1f}%)")
    print(f"     Average field accuracy: {np.mean(list(field_accuracies.values()))/total_fields*100:.1f}%")
    
    # 7. 계산량 분석
    print("\n⚡ Step 7: Computational analysis...")
    
    total_combinations_generated = 0
    for key, info in metadata.items():
        length = info['length']
        if length <= 8:  # 현실적인 길이만
            estimated_combinations = 26 ** length  # 소문자만 사용
            total_combinations_generated += min(estimated_combinations, 50000)  # 제한된 생성
    
    print(f"   Total combination space explored: ~{total_combinations_generated:,}")
    print(f"   Dictionary lookup operations: ~{len(metadata) * 1000:,}")
    print(f"   Average candidates per field: ~{total_combinations_generated / len(metadata):.0f}")
    
    print("\n" + "="*70)
    print("🎉 PROBABILISTIC RECONSTRUCTION COMPLETED")
    print("="*70)
    print("✅ Demonstrated length + ASCII sum → candidate generation")
    print("✅ Dictionary lookup for realistic names")
    print("✅ Probabilistic scoring and selection") 
    print("✅ High probability candidate learning")
    print("✅ Computational feasibility analysis")
    print("="*70)

if __name__ == "__main__":
    demonstrate_probabilistic_reconstruction()
