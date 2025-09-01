# 서버측 smashed data 생성 (유권자 데이터 기반) - 순환 은폐 구조 적용
# 연구 아이디어: Vector→Image→Text 복원으로 보안 강화
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from circular_obfuscation import CircularObfuscationModel

# 데이터 로드 및 전처리
print("🔄 Loading voter data for server-side smashed data generation...")
data = pd.read_csv("ncvoterb.csv", encoding='latin-1')

# 데이터 크기 제한: 실험을 위해 1,000개로 제한
# 전체 데이터: 약 224,061개 → 실험용: 1,000개 (약 0.4% 사용)
SAMPLE_SIZE = 1000
if len(data) > SAMPLE_SIZE:
    print(f"📊 Reducing data size from {len(data):,} to {SAMPLE_SIZE:,} for faster experimentation")
    data = data.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"✅ Data reduced successfully! Working with {len(data):,} records")

print(f"✅ Data loaded successfully! Total records: {len(data)}")

# 서버측 데이터로 사용할 샘플 수 (전체의 70%)
# 실험용 데이터에서 70% = 14,000개 사용
server_sample_size = int(len(data) * 0.7)
server_data = data.sample(n=server_sample_size, random_state=42)
print(f"📊 Server-side data size: {len(server_data)} (70% of {len(data)} = {len(server_data):,})")

# 모델 불러오는 경로 (Pre-trained 모델)
model_path = "Pre-trained_voter_final.pt"

# X_train 생성 (레이블은 smashed data 생성에 필요 없음)
X_train = []

for index, row in server_data.iterrows():
    voter_id = row["voter_id"]

    # 모든 컬럼 정보 결합 (ID와 레이블 제외)
    voter_info = []
    for col in data.columns:
        if col not in ['voter_id']:  # ID 컬럼 제외
            if pd.notna(row[col]):
                voter_info.append(f"{col}: {str(row[col])}")

    combined_info = ", ".join(voter_info)
    X_train.append(combined_info)

print(f"Generated {len(X_train)} training samples for server-side")

# BERT 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 모델이 이미 저장되어 있는지 확인하고, 저장된 모델이 있으면 불러오고 없으면 새로운 모델 생성
if os.path.exists(model_path):
    # 저장된 모델이 있을 경우 불러오기
    model = CircularObfuscationModel(num_classes=2, use_bert=False)
    try:
        checkpoint = torch.load(model_path)
        # Pre-trained 모델은 model_state_dict 키로 저장되어 있는지 확인
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 직접 state_dict로 저장된 경우
            model.load_state_dict(checkpoint)
        print("Pre-trained CircularObfuscationModel loaded for server-side.")
    except Exception as e:
        print(f"Warning: Could not load pre-trained model ({str(e)[:100]}...), using new model...")
        model = CircularObfuscationModel(num_classes=2, use_bert=False)
else:
    # 저장된 모델이 없을 경우 새로운 모델 생성
    model = CircularObfuscationModel(num_classes=2, use_bert=False)
    print("New CircularObfuscationModel generated for server-side.")

# ServerCircularModel은 smashed vector를 입력으로 받으므로 토크나이징 불필요
# 실제로는 client에서 생성된 smashed vector를 사용해야 함
# 현재는 테스트를 위해 더미 데이터를 생성하고 있지만,
# 실제 구현에서는 Client_smashed_data.csv에서 데이터를 로드해야 함

print("🔄 Loading actual smashed vectors from client...")
try:
    # Client가 생성한 실제 smashed data 로드
    client_smashed_df = pd.read_csv("Client_smashed_data.csv")
    smashed_vectors = torch.tensor(client_smashed_df.values, dtype=torch.float32)
    print(f"✅ Loaded {len(smashed_vectors)} smashed vectors from Client_smashed_data.csv")
    print(f"📊 Smashed vector shape: {smashed_vectors.shape}")
except FileNotFoundError:
    print("⚠️  Client_smashed_data.csv not found, using dummy data for testing...")
    # 더미 smashed vector 생성 (테스트용)
    dummy_smashed_vectors = []
    for i in range(len(X_train)):
        dummy_vector = torch.randn(768)  # 768차원 벡터
        dummy_smashed_vectors.append(dummy_vector)
    smashed_vectors = torch.stack(dummy_smashed_vectors)

# 실제 데이터 수에 맞는 더미 라벨 생성
dummy_labels = torch.zeros(len(smashed_vectors), dtype=torch.long)

# 데이터셋 생성
dataset = TensorDataset(smashed_vectors, dummy_labels)

# 데이터로더 생성
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# GPU 사용 가능 여부 확인 (Apple M4용 MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device for server-side: {device}")

# 모델을 GPU로 이동
model.to(device)

# 모델 평가 모드로 설정
model.eval()

# 모델 평가 모드로 설정
model.eval()
hidden_states_list = []  # 평가할 때 hidden state를 저장할 리스트

print("🔄 Generating smashed data for server-side...")
import time
start_time = time.time()

for batch_idx, batch in enumerate(dataloader):
    if batch_idx % 10 == 0:  # 10배치마다 진행 상황 출력
        print(f"  Processing batch {batch_idx}/{len(dataloader)}...")
    
    batch = tuple(t.to(device) for t in batch)
    smashed_vector = batch[0]  # smashed vector
    labels = batch[1]          # labels
    
    with torch.no_grad():
        outputs = model(smashed_vector, None, labels)  # attention_mask=None for server mode

    # ServerCircularModel은 (classification_logits, loss, text_logits) 또는 (classification_logits, text_logits)를 반환
    if len(outputs) == 3:
        classification_logits, loss, smashed_vector = outputs
    else:
        classification_logits, smashed_vector = outputs

    # 실제 smashed vector를 저장 (768차원)
    hidden_states_list.append(smashed_vector.cpu())

generation_time = time.time() - start_time
print(f"✅ Smashed data generation completed in {generation_time:.2f}s")

# hidden states를 하나의 텐서로 결합
hidden_states_concat = torch.cat(hidden_states_list, dim=0)

# classification_logits는 (batch_size, num_classes) 형태이므로 [:, 0, :] 대신 그대로 사용
# BERT hidden states가 아니므로 차원 조정 불필요
hidden_states_concat = hidden_states_concat.cpu().detach().numpy()

# DataFrame으로 변환 및 CSV 저장 (voter_id 포함)
# Client 데이터에서 voter_id 추출
client_df = pd.read_csv("Client_smashed_data.csv")
server_voter_ids = client_df['voter_id'].tolist()
hidden_states_df = pd.DataFrame(hidden_states_concat)
hidden_states_df.insert(0, 'voter_id', server_voter_ids[:len(hidden_states_df)])  # voter_id 추가
hidden_states_df.to_csv("Dictionary_smashed_data.csv", index=False)

print(f"💾 Server-side smashed data saved to 'Dictionary_smashed_data.csv'")
print(f"📊 Shape: {hidden_states_concat.shape}")
print("🎉 Server-side smashed data generation completed successfully!")
