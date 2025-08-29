# 클라이언트 측 smashed data 생성 (유권자 데이터 기반) - 순환 은폐 구조 적용
# 연구 아이디어: Text→Image→Vector 은폐로 보안 강화
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
print("🔄 Loading voter data for client-side smashed data generation...")
data = pd.read_csv("ncvoterb.csv", encoding='latin-1')

# 데이터 크기 제한: 실험을 위해 1,000개로 제한
# 전체 데이터: 약 224,061개 → 실험용: 1,000개 (약 0.4% 사용)
SAMPLE_SIZE = 1000
if len(data) > SAMPLE_SIZE:
    print(f"📊 Reducing data size from {len(data):,} to {SAMPLE_SIZE:,} for faster experimentation")
    data = data.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"✅ Data reduced successfully! Working with {len(data):,} records")

print(f"✅ Data loaded successfully! Total records: {len(data)}")

# 클라이언트 측 데이터로 사용할 샘플 수 (전체의 30%)
# 실험용 데이터에서 30% = 6,000개 사용
client_sample_size = int(len(data) * 0.3)
client_data = data.sample(n=client_sample_size, random_state=123)  # 다른 random_state 사용
print(f"📊 Client-side data size: {len(client_data)} (30% of {len(data)} = {len(client_data):,})")

# 모델 불러오는 경로 (Fine-tuned 모델)
model_path = "Fine-tuned_voter_final.pt"

# X_train 생성 (레이블은 smashed data 생성에 필요 없음)
X_train = []

for index, row in client_data.iterrows():
    voter_id = row["voter_id"]

    # 모든 컬럼 정보 결합 (ID와 레이블 제외)
    voter_info = []
    for col in data.columns:
        if col not in ['voter_id']:  # ID 컬럼 제외
            if pd.notna(row[col]):
                voter_info.append(f"{col}: {str(row[col])}")

    combined_info = ", ".join(voter_info)
    X_train.append(combined_info)

print(f"Generated {len(X_train)} training samples for client-side")

# BERT 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 모델이 이미 저장되어 있는지 확인하고, 저장된 모델이 있으면 불러오고 없으면 새로운 모델 생성
if os.path.exists(model_path):
    # 저장된 모델이 있을 경우 불러오기
    model = CircularObfuscationModel(num_classes=2, use_bert=True)
    try:
        checkpoint = torch.load(model_path)
        # Fine-tuned 모델은 직접 state_dict로 저장되어 있음 (model_state_dict 키 없음)
        model.load_state_dict(checkpoint)
        print("Fine-tuned CircularObfuscationModel loaded for client-side.")
    except Exception as e:
        print(f"Warning: Could not load fine-tuned model ({str(e)[:100]}...), using new model...")
        model = CircularObfuscationModel(num_classes=2, use_bert=True)
else:
    # 저장된 모델이 없을 경우 새로운 모델 생성
    model = CircularObfuscationModel(num_classes=2, use_bert=True)
    print("New CircularObfuscationModel generated for client-side.")

# 입력 데이터를 BERT의 입력 형식으로 변환
max_len = 128  # 입력 시퀀스의 최대 길이
print(f"🔄 Tokenizing {len(X_train)} client-side samples...")

input_ids = []
attention_masks = []

for i, info in enumerate(X_train):
    if i % 1000 == 0:  # 1000개마다 진행 상황 출력
        print(f"  Tokenizing sample {i}/{len(X_train)}...")
    encoded_dict = tokenizer.encode_plus(
                        info,                         # 유권자 정보
                        add_special_tokens = True,    # [CLS], [SEP] 토큰 추가
                        max_length = max_len,         # 최대 길이 지정
                        padding = 'max_length',       # 패딩을 추가하여 최대 길이로 맞춤
                        return_attention_mask = True, # 어텐션 마스크 생성
                        return_tensors = 'pt',        # PyTorch 텐서로 반환
                   )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

print("✅ Client-side tokenization completed!")
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# 더미 라벨 생성 (smashed data 생성에는 실제 라벨이 필요 없음)
dummy_labels = torch.zeros(len(X_train), dtype=torch.long)

# 데이터셋 생성
dataset = TensorDataset(input_ids, attention_masks, dummy_labels)

# 데이터로더 생성
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# GPU 사용 가능 여부 확인 (Apple M4용 MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device for client-side: {device}")

# 모델을 GPU로 이동
model.to(device)

# 모델 평가 모드로 설정
model.eval()
hidden_states_list = []  # 평가할 때 hidden state를 저장할 리스트

print("🔄 Generating smashed data for client-side...")
import time
start_time = time.time()

for batch_idx, batch in enumerate(dataloader):
    if batch_idx % 10 == 0:  # 10배치마다 진행 상황 출력
        print(f"  Processing batch {batch_idx}/{len(dataloader)}...")
    
    batch = tuple(t.to(device) for t in batch)
    input_ids = batch[0]
    attention_mask = batch[1]
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        # CircularObfuscationModel은 (classification_logits, loss, smashed_vector)를 반환
        if isinstance(outputs, tuple):
            smashed_vector = outputs[2]  # smashed_vector 추출
        else:
            # return_all=True인 경우
            smashed_vector = outputs['smashed_vector']

    # smashed vector를 저장
    hidden_states_list.append(smashed_vector.cpu())

generation_time = time.time() - start_time
print(f"✅ Client-side smashed data generation completed in {generation_time:.2f}s")

# hidden states를 하나의 텐서로 결합
hidden_states_concat = torch.cat(hidden_states_list, dim=0)
# smashed_vector는 이미 [batch_size, 768] 형태이므로 [:, 0, :] 제거
hidden_states_concat = hidden_states_concat.cpu().detach().numpy()

# DataFrame으로 변환 및 CSV 저장 (voter_id 포함)
client_voter_ids = [row["voter_id"] for _, row in client_data.iterrows()]
hidden_states_df = pd.DataFrame(hidden_states_concat)
hidden_states_df.insert(0, 'voter_id', client_voter_ids[:len(hidden_states_df)])  # voter_id 추가
hidden_states_df.to_csv("Client_smashed_data.csv", index=False)

print(f"💾 Client-side smashed data saved to 'Client_smashed_data.csv'")
print(f"📊 Shape: {hidden_states_concat.shape}")
print("🎉 Client-side smashed data generation completed successfully!")
