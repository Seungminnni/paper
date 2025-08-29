# 클라이언트 측 smashed data 생성 (유권자 데이터 기반)
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split

class CustomBertForSequenceClassification(BertForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        output_hidden_states=True
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            labels=labels,
            output_hidden_states=output_hidden_states
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[7]  # 7번째 레이어의 hidden states를 반환합니다.
        loss = outputs.loss
        return logits, loss, hidden_states

# 데이터 로드 및 전처리
print("Loading voter data for client-side smashed data generation...")
data = pd.read_csv("ncvoterb.csv")

# 클라이언트 측 데이터로 사용할 샘플 수 (전체의 30%)
client_sample_size = int(len(data) * 0.3)
client_data = data.sample(n=client_sample_size, random_state=123)  # 다른 random_state 사용

print(f"Client-side data size: {len(client_data)}")

# 모델 불러오는 경로 (Fine-tuned 모델)
model_path = "Fine_tuned_voter_epoch20_BERT_Medium.pt"

# X_train 생성 (레이블은 smashed data 생성에 필요 없음)
X_train = []

# 유권자 데이터 컬럼들
text_columns = ['first_name', 'middle_name', 'last_name', 'age', 'gender', 'race', 'ethnic',
                'street_address', 'city', 'state', 'zip_code', 'birth_place']

for index, row in client_data.iterrows():
    voter_id = row["voter_id"]

    # 텍스트 정보 결합 (이름, 주소, 인구통계 정보)
    voter_info = []
    for col in text_columns:
        if pd.notna(row[col]):
            voter_info.append(f"{col}: {str(row[col])}")

    # 추가 정보 결합 (등록일, 전화번호 등)
    if pd.notna(row.get('register_date')):
        voter_info.append(f"register_date: {str(row['register_date'])}")
    if pd.notna(row.get('full_phone_num')):
        voter_info.append(f"phone: {str(row['full_phone_num'])}")

    combined_info = ", ".join(voter_info)
    X_train.append(combined_info)

print(f"Generated {len(X_train)} training samples for client-side")

# BERT 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 모델이 이미 저장되어 있는지 확인하고, 저장된 모델이 있으면 불러오고 없으면 새로운 모델 생성
if os.path.exists(model_path):
    # 저장된 모델이 있을 경우 불러오기
    model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(model_path), strict=False)
    print("Fine-tuned model loaded for client-side.")
else:
    # 저장된 모델이 없을 경우 새로운 모델 생성
    model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    print("New model generated for client-side.")

# 입력 데이터를 BERT의 입력 형식으로 변환
max_len = 128  # 입력 시퀀스의 최대 길이

input_ids = []
attention_masks = []

for info in X_train:
    encoded_dict = tokenizer.encode_plus(
                        info,                         # 유권자 정보
                        add_special_tokens = True,    # [CLS], [SEP] 토큰 추가
                        max_length = max_len,         # 최대 길이 지정
                        pad_to_max_length = True,     # 패딩을 추가하여 최대 길이로 맞춤
                        return_attention_mask = True, # 어텐션 마스크 생성
                        return_tensors = 'pt',        # PyTorch 텐서로 반환
                   )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

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

print("Generating smashed data for client-side...")
for batch in dataloader:
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'labels': batch[2]}
    with torch.no_grad():
        outputs = model(**inputs)

    # hidden state를 저장합니다.
    hidden_states = outputs[2]
    hidden_states_list.append(hidden_states)

# hidden states를 하나의 텐서로 결합
hidden_states_concat = torch.cat(hidden_states_list, dim=0)
hidden_states_concat = hidden_states_concat[:, 0, :].cpu().detach().numpy()

# DataFrame으로 변환 및 CSV 저장
hidden_states_df = pd.DataFrame(hidden_states_concat)
hidden_states_df.to_csv("Client_smashed_data.csv", index=False)

print(f"Client-side smashed data saved to 'Client_smashed_data.csv'")
print(f"Shape: {hidden_states_concat.shape}")
print("Client-side smashed data generation completed!")
