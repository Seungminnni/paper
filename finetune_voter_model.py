# Fine-tune용 (ncvoterb.csv 기반 유권자 데이터)
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
        hidden_states = outputs.hidden_states[-8]  # 8번째 레이어의 hidden states를 반환합니다.
        loss = outputs.loss
        return logits, loss, hidden_states

# 데이터 로드 및 전처리
data_A = pd.read_csv("ncvoterb.csv")  # 유권자 데이터 파일
# 모델 불러오는 경로 (Pre-trained 모델)
model_path = "Pre_train_voter_epoch10_BERT_Medium.pt"
# 모델 저장경로
model_path2 = "Fine-tuned_voter.pt"

# X_train, Y_train 생성
X_train = []
Y_train = []

# 유권자 데이터 컬럼들
text_columns = ['first_name', 'middle_name', 'last_name', 'age', 'gender', 'race', 'ethnic',
                'street_address', 'city', 'state', 'zip_code', 'birth_place']

for index, row in data_A.iterrows():
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

    # 레이블 생성: gender 기반 (m=1, f=0)
    if pd.notna(row.get('gender')):
        gender = str(row['gender']).lower()
        if gender.startswith('m'):
            Y_train.append(1)
        elif gender.startswith('f'):
            Y_train.append(0)
        else:
            Y_train.append(0)  # 기타는 0으로
    else:
        Y_train.append(0)

print(f"Generated {len(X_train)} training samples")
print(f"Label distribution: {np.bincount(Y_train)}")

# BERT 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# 모델이 이미 저장되어 있는지 확인하고, 저장된 모델이 있으면 불러오고 없으면 새로운 모델 생성
if os.path.exists(model_path):
    # 저장된 모델이 있을 경우 불러오기
    model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(model_path))
    print("Pre-trained model loaded.")
else:
    # 저장된 모델이 없을 경우 새로운 모델 생성
    model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    print("New model generated.")

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
labels = torch.tensor(Y_train)

# 데이터셋 및 데이터로더 생성
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = 0.8
train_dataset, val_dataset = train_test_split(dataset, test_size=1-train_size, random_state=42)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# GPU 사용 가능 여부 확인 (Apple M4용 MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 모델을 GPU로 이동
model.to(device)

# 옵티마이저 및 학습률 설정
# 기본 학습률 : 2e-6 (Pre-train보다 낮음)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)

# 에폭 설정
epochs = 20

# 학습 루프
hidden_states_list = []  # 모든 에폭에 대한 hidden state를 저장할 리스트
# 학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[1]  # loss가 outputs의 두 번째 값입니다.
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}')

    # 모델 저장 및 평가
    model_save_path = f"Fine_tuned_voter_epoch{epoch + 1}_BERT_Medium.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved for epoch {epoch + 1} at {model_save_path}")

    model.eval()
    val_accuracy = 0
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs[0]  # logits가 outputs의 첫 번째 값입니다.
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        val_accuracy += (logits.argmax(axis=1) == label_ids).mean().item()

    print(f'Validation Accuracy for epoch {epoch + 1}: {val_accuracy / len(val_dataloader)}')

print("Fine-tuning completed!")
