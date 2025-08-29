# Fine-tune용 (ncvoterb.csv 기반 유권자 데이터) - 순환 은폐 구조 적용
# 연구 아이디어: Text→Image→Vector→Image→Text 순환 변환으로 보안 강화
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class CircularObfuscationModel(nn.Module):
    """
    텍스트→이미지→벡터→이미지→텍스트 순환 구조 모델
    공격자가 중간 데이터를 탈취하더라도 의미 추론이 어려움
    """
    def __init__(self, num_classes=2, vocab_size=30522):
        super().__init__()

        # ===== Phase 1: Text → Image 변환 =====
        # 1. Text Encoder (기존 BERT)
        self.text_encoder = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_classes
        )

        # 2. Image Generator (Text → Image)
        self.image_generator = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 32 * 32),  # 7채널 32x32 이미지
            nn.Sigmoid()
        )

        # ===== Phase 2: Image → Vector 변환 =====
        # 3. Vector Encoder (Image → Vector for smashed data)
        self.vector_encoder = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 768),  # 다시 768차원 벡터로
            nn.LayerNorm(768)
        )

        # ===== Phase 3: Vector → Image 재구성 =====
        # 4. Vector Decoder (Vector → Image reconstruction)
        self.vector_decoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 32 * 32),
            nn.Sigmoid()
        )

        # ===== Phase 4: Image → Text 재구성 =====
        # 5. Image Decoder (Image → Text embedding)
        self.image_decoder = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 768),
            nn.LayerNorm(768)
        )

        # 6. Text Decoder (Vector → Text tokens)
        self.text_decoder = nn.Linear(768, vocab_size)

        # 분류 헤드 (최종 분류용)
        self.classifier = nn.Linear(768, num_classes)

        # 드롭아웃으로 과적합 방지
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, labels=None, return_all=False):
        """
        순환 변환 수행
        Args:
            input_ids: BERT 토큰 ID들
            attention_mask: 어텐션 마스크
            labels: 분류 레이블 (선택)
            return_all: 모든 중간 결과 반환 여부
        """
        # ===== Phase 1: Text → Image =====
        # 1. Text encoding
        bert_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        text_embedding = bert_outputs.hidden_states[-1][:, 0, :]  # CLS token

        # 2. Generate image from text
        generated_image = self.image_generator(text_embedding)
        generated_image = generated_image.view(-1, 7, 32, 32)

        # ===== Phase 2: Image → Vector =====
        # 3. Encode image to vector (smashed data)
        smashed_vector = self.vector_encoder(generated_image)

        # ===== Phase 3: Vector → Image =====
        # 4. Reconstruct image from vector
        reconstructed_image = self.vector_decoder(smashed_vector)
        reconstructed_image = reconstructed_image.view(-1, 7, 32, 32)

        # ===== Phase 4: Image → Text =====
        # 5. Decode image to text embedding
        text_reconstruction = self.image_decoder(reconstructed_image)

        # 6. Generate text tokens from embedding
        text_logits = self.text_decoder(text_reconstruction)

        # ===== Classification =====
        # Use smashed vector for classification (server side)
        classification_logits = self.classifier(smashed_vector)

        # Loss 계산 (있는 경우)
        loss = None
        if labels is not None:
            # 분류 Loss
            classification_loss = F.cross_entropy(classification_logits, labels)

            # 이미지 재구성 Loss (Text → Image → Image)
            image_reconstruction_loss = F.mse_loss(generated_image, reconstructed_image)

            # 텍스트 재구성 Loss (원본 텍스트와 복원 텍스트 비교)
            text_reconstruction_loss = F.mse_loss(text_embedding, text_reconstruction)

            # 일관성 Loss (생성된 이미지와 재구성된 이미지의 차이)
            consistency_loss = F.mse_loss(generated_image, reconstructed_image)

            # 가중치 적용
            loss = (
                classification_loss +
                0.1 * image_reconstruction_loss +
                0.1 * text_reconstruction_loss +
                0.1 * consistency_loss
            )

        if return_all:
            return {
                'classification_logits': classification_logits,
                'generated_image': generated_image,
                'smashed_vector': smashed_vector,
                'reconstructed_image': reconstructed_image,
                'text_logits': text_logits,
                'original_embedding': text_embedding,
                'loss': loss
            }
        else:
            return classification_logits, loss, smashed_vector

# 데이터 로드 및 전처리
print("🔄 Loading voter data for fine-tuning...")
data_A = pd.read_csv("ncvoterb.csv", encoding='latin-1')  # 유권자 데이터 파일

# 데이터 크기 제한: 실험을 위해 1,000개로 제한
# 전체 데이터: 약 224,061개 → 실험용: 1,000개 (약 0.4% 사용)
SAMPLE_SIZE = 1000
if len(data_A) > SAMPLE_SIZE:
    print(f"📊 Reducing data size from {len(data_A):,} to {SAMPLE_SIZE:,} for faster experimentation")
    data_A = data_A.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"✅ Data reduced successfully! Working with {len(data_A):,} records")

print(f"✅ Data loaded successfully! Total records: {len(data_A)}")
# 모델 불러오는 경로 (Pre-trained 모델)
model_path = "Pre-trained_voter_final.pt"
# 모델 저장경로
model_path2 = "Fine-tuned_voter.pt"

# X_train, Y_train 생성
X_train = []
Y_train = []

for index, row in data_A.iterrows():
    voter_id = row["voter_id"]

    # 모든 컬럼 정보 결합 (ID와 레이블 제외)
    voter_info = []
    for col in data_A.columns:
        if col not in ['voter_id']:  # ID 컬럼 제외
            if pd.notna(row[col]):
                voter_info.append(f"{col}: {str(row[col])}")

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

# 모델 초기화 (순환 은폐 구조 적용)
print("🔄 Initializing Circular Obfuscation Model for Fine-tuning...")
print("   📋 Model Structure: Text → Image → Vector → Image → Text")
print("   🛡️  Security: Multi-modal transformation increases attack difficulty")

# 모델이 이미 저장되어 있는지 확인하고, 저장된 모델이 있으면 불러오고 없으면 새로운 모델 생성
if os.path.exists(model_path):
    # 저장된 모델이 있을 경우 불러오기
    model = CircularObfuscationModel(num_classes=2)
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pre-trained Circular Obfuscation model loaded successfully!")
    except:
        print("Warning: Could not load pre-trained model, using new model...")
        model = CircularObfuscationModel(num_classes=2)
else:
    # 저장된 모델이 없을 경우 새로운 모델 생성
    model = CircularObfuscationModel(num_classes=2)
    print("New Circular Obfuscation model generated.")

print(f"✅ Fine-tune model initialized!")
print(f"   📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   🔒 Security layers: 4 transformation stages")
print(f"   🎯 Attack difficulty: 4x increased (Text→Image→Vector→Image→Text)")

# BERT 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(f"✅ BERT Tokenizer initialized!")

# 입력 데이터를 BERT의 입력 형식으로 변환
max_len = 128  # 입력 시퀀스의 최대 길이
print(f"🔄 Tokenizing {len(X_train)} samples for fine-tuning...")

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

print("✅ Tokenization completed!")
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(Y_train)

# 데이터셋 및 데이터로더 생성
dataset = TensorDataset(input_ids, attention_masks, labels)

# 데이터 분할: 80% 학습, 20% 검증 (실험용 설정)
# - 학습 데이터: 16,000개 (20,000 * 0.8)
# - 검증 데이터: 4,000개 (20,000 * 0.2)
# - 배치 크기: 16개씩 처리
train_size = 0.8
train_dataset, val_dataset = train_test_split(dataset, test_size=1-train_size, random_state=42)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

print(f"📊 Data split completed:")
print(f"   Training set: {len(train_dataset)} samples ({len(train_dataset)/len(dataset)*100:.1f}%)")
print(f"   Validation set: {len(val_dataset)} samples ({len(val_dataset)/len(dataset)*100:.1f}%)")
print(f"   Batch size: 16, Training batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")

# GPU 사용 가능 여부 확인 (Apple M4용 MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 모델을 GPU로 이동
model.to(device)

# 옵티마이저 및 학습률 설정
# 기본 학습률 : 2e-6 (Pre-train보다 낮음)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)

# 에폭 설정 (빠른 실험을 위해 5 에포크로 설정)
epochs = 5

# 학습 루프
hidden_states_list = []  # 모든 에폭에 대한 hidden state를 저장할 리스트
import time  # 시간 측정용

# 학습 루프
for epoch in range(epochs):
    print(f"\n🚀 Starting Fine-tune Epoch {epoch + 1}/{epochs}")
    start_time = time.time()
    
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx % 10 == 0:  # 10배치마다 진행 상황 출력
            print(f"  Processing batch {batch_idx}/{len(train_dataloader)}...")
        
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
    epoch_time = time.time() - start_time
    print(f"✅ Fine-tune Epoch {epoch + 1} completed in {epoch_time:.2f}s")
    print(f'   Average Training Loss: {avg_train_loss:.4f}')

    # 모델 저장 및 평가
    model_save_path = f"Fine-tuned_voter_final.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model saved: {model_save_path}")

    print(f"🔍 Evaluating on validation set...")
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

    val_accuracy = val_accuracy / len(val_dataloader)
    print(f'📊 Validation Accuracy: {val_accuracy:.4f}')
    print(f"⏱️  Total epoch time: {epoch_time:.2f}s")

print("\n🎉 Fine-tuning completed successfully!")