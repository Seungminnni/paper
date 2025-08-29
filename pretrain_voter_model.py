# Pre-train용 (ncvoterb.csv 기반 유권자 데이터)
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
print("🔄 Loading voter data for pre-training...")
data_A = pd.read_csv("ncvoterb.csv", encoding='latin-1')  # 유권자 데이터 파일 (latin-1 인코딩으로 읽기)

# 데이터 크기 제한: 빠른 실험을 위해 5,000개로 제한
# 전체 데이터: 약 224,061개 → 빠른 실험용: 5,000개 (약 2.2% 사용)
# 이렇게 하면 10 에포크 학습이 약 5-10분 내에 완료됩니다
SAMPLE_SIZE = 5000
if len(data_A) > SAMPLE_SIZE:
    print(f"📊 Reducing data size from {len(data_A):,} to {SAMPLE_SIZE:,} for faster experimentation")
    print(f"   This will make training {len(data_A)//SAMPLE_SIZE}x faster!")
    data_A = data_A.sample(n=SAMPLE_SIZE, random_state=42)  # 랜덤 샘플링으로 데이터 선택
    print(f"✅ Data reduced successfully! Working with {len(data_A):,} records")

print(f"✅ Data loaded successfully! Total records: {len(data_A)}")
print(f"   Data shape: {data_A.shape}")
print(f"   Columns: {list(data_A.columns)}")
# 모델 저장 경로 (최종 모델만 저장)
# 에포크마다 저장하지 않고, 학습 완료 후 최종 모델만 저장합니다
model_path = "Pre-trained_voter_final.pt"  # 최종 모델 파일명
print(f"📁 Final model will be saved as: {model_path}")

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

    # 레이블 생성: gender 기반 (m=1, f=0, 기타=-1로 처리)
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
print(f"Sample text: {X_train[0][:200]}...")
print(f"Label distribution: {np.bincount(Y_train)}")

# BERT 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 입력 데이터를 BERT의 입력 형식으로 변환
max_len = 128  # 입력 시퀀스의 최대 길이
print(f"🔄 Tokenizing {len(X_train)} samples...")

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

# 데이터 분할: 80% 학습, 20% 검증 (빠른 실험용 설정)
# - 전체 데이터: 5,000개
# - 학습 데이터: 4,000개 (5,000 * 0.8)
# - 검증 데이터: 1,000개 (5,000 * 0.2)
# - 배치 크기: 16개씩 처리 (한 배치에 16개 샘플씩 GPU로 처리)
train_size = 0.8
train_dataset, val_dataset = train_test_split(dataset, test_size=1-train_size, random_state=42)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 학습 데이터는 섞어서 과적합 방지
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)   # 검증 데이터도 섞음

print(f"📊 Data split completed:")
print(f"   Training set: {len(train_dataset)} samples ({len(train_dataset)/len(dataset)*100:.1f}%)")
print(f"   Validation set: {len(val_dataset)} samples ({len(val_dataset)/len(dataset)*100:.1f}%)")
print(f"   Batch size: 16, Training batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")
print(f"   Total training steps per epoch: {len(train_dataloader)}")
print(f"   Total validation steps per epoch: {len(val_dataloader)}")

# GPU 사용 가능 여부 확인
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 모델을 GPU로 이동
model.to(device)

# 옵티마이저 및 학습률 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # BERT의 표준 학습률
print(f"⚙️  Optimizer: AdamW with learning rate {2e-5}")
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 에폭 설정 (빠른 실험을 위해 5 에포크로 설정)
# 5,000개 데이터, 5 에포크 학습 시 약 3-5분 소요 (Apple M4 기준)
epochs = 5
print(f"🎯 Training configuration:")
print(f"   Total epochs: {epochs}")
print(f"   Samples per epoch: {len(train_dataset)}")
print(f"   Estimated training time: ~{epochs * 0.8:.1f} minutes (rough estimate)")

# 학습 루프
import time  # 시간 측정용
total_training_start = time.time()  # 전체 학습 시간 측정 시작

# 최고 검증 정확도를 추적하기 위한 변수들
best_val_accuracy = 0.0
best_epoch = 0

print(f"\n🚀 Starting Pre-training with {epochs} epochs...")
print("=" * 60)

for epoch in range(epochs):
    epoch_start_time = time.time()
    print(f"\n🚀 Epoch {epoch + 1}/{epochs} - Training Phase")
    print("-" * 40)
    
    # ===== 학습 단계 =====
    model.train()  # 모델을 학습 모드로 설정
    total_loss = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx % 10 == 0:  # 10배치마다 진행 상황 출력
            progress = (batch_idx / len(train_dataloader)) * 100
            print(f"  📈 Training progress: {progress:.1f}% ({batch_idx}/{len(train_dataloader)} batches)")
        
        # 배치를 GPU로 이동
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }
        
        # 그래디언트 초기화 및 순전파
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[1]  # loss가 outputs의 두 번째 값
        
        # 역전파 및 옵티마이저 스텝
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        batch_count += 1

    # 에포크별 평균 손실 계산
    avg_train_loss = total_loss / batch_count
    epoch_time = time.time() - epoch_start_time
    
    print(f"✅ Epoch {epoch + 1} Training completed:")
    print(f"   ⏱️  Training time: {epoch_time:.2f}s")
    print(f'   📉 Average Training Loss: {avg_train_loss:.4f}')
    print(f"   📊 Processed {batch_count} batches, {len(train_dataset)} samples")

    # ===== 검증 단계 =====
    print(f"\n🔍 Epoch {epoch + 1} - Validation Phase")
    print("-" * 40)
    
    model.eval()  # 모델을 평가 모드로 설정
    val_accuracy = 0
    val_loss = 0
    val_batch_count = 0
    
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }
        
        with torch.no_grad():  # 그래디언트 계산하지 않음
            outputs = model(**inputs)
            logits = outputs[0]  # logits가 outputs의 첫 번째 값
            loss = outputs[1]    # loss가 outputs의 두 번째 값
            
        # 정확도 계산
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        batch_accuracy = (logits.argmax(axis=1) == label_ids).mean().item()
        val_accuracy += batch_accuracy
        val_loss += loss.item()
        val_batch_count += 1

    # 검증 결과 계산
    avg_val_accuracy = val_accuracy / val_batch_count
    avg_val_loss = val_loss / val_batch_count
    
    print(f"✅ Epoch {epoch + 1} Validation completed:")
    print(f'   📊 Validation Accuracy: {avg_val_accuracy:.4f}')
    print(f'   📉 Validation Loss: {avg_val_loss:.4f}')
    print(f"   📈 Processed {val_batch_count} validation batches")
    
    # 최고 성능 모델 추적
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        best_epoch = epoch + 1
        print(f"   🏆 New best model! (Accuracy: {best_val_accuracy:.4f})")
    
    print(f"⏱️  Total epoch time: {epoch_time:.2f}s")
    print("=" * 60)

# ===== 최종 모델 저장 =====
print(f"\n💾 Saving final model...")
print(f"   📁 Model path: {model_path}")
print(f"   🏆 Best validation accuracy: {best_val_accuracy:.4f} (Epoch {best_epoch})")

# 최종 모델 저장 (에포크마다 저장하지 않고 최종 모델만 저장)
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_accuracy': best_val_accuracy,
    'best_epoch': best_epoch,
    'total_samples': len(dataset),
    'train_samples': len(train_dataset),
    'val_samples': len(val_dataset)
}, model_path)

total_training_time = time.time() - total_training_start
print(f"✅ Final model saved successfully!")
print(f"   📊 Model file: {model_path}")
print(f"   ⏱️  Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} minutes)")
print(f"   📈 Best performance: {best_val_accuracy:.4f} accuracy at epoch {best_epoch}")

print("\n🎉 Pre-training completed successfully!")
print("=" * 60)
print("📋 Summary:")
print(f"   • Total samples: {len(dataset)}")
print(f"   • Training samples: {len(train_dataset)}")
print(f"   • Validation samples: {len(val_dataset)}")
print(f"   • Total epochs: {epochs}")
print(f"   • Best validation accuracy: {best_val_accuracy:.4f}")
print(f"   • Training time: {total_training_time:.2f}s")
print(f"   • Final model saved: {model_path}")
print("=" * 60)