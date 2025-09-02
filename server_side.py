#!/usr/bin/env python3
"""
Server-side Smashed Data Generation
서버 사이드 Smashed Data 생성 (500개 샘플)
"""

import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch

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
        hidden_states = outputs.hidden_states[-8]  # 8번째 레이어의 hidden states
        loss = outputs.loss
        return logits, loss, hidden_states

def main():
    print("🖥️  Starting Server-side Smashed Data Generation")
    print("=" * 60)

    # 데이터 로드 및 전처리
    print("📊 Loading data for server-side processing...")
    data_A = pd.read_csv("random_500.csv")  # 500개 샘플 데이터셋
    data_B = pd.read_csv("infected.csv")  # 감염 상태 데이터셋
    model_path = "Pre_train_final_BERT_Based.pt"

    print(f"📈 Processing {len(data_A)} patient records for server-side")

    # X_train, Y_train 생성
    X_train = []
    Y_train = []

    for index, row in data_A.iterrows():
        patient_id = row["ID"]
        patient_info = [str(row[column]) for column in data_A.columns if column != "ID" and column != "DESCRIPTION"]
        symptoms = ", ".join(data_A[data_A["ID"] == patient_id]["DESCRIPTION"].tolist())
        combined_info = ", ".join(patient_info) + ", " + symptoms
        X_train.append(combined_info)
        if patient_id in data_B.values:
            Y_train.append(1)  # Infected
        else:
            Y_train.append(0)  # Not infected

    print(f"✅ Processed {len(X_train)} patient records")
    print(f"📋 Sample combined info: {X_train[0][:100]}...")

    # BERT 토크나이저 로드
    print("\n🤖 Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 모델 로드
    print("🔧 Loading fine-tuned model...")
    if os.path.exists(model_path):
        model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.load_state_dict(torch.load(model_path))
        print("✅ Fine-tuned model loaded successfully")
    else:
        print("❌ Fine-tuned model not found!")
        return

    # 입력 데이터 변환
    max_len = 128
    print(f"🔄 Tokenizing {len(X_train)} texts with max length {max_len}...")

    input_ids = []
    attention_masks = []

    for info in X_train:
        encoded_dict = tokenizer.encode_plus(
            info,
            add_special_tokens=True,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(Y_train)

    # 데이터셋 및 데이터로더 생성
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    print(f"📦 Dataset created with {len(dataset)} samples")

    # GPU 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"🖥️  Using device: {device}")

    # --- 아이디어 적용: 클라이언트로부터 받은 이미지 배열을 벡터로 복원 ---
    import json

    def image_to_vector(img, original_dim, vmin, vmax):
        """
        절차적으로 생성된 이미지를 원본 벡터로 복원합니다.
        """
        img = np.asarray(img, dtype=np.float32)
        v_padded = img.flatten()
        v_scaled = v_padded[:original_dim]
        v_original = v_scaled * (vmax - vmin + 1e-8) + vmin
        return v_original

    # 1. 클라이언트로부터 Smashed Image 배열 수신 (가정)
    # 실제 구현에서는 네트워크를 통해 수신한 데이터를 역직렬화해야 합니다.
    # 예: received_data = receive_from_client()
    #     smashed_images = pickle.loads(received_data)
    print("☁️  Waiting for data from client...")
    print("-> (Demonstration mode: Simulating received data)")
    # 시연을 위해, 클라이언트가 생성했을 법한 임시 데이터 생성
    # 실제로는 이 부분이 필요 없습니다.
    with open("vector_image_config.json", "r") as f:
        temp_config = json.load(f)
    smashed_images = np.random.rand(10, temp_config["image_side"], temp_config["image_side"], 1).astype(np.float32)
    print(f"-> Received a batch of {smashed_images.shape[0]} image arrays.")

    # 2. 공유 설정 파일 로드
    with open("vector_image_config.json", "r") as f:
        config = json.load(f)

    # 3. 이미지 배열을 벡터로 복원 (배치 처리)
    print("\n🔍 Reconstructing vectors from image arrays...")
    reconstructed_vectors = []
    for image_array in smashed_images:
        vec = image_to_vector(
            image_array,
            original_dim=config["original_dim"],
            vmin=config["vmin"],
            vmax=config["vmax"]
        )
        reconstructed_vectors.append(vec)

    reconstructed_vectors = np.array(reconstructed_vectors, dtype=np.float32)

    print(f"✅ Vectors reconstructed successfully.")
    print(f"   - Vector batch shape: {reconstructed_vectors.shape}")

    # 4. 복원된 벡터를 서버의 다음 로직에 사용
    # 예: pre-train 모델의 입력으로 사용하거나, DB와 유사도 비교 등
    print("\n➡️  Passing reconstructed vectors to the next server-side logic...")
    # server_model.predict(reconstructed_vectors)

    print("\n🎉 Server-side process completed!")

if __name__ == "__main__":
    main()
