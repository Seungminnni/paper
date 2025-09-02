#!/usr/bin/env python3
"""
Client-side Smashed Data Generation
클라이언트 사이드 Smashed Data 생성 (300개 샘플)
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
    print("💻 Starting Client-side Smashed Data Generation")
    print("=" * 60)

    # 데이터 로드 및 전처리
    print("📊 Loading data for client-side processing...")
    data_A = pd.read_csv("random_100.csv")  # 100개 샘플 데이터셋
    data_B = pd.read_csv("infected.csv")  # 감염 상태 데이터셋
    model_path = "Fine_tuned_final_BERT_Based.pt"

    print(f"📈 Processing {len(data_A)} patient records for client-side")

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
        model.load_state_dict(torch.load(model_path), strict=False)
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

    # Hidden states 추출
    print("\n🔍 Extracting hidden states for smashed data generation...")
    hidden_states_list = []

    for batch_idx, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs[2]  # Custom model의 hidden_states
        hidden_states_list.append(hidden_states)

        if batch_idx % 5 == 0:
            print(f"  Processed batch {batch_idx + 1}/{len(dataloader)}")

    # Hidden states 결합
    print("\n💾 Processing and converting smashed data to image array...")
    hidden_states_concat = torch.cat(hidden_states_list, dim=0)
    hidden_states_concat = hidden_states_concat[:, 0, :].cpu().detach().numpy()  # [CLS] 토큰의 hidden states

    # --- 아이디어 적용: 벡터를 이미지 배열로 변환 ---
    import json

    def vector_to_image(v, side=None, vmin=None, vmax=None, robust=False, cmap=None):
        v = np.asarray(v, dtype=np.float32).copy()
        if robust:
            lo, hi = np.percentile(v, [1, 99])
            v = np.clip(v, lo, hi)
            vmin, vmax = lo, hi
        else:
            assert vmin is not None and vmax is not None, "train에서 얻은 vmin/vmax를 넘겨주세요."
        v = (v - vmin) / (vmax - vmin + 1e-8)
        v = np.clip(v, 0.0, 1.0)
        if side is None:
            side = int(np.ceil(np.sqrt(len(v))))
        pad = side*side - len(v)
        if pad > 0:
            v = np.pad(v, (0, pad), constant_values=0.0)
        img = v.reshape(side, side, 1)
        if cmap is not None:
            import matplotlib.cm as cm
            rgb = cm.get_cmap(cmap)(img[..., 0])[..., :3]
            return rgb.astype(np.float32)
        return img

    # 1. 공유 설정 파일 로드
    with open("vector_image_config.json", "r") as f:
        config = json.load(f)

    # 2. 벡터를 이미지 배열로 변환 (배치 처리)
    smashed_images = []
    for vector in hidden_states_concat:
        img = vector_to_image(
            vector,
            side=config["image_side"],
            vmin=config["vmin"],
            vmax=config["vmax"]
        )
        smashed_images.append(img)

    smashed_images = np.array(smashed_images, dtype=np.float32)

    print(f"🖼️  Smashed data converted to image arrays.")
    print(f"   - Array batch shape: {smashed_images.shape}")
    print(f"   - Data type: {smashed_images.dtype}")

    # 3. 생성된 이미지 배열을 서버로 전송
    # 이 단계에서 `smashed_images` 배열을 직렬화(예: pickle)하여
    # 네트워크를 통해 서버로 전송하는 로직을 구현해야 합니다.
    # 예: send_to_server(pickle.dumps(smashed_images))
    print("\n🎉 Client-side process completed!")
    print("🔒 Smashed image arrays are ready to be sent to the server.")

if __name__ == "__main__":
    main()
