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
    model_path = "Fine_tuned_final_BERT_Based.pt"

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

    # Hidden states 결합 및 저장
    print("\n💾 Processing and saving smashed data...")
    hidden_states_concat = torch.cat(hidden_states_list, dim=0)
    hidden_states_concat = hidden_states_concat[:, 0, :].cpu().detach().numpy()  # [CLS] 토큰의 hidden states

    hidden_states_df = pd.DataFrame(hidden_states_concat)
    output_file = "Dictionary_smashed_data_layer2.csv"
    hidden_states_df.to_csv(output_file, index=False)

    print("✅ Server-side smashed data saved successfully!")
    print(f"📁 Output file: {output_file}")
    print(f"📊 Data shape: {hidden_states_concat.shape}")
    print(f"🔢 Features: {hidden_states_concat.shape[1]} dimensions")
    print(f"📈 Samples: {hidden_states_concat.shape[0]} patients")

    # 통계 정보 출력
    print("\n📊 Smashed Data Statistics:")
    print(f"   • Mean: {np.mean(hidden_states_concat):.6f}")
    print(f"   • Std: {np.std(hidden_states_concat):.6f}")
    print(f"   • Min: {np.min(hidden_states_concat):.6f}")
    print(f"   • Max: {np.max(hidden_states_concat):.6f}")

    print("\n🎉 Server-side smashed data generation completed!")
    print("🔒 Data is now anonymized and ready for privacy-preserving analysis")

if __name__ == "__main__":
    main()
