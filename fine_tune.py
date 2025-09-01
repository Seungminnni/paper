#!/usr/bin/env python3
"""
Fine-tune Script for Medical Data BERT Model
의료 데이터에 맞게 조정된 BERT 모델 파인튜닝
"""

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
        hidden_states = outputs.hidden_states[-8]
        loss = outputs.loss
        return logits, loss, hidden_states

def main():
    print("🔧 Starting Fine-tuning for Medical Data BERT Model")
    print("=" * 60)

    # 데이터 로드 및 전처리
    print("📊 Loading and preprocessing data...")
    data_A = pd.read_csv("output3.csv")  # 환자 정보 데이터셋
    data_B = pd.read_csv("infected.csv")  # 감염 상태 데이터셋

    SAMPLE_SIZE = 1000
    if len(data_A) > SAMPLE_SIZE:
        data_A = data_A.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"📈 Sampled {SAMPLE_SIZE} patients from dataset")

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
    print(f"📋 Sample X_train: {X_train[:2]}")
    print(f"🏥 Sample Y_train: {Y_train[:10]}")

    # BERT 토크나이저 및 모델 로드
    print("\n🤖 Loading BERT tokenizer and pre-trained model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_path = "Pre_train_final_BERT_Based.pt"

    if os.path.exists(model_path):
        model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.load_state_dict(torch.load(model_path))
        print("✅ Pre-trained model loaded successfully")
    else:
        print("⚠️  Pre-trained model not found, using base model")
        model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

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
    train_size = 0.8
    train_dataset, val_dataset = train_test_split(dataset, test_size=1-train_size, random_state=42)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    print(f"📦 Train dataset: {len(train_dataset)} samples")
    print(f"📦 Validation dataset: {len(val_dataset)} samples")

    # GPU 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # 옵티마이저 및 학습률 설정 (Fine-tuning용 낮은 학습률)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    epochs = 20

    print(f"\n🏃 Starting fine-tuning for {epochs} epochs...")
    print("=" * 60)

    # 학습 루프
    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs[1]
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # 마지막 에포크에서만 모델 저장
        if epoch == epochs - 1:
            model_save_path = "Fine_tuned_final_BERT_Based.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved: {model_save_path}")

        # 검증
        print("🔍 Validating...")
        model.eval()
        val_accuracy = 0

        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            val_accuracy += (logits.argmax(axis=1) == label_ids).mean().item()

        print(f"Validation Accuracy: {val_accuracy / len(val_dataloader):.4f}")

    print("\n Fine-tuning completed successfully!")
    print("Final model saved as: Fine_tuned_final_BERT_Based.pt")

if __name__ == "__main__":
    main()
