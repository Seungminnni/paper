# BERT-based Split Learning Client-side Preprocessing for ncvotera.csv
# Based on BIOTF_v1.5_Medium.ipynb structure

import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CustomBertForSequenceClassification(BertForSequenceClassification):
    """
    Custom BERT model for extracting hidden states at specific layers
    """
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
        hidden_states = outputs.hidden_states[7]  # 7번째 레이어의 hidden states 추출
        loss = outputs.loss
        return logits, loss, hidden_states

class NCVoteraClientPreprocessor:
    """
    ncvotera.csv 데이터를 위한 클라이언트 측 전처리기
    """
    def __init__(self, model_name='lyeonii/bert-medium', max_len=128):
        self.model_name = model_name
        self.max_len = max_len
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_data(self, file_path):
        """
        ncvotera.csv 데이터 로드 및 기본 전처리
        """
        print(f"Loading data from {file_path}...")
        data = pd.read_csv(file_path)
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        return data
    
    def preprocess_voter_data(self, data, target_column=None):
        """
        유권자 데이터 전처리 및 텍스트 결합
        """
        print("Preprocessing voter data...")
        
        X_train = []
        Y_train = []
        
        # 텍스트 컬럼들을 식별 (숫자가 아닌 컬럼들)
        text_columns = []
        for col in data.columns:
            if data[col].dtype == 'object':
                text_columns.append(col)
        
        print(f"Text columns identified: {text_columns}")
        
        for index, row in data.iterrows():
            # 모든 텍스트 정보를 결합
            voter_info = []
            for col in text_columns:
                if pd.notna(row[col]):
                    voter_info.append(f"{col}: {str(row[col])}")
            
            combined_info = ", ".join(voter_info)
            X_train.append(combined_info)
            
            # 타겟 라벨 생성 (예시: 특정 조건에 따라)
            if target_column and target_column in data.columns:
                Y_train.append(1 if pd.notna(row[target_column]) else 0)
            else:
                # 기본적으로 인덱스 기반 이진 분류 (예시)
                Y_train.append(1 if index % 2 == 0 else 0)
        
        print(f"Generated {len(X_train)} training samples")
        print(f"Sample text: {X_train[0][:200]}...")
        
        return X_train, Y_train
    
    def initialize_model(self, model_path=None):
        """
        BERT 토크나이저 및 모델 초기화
        """
        print("Initializing BERT model and tokenizer...")
        
        # 토크나이저 로드
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # 모델 로드
        if model_path and os.path.exists(model_path):
            print(f"Loading pre-trained model from {model_path}")
            self.model = CustomBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Creating new model")
            self.model = CustomBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
        
        self.model.to(self.device)
        print(f"Model moved to {self.device}")
        
    def tokenize_data(self, X_train):
        """
        텍스트 데이터를 BERT 입력 형식으로 토크나이징
        """
        print("Tokenizing data...")
        
        input_ids = []
        attention_masks = []
        
        for i, text in enumerate(X_train):
            if i % 100 == 0:
                print(f"Tokenizing progress: {i}/{len(X_train)}")
                
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        print(f"Tokenization complete. Shape: {input_ids.shape}")
        return input_ids, attention_masks
    
    def create_dataset(self, input_ids, attention_masks, labels):
        """
        PyTorch 데이터셋 및 데이터로더 생성
        """
        print("Creating dataset and dataloader...")
        
        labels = torch.tensor(labels)
        dataset = TensorDataset(input_ids, attention_masks, labels)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        print(f"Dataset created with {len(dataset)} samples")
        return dataset, dataloader
    
    def generate_smashed_data(self, dataloader, output_file=None):
        """
        클라이언트 측 smashed data 생성
        """
        print("Generating smashed data...")
        
        self.model.eval()
        hidden_states_list = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
                
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }
                
                outputs = self.model(**inputs)
                hidden_states = outputs[2]  # hidden states from layer 7
                
                # CLS token의 hidden states 추출 ([CLS] 토큰은 첫 번째 토큰)
                cls_hidden_states = hidden_states[:, 0, :]
                hidden_states_list.append(cls_hidden_states.cpu())
                all_labels.extend(batch[2].cpu().numpy())
        
        # 모든 hidden states 결합
        hidden_states_concat = torch.cat(hidden_states_list, dim=0)
        hidden_states_numpy = hidden_states_concat.numpy()
        
        print(f"Smashed data shape: {hidden_states_numpy.shape}")
        
        # 결과 저장
        if output_file:
            hidden_states_df = pd.DataFrame(hidden_states_numpy)
            hidden_states_df.to_csv(output_file, index=False)
            print(f"Smashed data saved to {output_file}")
            
            # 라벨도 함께 저장
            labels_df = pd.DataFrame({'label': all_labels})
            label_file = output_file.replace('.csv', '_labels.csv')
            labels_df.to_csv(label_file, index=False)
            print(f"Labels saved to {label_file}")
        
        return hidden_states_numpy, all_labels
    
    def run_complete_preprocessing(self, data_file, model_path=None, output_file=None):
        """
        전체 전처리 파이프라인 실행
        """
        print("=== Starting complete preprocessing pipeline ===")
        
        # 1. 데이터 로드
        data = self.load_data(data_file)
        
        # 2. 데이터 전처리
        X_train, Y_train = self.preprocess_voter_data(data)
        
        # 3. 모델 초기화
        self.initialize_model(model_path)
        
        # 4. 토크나이징
        input_ids, attention_masks = self.tokenize_data(X_train)
        
        # 5. 데이터셋 생성
        dataset, dataloader = self.create_dataset(input_ids, attention_masks, Y_train)
        
        # 6. Smashed data 생성
        if output_file is None:
            output_file = "ncvotera_client_smashed_data.csv"
        
        smashed_data, labels = self.generate_smashed_data(dataloader, output_file)
        
        print("=== Preprocessing pipeline completed ===")
        return smashed_data, labels, dataset

def main():
    """
    메인 실행 함수
    """
    # 설정
    data_file = "ncvotera.csv"
    model_path = None  # 사전 훈련된 모델이 있다면 경로 지정
    output_file = "ncvotera_client_smashed_data.csv"
    
    # 전처리기 초기화
    preprocessor = NCVoteraClientPreprocessor()
    
    # 전체 파이프라인 실행
    try:
        smashed_data, labels, dataset = preprocessor.run_complete_preprocessing(
            data_file, model_path, output_file
        )
        
        print(f"\nResults:")
        print(f"Smashed data shape: {smashed_data.shape}")
        print(f"Number of labels: {len(labels)}")
        print(f"Label distribution: {np.bincount(labels)}")
        
    except FileNotFoundError:
        print(f"Error: {data_file} not found. Please ensure the file exists in the current directory.")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main()
