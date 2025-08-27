#!/usr/bin/env python3
"""
최종 버전: 확인된 컬럼명으로 스플릿 러닝 실행
- 10개의 텍스트/범주형 컬럼 사용
- 개별 벡터 크기 64, 최종 640차원
- 서버 모델 용량 확보 (Dense 1024)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import time
import sys

# -----------------------------------
# Word2Vec Processor
# -----------------------------------
class SimpleWord2VecProcessor:
    def __init__(self, vector_size=64, window=3, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.vocab = None

    def train(self, texts):
        tokenized_texts = [str(text).split() for text in texts if str(text)]
        if not tokenized_texts: return
        self.model = Word2Vec(sentences=tokenized_texts, vector_size=self.vector_size,
                              window=self.window, min_count=self.min_count, workers=4)
        self.vocab = set(self.model.wv.index_to_key)

    def vectorize(self, text):
        if self.model is None: return np.zeros(self.vector_size, dtype=np.float32)
        words = str(text).split()
        word_vectors = [self.model.wv[word] for word in words if word in self.vocab]
        if not word_vectors:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(word_vectors, axis=0).astype(np.float32)

# -----------------------------------
# 데이터 로더
# -----------------------------------
def load_voter_data_structured(filepath="ncvotera.csv", columns_to_use=None, max_samples=5000):
    print(f"📊 Loading structured voter data from '{filepath}'...")
    try:
        # usecols를 사용하여 필요한 컬럼만 읽어 메모리 효율성 증대
        df = pd.read_csv(filepath, usecols=columns_to_use, nrows=max_samples, low_memory=False)
        df = df.fillna('').astype(str)
    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        return None
    except ValueError:
        print(f"오류: 코드의 컬럼 리스트와 실제 파일의 컬럼이 일치하지 않습니다.")
        return None
    
    structured_texts = df[columns_to_use].values.tolist()
    print(f"✅ Loaded {len(structured_texts)} records with {len(columns_to_use)} columns each.")
    return structured_texts

# -----------------------------------
# Client, Server, MLP 모델
# -----------------------------------
class ConcatClient:
    def __init__(self, processors, scalers, columns):
        self.processors = processors
        self.scalers = scalers
        self.columns = columns
        print("✅ ConcatClient initialized.")

    def create_smashed_data(self, structured_texts):
        all_concatenated_vectors = []
        for row_texts in structured_texts:
            vectors_for_this_row = []
            for i, col_name in enumerate(self.columns):
                vector = self.processors[col_name].vectorize(row_texts[i])
                scaled_vector = self.scalers[col_name].transform(vector.reshape(1, -1))
                vectors_for_this_row.append(scaled_vector.flatten())
            
            concatenated_vector = np.concatenate(vectors_for_this_row)
            all_concatenated_vectors.append(concatenated_vector)
            
        smashed_data = np.array(all_concatenated_vectors, dtype=np.float32)
        return smashed_data

class SimpleServer:
    def __init__(self, mlp_model):
        self.model = mlp_model
        print("✅ SimpleServer initialized.")

    def train_model(self, smashed_data, y_true, val_data, callbacks):
        history = self.model.fit(smashed_data, y_true, validation_data=val_data,
                                 epochs=50, batch_size=32, callbacks=callbacks, verbose=1)
        return history

    def predict(self, smashed_data):
        return self.model.predict(smashed_data, batch_size=256)

def build_mlp_predictor(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(1024, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(output_dim, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5), loss='mse')
    return model

# -----------------------------------
# Main Pipeline
# -----------------------------------
def main_pipeline():
    print("🚀 Starting Split Learning (Final Verified Columns Version)")
    print("="*70)
    
    # === 하이퍼파라미터 설정 ===
    MAX_SAMPLES = 5000
    VECTOR_SIZE = 64
    
    # 확인된 실제 컬럼명 중 숫자형(age, zip_code)을 제외한 10개 컬럼
    COLUMNS_TO_USE = [
        'first_name', 'middle_name', 'last_name', 'name_suffix', 
        'gender', 'race', 'ethnic', 'city', 'state', 'birth_place'
    ]
    
    FINAL_VECTOR_DIM = VECTOR_SIZE * len(COLUMNS_TO_USE)
    print(f"Final vector dimension: {VECTOR_SIZE} * {len(COLUMNS_TO_USE)} = {FINAL_VECTOR_DIM}")
    # ==========================

    structured_texts = load_voter_data_structured(columns_to_use=COLUMNS_TO_USE, max_samples=MAX_SAMPLES)
    if structured_texts is None:
        sys.exit() # 오류 발생 시 종료
    
    train_texts, val_texts = train_test_split(structured_texts, test_size=0.3, random_state=42)

    print("\n--- Pre-training Shared Components for each column ---")
    processors = {}; scalers = {}
    df_train = pd.DataFrame(train_texts, columns=COLUMNS_TO_USE)

    for col_name in COLUMNS_TO_USE:
        print(f"  > Training for column: '{col_name}'")
        processors[col_name] = SimpleWord2VecProcessor(vector_size=VECTOR_SIZE)
        processors[col_name].train(df_train[col_name].tolist())
        col_vectors = np.array([processors[col_name].vectorize(text) for text in df_train[col_name]])
        if np.all(col_vectors == 0):
            scalers[col_name] = MinMaxScaler().fit(np.zeros((1,VECTOR_SIZE)))
        else:
            scalers[col_name] = MinMaxScaler().fit(col_vectors)

    client = ConcatClient(processors, scalers, COLUMNS_TO_USE)
    mlp_predictor = build_mlp_predictor(input_dim=FINAL_VECTOR_DIM, output_dim=FINAL_VECTOR_DIM)
    server = SimpleServer(mlp_predictor)

    print("\n--- Starting Split Learning Training Flow ---")
    smashed_train = client.create_smashed_data(train_texts)
    y_train = smashed_train
    smashed_val = client.create_smashed_data(val_texts)
    y_val = smashed_val
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)]
    
    history = server.train_model(smashed_train, y_train, (smashed_val, y_val), callbacks)

    print("\n--- Final Evaluation ---")
    val_pred = server.predict(smashed_val)
    r2 = r2_score(y_val, val_pred)
    
    print(f"\n✅ Final Validation R² Score: {r2:.4f}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('MLP Predictor Training Loss'); plt.xlabel('Epochs'); plt.ylabel('MSE Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.scatter(y_val.flatten(), val_pred.flatten(), alpha=0.05)
    min_val, max_val = min(y_val.min(), val_pred.min()), max(y_val.max(), val_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'Prediction vs True (R²: {r2:.3f})'); plt.xlabel('True Values'); plt.ylabel('Predicted Values'); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig('final_results.png'); plt.show()

if __name__ == "__main__":
    main_pipeline()