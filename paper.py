
import tensorflow as tf
from tensorflow.keras.layers import Input, TextVectorization, Embedding, LSTM, Dense, Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 기본 설정 및 데이터 준비 ---

# 재현성을 위한 시드 설정
tf.random.set_seed(42)
np.random.seed(42)

def load_patient_data(file_path, n_samples=2000):
    """patients.csv 파일에서 데이터를 로드하고 텍스트 문장을 생성합니다."""
    print(f"--- Loading data from {file_path} ---")
    try:
        df = pd.read_csv(file_path, nrows=n_samples)
        print(f"✅ Successfully loaded {len(df)} records.")

        # 텍스트 생성을 위한 컬럼 선택
        # 여러 인구통계학적 정보를 조합하여 환자의 특징을 나타내는 문장을 만듭니다.
        cols_to_use = ['RACE', 'ETHNICITY', 'GENDER', 'MARITAL', 'BIRTHPLACE']
        df[cols_to_use] = df[cols_to_use].fillna('unknown') # 결측값 처리

        # 선택된 컬럼들을 합쳐 하나의 텍스트 문장으로 만듭니다.
        patient_texts = df[cols_to_use].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()
        print(f"✅ Generated {len(patient_texts)} text samples.")
        print(f"   Example: "{patient_texts[0]}""")
        return patient_texts

    except FileNotFoundError:
        print(f"❌ ERROR: Data file not found at '{file_path}'")
        print("Please ensure 'patients.csv' is in the same directory.")
        return None
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return None

# 모델 하이퍼파라미터
VOCAB_SIZE = 2000  # 어휘 사전의 크기 (데이터가 다양해졌으므로 증가)
MAX_SEQUENCE_LENGTH = 25  # 문장의 최대 길이 (컬럼 조합으로 길어짐)
EMBEDDING_DIM = 128
LATENT_DIM = 256
IMAGE_SIZE = 32

# 데이터 로드
sample_texts = load_patient_data('patients.csv')

# 데이터 로드에 실패하면 스크립트 중단
if sample_texts is None:
    exit()

# 텍스트를 정수 시퀀스로 변환하는 레이어
text_vectorizer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=MAX_SEQUENCE_LENGTH,
    name="text_vectorizer"
)
# 텍스트 데이터로 어휘 사전을 구축
text_vectorizer.adapt(sample_texts)


# --- 2. 새로운 아키텍처 모델 정의 ---

class EndToEndTextToImageSystem:
    def __init__(self, text_vectorizer, embedding_dim, latent_dim, image_size):
        self.text_vectorizer = text_vectorizer
        self.vocab_size = text_vectorizer.get_vocabulary_size()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.image_shape = (image_size, image_size, 3)

    def build_client_model(self):
        text_input = Input(shape=(1,), dtype=tf.string, name="client_text_input")
        vectorized_text = self.text_vectorizer(text_input)
        embedded_text = Embedding(self.vocab_size, self.embedding_dim, name="embedding")(vectorized_text)
        latent_vector = LSTM(self.latent_dim, name="lstm_encoder")(embedded_text)
        x = Dense(512, activation='relu', name="img_gen_dense_1")(latent_vector)
        x = Dense(1024, activation='relu', name="img_gen_dense_2")(x)
        x = Dense(self.image_size * self.image_size * 3, activation='sigmoid', name="img_gen_dense_3")(x)
        image_output = Reshape(self.image_shape, name="client_image_output")(x)
        client_model = Model(text_input, image_output, name="client_text_to_image_model")
        print("✅ Client Model (Text -> Image) built.")
        return client_model

    def build_enhanced_server_model(self):
        """
        서버 측 모델 (Image -> Vector)
        - 손실률을 줄이기 위해 Conv2D 레이어를 여러 층으로 깊게 쌓은 구조
        """
        image_input = Input(shape=self.image_shape, name="server_image_input")

        # --- (A) 이미지 인코더 부분 (더 깊어진 CNN 구조) ---
        # 1차 특징 추출
        x = Conv2D(64, 3, activation='relu', padding='same')(image_input)
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)  # 이미지 크기 절반으로 줄임 (16x16)

        # 2차 특징 추출
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = Conv2D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)  # 이미지 크기 다시 절반으로 줄임 (8x8)
        
        # 3차 특징 추출
        x = Conv2D(256, 3, activation='relu', padding='same')(x)
        x = Conv2D(256, 3, activation='relu', padding='same')(x)
        
        # 특징 요약
        x = GlobalAveragePooling2D(name="gap_pooling")(x)
        
        # 의미 벡터 복원
        reconstructed_latent_vector = Dense(self.latent_dim, name="reconstructed_latent_vector")(x)

        server_model = Model(image_input, reconstructed_latent_vector, name="server_image_to_vector_model")
        print("✅ Enhanced Server Model (Image -> Vector) built.")
        return server_model

    def build_end_to_end_for_training(self, client_model, server_model):
        text_input = client_model.input
        latent_vector_true = client_model.get_layer("lstm_encoder").output
        generated_image = client_model(text_input)
        reconstructed_latent_vector = server_model(generated_image)
        training_model = Model(text_input, reconstructed_latent_vector, name="training_autoencoder")
        print("✅ End-to-End Training Model built.")
        return training_model


# --- 3. 모델 생성, 컴파일 및 학습 ---

print("\n--- Building and Training Model ---")
# 시스템 인스턴스 생성
text_to_image_system = EndToEndTextToImageSystem(text_vectorizer, EMBEDDING_DIM, LATENT_DIM, IMAGE_SIZE)

# 클라이언트와 서버 모델 빌드 (서버는 강화된 버전 사용)
client_model = text_to_image_system.build_client_model()
server_model = text_to_image_system.build_enhanced_server_model()

# 훈련용 모델 빌드 및 컴파일
training_model = text_to_image_system.build_end_to_end_for_training(client_model, server_model)
training_model.compile(optimizer='adam', loss='mse')

# 모델 구조 출력
training_model.summary()

# 데이터셋 생성
# TextVectorization 레이어는 tf.data.Dataset에 최적화되어 있습니다.
train_dataset = tf.data.Dataset.from_tensor_slices(sample_texts).batch(32)

# 훈련을 위한 입력(x)과 정답(y)을 생성하는 함수
# 이 모델의 목표는 텍스트를 입력받아, 그 텍스트의 '의미 벡터'를 복원하는 것입니다.
text_to_vector_model = Model(client_model.input, client_model.get_layer("lstm_encoder").output)

def get_training_data(text_batch):
    original_vectors = text_to_vector_model(text_batch)
    return text_batch, original_vectors

train_dataset = train_dataset.map(get_training_data)

print("\n--- Starting Model Training ---")
# 모델 훈련 (실제 환경에서는 더 많은 epoch 필요)
history = training_model.fit(train_dataset, epochs=15, verbose=1)
print("✅ Training finished.")


# --- 4. 결과 시연 ---

print("\n--- Demonstrating the End-to-End Flow (After Training) ---")

# 1. 테스트할 샘플 선택
test_text = sample_texts[10]
print(f"Original Text: '{test_text}'")
input_text_tensor = tf.constant([test_text])

# 2. 클라이언트: 텍스트를 이미지로 변환
generated_image = client_model.predict(input_text_tensor)

# 3. 서버: 이미지를 중간 벡터로 복원
reconstructed_vector = server_model.predict(generated_image)

# 4. 비교: 원본 텍스트의 실제 중간 벡터와 서버가 복원한 벡터 비교
original_vector = text_to_vector_model.predict(input_text_tensor)
mse_loss = np.mean((original_vector - reconstructed_vector)**2)
print(f"MSE between original and reconstructed vectors (after training): {mse_loss:.6f}")
print("\n💡 A low MSE after training would prove that the image successfully carried the text's information.")

# 5. 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'])
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(1, 3, 2)
plt.imshow(generated_image[0])
plt.title("Generated Mosaic Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.plot(original_vector[0], label='Original Vector', color='blue', alpha=0.7)
plt.plot(reconstructed_vector[0], label='Reconstructed Vector', color='red', linestyle='--', alpha=0.7)
plt.title("Vector Comparison")
plt.legend()
plt.suptitle("End-to-End Pipeline Results (Trained)")
plt.tight_layout()
plt.show()
