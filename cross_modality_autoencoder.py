# pylint: disable=all
# type: ignore
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Input, Dense, Flatten, Reshape, 
                                     Conv2D, Conv2DTranspose, Embedding)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# GPU 설정
print("=== GPU 설정 확인 ===")

# tensorflow-metal GPU 설정
try:
    # GPU 장치 확인
    gpus = tf.config.list_physical_devices('GPU')
    print(f"감지된 GPU: {gpus}")
    
    if gpus:
        try:
            # GPU 메모리 점진적 할당 설정
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU 설정 완료: {len(gpus)}개 GPU 사용 가능")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            gpu_available = True
        except RuntimeError as e:
            print(f"⚠️ GPU 설정 오류: {e}")
            gpu_available = False
    else:
        print("❌ GPU를 찾을 수 없습니다. CPU를 사용합니다.")
        gpu_available = False
        
except Exception as e:
    print(f"❌ GPU 초기화 오류: {e}")
    gpus = []
    gpu_available = False

# 현재 사용 중인 디바이스 확인
print(f"TensorFlow 버전: {tf.__version__}")
print(f"사용 가능한 물리적 디바이스: {tf.config.list_physical_devices()}")

if gpu_available:
    print("🚀 GPU 가속을 사용합니다!")
else:
    print("🔧 CPU 최적화를 사용합니다.")

# MNIST 데이터를 읽고 신경망에 입력할 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# 텍스트 레이블을 원-핫 인코딩
y_train_onehot = to_categorical(y_train, 10)
y_test_onehot = to_categorical(y_test, 10)

zdim = 32  # 잠재 공간의 차원
text_dim = 64  # 텍스트 임베딩 차원

print("=== 간단한 Cross-Modality Autoencoder ===")

# 사용할 디바이스 결정
device_name = '/GPU:0' if gpu_available else '/CPU:0'
print(f"모델 학습에 사용할 디바이스: {device_name}")

with tf.device(device_name):
    # ===================== 모델 1: Text-to-Image =====================
    print("\n--- Text-to-Image 모델 ---")
    
    # Text Encoder (간단한 Dense 레이어)
    text_input = Input(shape=(10,), name='text_input')  # 원-핫 인코딩된 레이블
    x = Dense(128, activation='relu', name='text_dense1')(text_input)
    x = Dense(256, activation='relu', name='text_dense2')(x)
    text_encoded = Dense(zdim, activation='relu', name='text_latent')(x)
    
    text_encoder = Model(text_input, text_encoded, name='text_encoder')
    
    # Image Decoder (CNN Transpose)
    latent_input = Input(shape=(zdim,), name='latent_input')
    x = Dense(7 * 7 * 32, activation='relu', name='decode_dense')(latent_input)
    x = Reshape((7, 7, 32), name='decode_reshape')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2), name='deconv1')(x)
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=(2, 2), name='deconv2')(x)
    image_output = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', name='image_output')(x)
    
    image_decoder = Model(latent_input, image_output, name='image_decoder')
    
    # Text-to-Image 모델 결합
    text_to_image = Model(text_input, image_decoder(text_encoder(text_input)), name='text_to_image')
    text_to_image.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("Text Encoder 구조:")
    text_encoder.summary()
    print("\nImage Decoder 구조:")
    image_decoder.summary()
    
    # ===================== 모델 2: Image-to-Text =====================
    print("\n--- Image-to-Text 모델 ---")
    
    # Image Encoder (CNN)
    image_input = Input(shape=(28, 28, 1), name='image_input')
    x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2), name='conv1')(image_input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2), name='conv2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='img_dense1')(x)
    image_encoded = Dense(zdim, activation='relu', name='img_latent')(x)
    
    image_encoder = Model(image_input, image_encoded, name='image_encoder')
    
    # Text Decoder (Dense 레이어)
    latent_to_text_input = Input(shape=(zdim,), name='latent_to_text_input')
    x = Dense(128, activation='relu', name='text_decode_dense1')(latent_to_text_input)  
    x = Dense(64, activation='relu', name='text_decode_dense2')(x)
    text_output = Dense(10, activation='softmax', name='text_classification')(x)
    
    text_decoder = Model(latent_to_text_input, text_output, name='text_decoder')
    
    # Image-to-Text 모델 결합
    image_to_text = Model(image_input, text_decoder(image_encoder(image_input)), name='image_to_text')
    image_to_text.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Image Encoder 구조:")
    image_encoder.summary()
    print("\nText Decoder 구조:")
    text_decoder.summary()

print("\n=== 모델 학습 시작 ===")

with tf.device(device_name):
    # 1. Text-to-Image 모델 학습
    print("\n🚀 1. Text → Image 모델 학습 중...")
    history1 = text_to_image.fit(
        y_train_onehot, x_train,
        epochs=5, batch_size=128,
        validation_data=(y_test_onehot, x_test),
        verbose=1
    )
    
    # 2. Image-to-Text 모델 학습  
    print("\n🚀 2. Image → Text 모델 학습 중...")
    history2 = image_to_text.fit(
        x_train, y_train_onehot,
        epochs=5, batch_size=128,
        validation_data=(x_test, y_test_onehot), 
        verbose=1
    )

print("\n=== 실험 및 결과 시각화 ===")

# 실험 1: Text → Image 생성
print("\n실험 1: 텍스트 레이블로부터 이미지 생성")
test_labels = np.eye(10)  # 0~9 각 숫자의 원-핫 벡터
generated_images = text_to_image.predict(test_labels)

plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Label: {i}')
    plt.xticks([])
    plt.yticks([])
plt.suptitle('Text → Image: 각 숫자 레이블로부터 생성된 이미지')
plt.tight_layout()
plt.show()

# 실험 2: Image → Text 분류
print("\n실험 2: 이미지로부터 텍스트 레이블 예측")
predicted_labels = image_to_text.predict(x_test[:10])
predicted_classes = np.argmax(predicted_labels, axis=1)
actual_classes = y_test[:10]

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'True: {actual_classes[i]}')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 10, i+11)
    plt.bar(range(10), predicted_labels[i])
    plt.title(f'Pred: {predicted_classes[i]}')
    plt.xticks(range(10))
    plt.ylim(0, 1)
plt.suptitle('Image → Text: 이미지로부터 레이블 예측')
plt.tight_layout()
plt.show()

# 실험 3: Cross-Modality 변환 실험
print("\n실험 3: Cross-Modality 변환 (Text → Image → Text)")

# 텍스트에서 이미지 생성
generated_images_full = text_to_image.predict(test_labels)

# 생성된 이미지에서 다시 텍스트 예측
reconstructed_labels = image_to_text.predict(generated_images_full)
reconstructed_classes = np.argmax(reconstructed_labels, axis=1)

plt.figure(figsize=(15, 9))
for i in range(10):
    # 원본 레이블
    plt.subplot(3, 10, i+1)
    plt.text(0.5, 0.5, f'{i}', fontsize=20, ha='center', va='center')
    plt.title(f'Input: {i}')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 생성된 이미지
    plt.subplot(3, 10, i+11)
    plt.imshow(generated_images_full[i].reshape(28, 28), cmap='gray')
    plt.title('Generated Image')
    plt.xticks([])
    plt.yticks([])
    
    # 재구성된 레이블
    plt.subplot(3, 10, i+21)
    plt.bar(range(10), reconstructed_labels[i])
    plt.title(f'Output: {reconstructed_classes[i]}')
    plt.xticks(range(10))
    plt.ylim(0, 1)

plt.suptitle('Cross-Modality: Text → Image → Text')
plt.tight_layout()
plt.show()

# 성능 평가
print("\n=== 최종 성능 평가 ===")
text_to_image_loss = text_to_image.evaluate(y_test_onehot, x_test, verbose=0)
image_to_text_metrics = image_to_text.evaluate(x_test, y_test_onehot, verbose=0)

print(f"📊 Text → Image MSE Loss: {text_to_image_loss[0]:.4f}")
print(f"📊 Text → Image MAE: {text_to_image_loss[1]:.4f}")
print(f"📊 Image → Text Loss: {image_to_text_metrics[0]:.4f}")
print(f"📊 Image → Text Accuracy: {image_to_text_metrics[1]:.4f}")

# Cross-modality 정확도 계산
cross_modality_accuracy = np.mean(np.arange(10) == reconstructed_classes)
print(f"📊 Cross-Modality Accuracy (Text→Image→Text): {cross_modality_accuracy:.4f}")

print(f"\n🎉 GPU 가속 Cross-Modality Autoencoder 완료!")
print(f"🔧 사용된 디바이스: {device_name}")
