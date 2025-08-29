# 순환 은폐 변환 모듈 (Text→Image→Vector→Image→Text)
# 연구 아이디어: 벡터를 이미지로 변환하여 공격 난이도 증가
# 사용법: from circular_obfuscation import CircularObfuscationModel, ClientCircularModel, ServerCircularModel

import torch
import torch.nn as nn
import torch.nn.functional as F

# BERT 관련 import (필요시 주석 해제)
# from transformers import BertTokenizer, BertForSequenceClassification

class CircularObfuscationModel(nn.Module):
    """
    텍스트→이미지→벡터→이미지→텍스트 순환 구조 모델
    공격자가 중간 데이터를 탈취하더라도 의미 추론이 어려움
    """
    def __init__(self, num_classes=2, vocab_size=30522):
        super().__init__()

        # ===== Phase 1: Text → Image 변환 =====
        # 1. Text Encoder (기존 BERT)
        # self.text_encoder = BertForSequenceClassification.from_pretrained(
        #     'bert-base-uncased', num_labels=num_classes
        # )

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
        # 1. Text encoding (BERT 임베딩 시뮬레이션)
        # 실제로는 BERT를 사용하지만 여기서는 간단히 랜덤 임베딩 사용
        batch_size = input_ids.size(0)
        text_embedding = torch.randn(batch_size, 768).to(input_ids.device)  # CLS token

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

class ClientCircularModel(nn.Module):
    """
    클라이언트 측: Text → Image → Vector (은폐된 smashed data 생성)
    """
    def __init__(self):
        super().__init__()
        # Text → Image → Vector 파이프라인
        # self.text_encoder = BertForSequenceClassification.from_pretrained(
        #     'bert-base-uncased', num_labels=2
        # )
        self.image_generator = nn.Sequential(
            nn.Linear(768, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 32 * 32), nn.Sigmoid()
        )
        self.vector_encoder = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 768), nn.LayerNorm(768)
        )

    def forward(self, input_ids, attention_mask):
        # Text → BERT embedding (시뮬레이션)
        batch_size = input_ids.size(0)
        text_embedding = torch.randn(batch_size, 768).to(input_ids.device)

        # BERT embedding → Image
        image = self.image_generator(text_embedding)
        image = image.view(-1, 7, 32, 32)

        # Image → Vector (smashed data)
        smashed_vector = self.vector_encoder(image)

        return smashed_vector

class ServerCircularModel(nn.Module):
    """
    서버 측: Vector → Image → Text (복원 및 분류)
    """
    def __init__(self, num_classes=2, vocab_size=30522):
        super().__init__()
        # Vector → Image → Text 파이프라인
        self.vector_decoder = nn.Sequential(
            nn.Linear(768, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 32 * 32), nn.Sigmoid()
        )
        self.image_decoder = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(),
            nn.Linear(64 * 8 * 8, 768), nn.LayerNorm(768)
        )
        self.classifier = nn.Linear(768, num_classes)
        self.text_decoder = nn.Linear(768, vocab_size)

    def forward(self, smashed_vector, labels=None):
        # Vector → Image
        reconstructed_image = self.vector_decoder(smashed_vector)
        reconstructed_image = reconstructed_image.view(-1, 7, 32, 32)

        # Image → Text embedding
        text_embedding = self.image_decoder(reconstructed_image)

        # 분류
        classification_logits = self.classifier(text_embedding)

        # Text tokens (재구성된 텍스트)
        text_logits = self.text_decoder(text_embedding)

        if labels is not None:
            loss = F.cross_entropy(classification_logits, labels)
            return classification_logits, loss, text_logits
        return classification_logits, text_logits

# 사용 예시:
"""
# 1. 모델 초기화
model = CircularObfuscationModel(num_classes=2)

# 2. 순환 변환 수행
outputs = model(input_ids, attention_mask, labels, return_all=True)

# 3. 결과 확인
print(f"Generated Image Shape: {outputs['generated_image'].shape}")
print(f"Smashed Vector Shape: {outputs['smashed_vector'].shape}")
print(f"Classification Accuracy: {outputs['classification_logits'].argmax(dim=1) == labels}")
"""

def add_obfuscation_noise(tensor, noise_factor=0.05):
    """
    은폐 강화를 위한 노이즈 추가
    공격자가 패턴을 학습하기 어렵게 함
    """
    noise = torch.randn_like(tensor) * noise_factor
    return tensor + noise

def pixel_shuffle_encrypt(image, key=42):
    """
    픽셀 셔플링으로 추가 암호화
    간단한 암호화 기법으로 보안 강화
    """
    torch.manual_seed(key)
    batch_size, channels, height, width = image.shape

    # 픽셀 위치를 무작위로 섞음
    indices = torch.randperm(height * width)
    shuffled_image = image.view(batch_size, channels, -1)
    shuffled_image = shuffled_image[:, :, indices]
    shuffled_image = shuffled_image.view(batch_size, channels, height, width)

    return shuffled_image

def add_obfuscation_noise(tensor, noise_factor=0.05):
    """
    은폐 강화를 위한 노이즈 추가
    공격자가 패턴을 학습하기 어렵게 함
    """
    noise = torch.randn_like(tensor) * noise_factor
    return tensor + noise

def pixel_shuffle_encrypt(image, key=42):
    """
    픽셀 셔플링으로 추가 암호화
    간단한 암호화 기법으로 보안 강화
    """
    torch.manual_seed(key)
    batch_size, channels, height, width = image.shape

    # 픽셀 위치를 무작위로 섞음
    indices = torch.randperm(height * width)
    shuffled_image = image.view(batch_size, channels, -1)
    shuffled_image = shuffled_image[:, :, indices]
    shuffled_image = shuffled_image.view(batch_size, channels, height, width)

    return shuffled_image

# 사용 예시:
"""
# 1. 모델 초기화
model = CircularObfuscationModel(num_classes=2)

# 2. 순환 변환 수행
outputs = model(input_ids, attention_mask, labels, return_all=True)

# 3. 결과 확인
print(f"Generated Image Shape: {outputs['generated_image'].shape}")
print(f"Smashed Vector Shape: {outputs['smashed_vector'].shape}")
print(f"Classification Accuracy: {outputs['classification_logits'].argmax(dim=1) == labels}")

# 4. 보안 테스트
noisy_vector = add_obfuscation_noise(outputs['smashed_vector'])
encrypted_image = pixel_shuffle_encrypt(outputs['generated_image'])
"""
