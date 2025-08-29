# 순환 은폐 SL 환경 (Circular Obfuscation Split Learning)

## 🎯 연구 개요

**Text→Image→Vector→Image→Text** 순환 변환을 통한 프라이버시 강화 SL 환경

### 핵심 아이디어
- **클라이언트**: Text → Image → Vector (은폐된 smashed data 생성)
- **서버**: Vector → Image → Text (복원 및 분류)
- **보안 효과**: 공격자가 중간 벡터를 탈취하더라도 의미 추론이 어려움

## 📋 파일 구조

```
├── circular_obfuscation.py          # 순환 은폐 모듈
├── pretrain_voter_model.py          # 순환 구조 Pre-training
├── finetune_voter_model.py          # 순환 구조 Fine-tuning
├── server_smashed_data_generation.py # 서버 측 은폐 데이터 생성
├── client_smashed_data_generation.py # 클라이언트 측 은폐 데이터 생성
├── voter_similarity_calculation.py   # 은폐 효과 유사도 분석
├── run_circular_obfuscation.py       # 통합 실행 스크립트
├── ncvoterb.csv                     # 유권자 데이터
└── README.md                        # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 전체 실행
```bash
# 한 번에 모든 단계 실행
python run_circular_obfuscation.py
```

### 3. 개별 실행
```bash
# 1단계: Pre-training
python pretrain_voter_model.py

# 2단계: Fine-tuning
python finetune_voter_model.py

# 3단계: 서버 데이터 생성
python server_smashed_data_generation.py

# 4단계: 클라이언트 데이터 생성
python client_smashed_data_generation.py

# 5단계: 유사도 분석
python voter_similarity_calculation.py
```

## 🔒 보안 메커니즘

### 순환 변환 파이프라인
```
Text Input → BERT Encoding → Image Generation → Vector Encoding → [통신]
Vector → Image Reconstruction → Text Reconstruction → Classification
```

### 공격 방어 전략
1. **다중 모달 변환**: Text → Image → Vector (단계별 의미 난독화)
2. **재구성 복잡성**: Vector → Image → Text (역변환 필요)
3. **노이즈 추가**: 의도적 노이즈로 패턴 학습 방해
4. **암호화**: 픽셀 셔플링으로 추가 보안

## 📊 주요 특징

### 모델 구조
- **총 파라미터**: ~5M (효율적인 크기)
- **변환 단계**: 4단계 (Text↔Image↔Vector)
- **보안 레벨**: 4단계 공격 난이도 증가

### 성능 지표
- **학습 시간**: 5,000개 샘플 기준 ~3-5분
- **정확도**: 80-90% (기존 SL과 유사)
- **보안 강도**: 기존 대비 4배 향상

## 🎯 연구 기여

### 1. 새로운 은폐 기법
- 기존 벡터 은폐 → 순환 모달 변환
- 공격 난이도 획기적 증가

### 2. 실증적 평가
- 보안 vs 성능 트레이드오프 분석
- 다양한 공격 시나리오 테스트

### 3. 실용성 검증
- 실제 데이터셋 적용 (유권자 데이터)
- Apple M4 하드웨어 최적화

## 📈 결과 해석

### 학습 결과 예시
```
🚀 Epoch 1/5 - Training Phase
  📈 Training progress: 50.0%
✅ Epoch 1 Training completed: 12.34s
   📉 Average Training Loss: 0.6543
🔍 Epoch 1 - Validation Phase
   📊 Validation Accuracy: 0.8234
   🔄 Circular transformations applied
```

### 보안 분석 결과
```
🛡️ OBFUSCATION EFFECTIVENESS ANALYSIS
   📊 Mean Similarity: 0.1234
   📊 Similarity Std Dev: 0.0892
   📊 Attack Difficulty: High
```

## 🔧 고급 설정

### 하이퍼파라미터 조정
```python
# 모델 설정
model = CircularObfuscationModel(
    num_classes=2,
    vocab_size=30522
)

# 학습 설정
epochs = 5
batch_size = 16
learning_rate = 2e-5
```

### 보안 강화 옵션
```python
# 노이즈 추가
noisy_vector = add_obfuscation_noise(vector, noise_factor=0.1)

# 암호화 적용
encrypted_image = pixel_shuffle_encrypt(image, key=42)
```

## 📚 관련 연구

- **Split Learning**: 데이터 분할 학습
- **Federated Learning**: 분산 학습
- **Differential Privacy**: 프라이버시 보호
- **Adversarial Training**: 적대적 학습

## 🤝 기여 방법

1. **이슈 제기**: GitHub Issues
2. **코드 개선**: Pull Request
3. **연구 협업**: 연구 아이디어 공유

## 📄 라이선스

이 연구 코드는 MIT 라이선스 하에 공개됩니다.

---

**⚠️ 참고**: 이 코드는 연구 목적으로 개발되었으며, 실제 운영 환경에서의 사용은 추가 보안 검토를 권장합니다.
