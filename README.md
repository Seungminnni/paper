# 의료 데이터 기반 프라이버시 보호 연구
## Medical Data Privacy Protection Research using BERT

이 프로젝트는 환자 데이터를 활용한 BERT 기반 감염 예측 모델을 개발하고, 순환 은폐 기법으로 프라이버시를 보호하는 연구입니다.

## 📋 파일 구조

```
├── pre_train.py           # BERT 모델 프리 트레이닝
├── fine_tune.py           # 모델 파인튜닝
├── server_side.py         # 서버 사이드 Smashed Data 생성 (500개 샘플)
├── client_side.py         # 클라이언트 사이드 Smashed Data 생성 (300개 샘플)
├── run_all.sh            # 전체 파이프라인 배치 실행
├── requirements.txt       # 필요한 패키지 목록
└── README.md             # 이 파일
```

## 🚀 사용 방법

### 1. 환경 설정

```bash
# 가상환경 활성화 (이미 활성화되어 있다고 가정)
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

다음 CSV 파일들이 필요합니다:
- `output1.csv`: 환자 기본 정보
- `output3.csv`: 추가 환자 데이터
- `infected.csv`: 감염 상태 레이블
- `random_500.csv`: 500개 샘플 데이터 (서버용)
- `random_300.csv`: 300개 샘플 데이터 (클라이언트용)

### 3. 실행 방법

#### 개별 실행
```bash
# 1. 프리 트레이닝
python pre_train.py

# 2. 파인튜닝
python fine_tune.py

# 3. 서버 사이드 Smashed Data 생성
python server_side.py

# 4. 클라이언트 사이드 Smashed Data 생성
python client_side.py
```

#### 전체 파이프라인 실행
```bash
# 모든 단계를 순차적으로 실행
chmod +x run_all.sh
./run_all.sh
```

## � 연구 파이프라인

### 1. Pre-training (`pre_train.py`)
- **목적**: BERT 모델을 의료 데이터에 맞게 프리 트레이닝
- **입력**: `output1.csv`, `infected.csv`
- **출력**: `Pre_train_epoch*_BERT_Based.pt`
- **특징**:
  - 1000개 샘플 사용
  - 10 에폭 학습
  - 8번째 레이어 hidden states 추출

### 2. Fine-tuning (`fine_tune.py`)
- **목적**: 프리 트레이닝된 모델을 추가 학습
- **입력**: `output3.csv`, `infected.csv`, `Pre_train_epoch10_BERT_Based.pt`
- **출력**: `Fine_tuned_epoch*_BERT_Based.pt`
- **특징**:
  - 낮은 학습률 (2e-6) 사용
  - 20 에폭 추가 학습

### 3. Server-side Smashed Data (`server_side.py`)
- **목적**: 서버에서 Smashed Data 생성
- **입력**: `random_500.csv`, `infected.csv`, `Fine_tuned_epoch20_BERT_Based.pt`
- **출력**: `Dictionary_smashed_data_layer2.csv`
- **특징**:
  - 500개 샘플 처리
  - 5번째 레이어 hidden states 사용

### 4. Client-side Smashed Data (`client_side.py`)
- **목적**: 클라이언트에서 Smashed Data 생성
- **입력**: `random_300.csv`, `infected.csv`, `Fine_tuned_epoch20_BERT_Based.pt`
- **출력**: `Client_smashed_data_layer2.csv`
- **특징**:
  - 300개 샘플 처리
  - 5번째 레이어 hidden states 사용

## 🔧 기술 스택

- **언어**: Python 3.12
- **머신러닝**: PyTorch, Transformers
- **데이터 처리**: Pandas, NumPy
- **하드웨어**: Apple Silicon (MPS 지원)

## 📈 주요 성과

1. **모델 성능**: Pre-training + Fine-tuning으로 안정적인 학습
2. **프라이버시 보호**: BERT hidden states를 통한 데이터 변환
3. **데이터 효율성**: 500개 샘플로 빠른 실험 가능
4. **유사도 분석**: 유클리드 거리 기반 정확도 및 통계 분석

## 🔒 보안 메커니즘

- **BERT 기반 변환**: 텍스트 데이터를 768차원 벡터로 변환
- **Hidden States 추출**: 5번째 레이어의 특징 벡터 사용
- **익명화**: 원본 데이터와의 연결성 제거

## � 참고사항

- Apple Silicon Mac에서 MPS를 활용한 GPU 가속 지원
- 각 단계마다 모델이 저장되어 중간에 중단해도 재시작 가능
- Smashed Data는 원본 데이터를 복원할 수 없는 방식으로 변환

## 🎯 연구 목표 달성도

- ✅ **환자 데이터 기반 예측 모델**: BERT 기반 감염 분류 성공
- ✅ **순환 은폐 프레임워크**: Hidden states를 통한 Smashed Data 생성
- ✅ **프라이버시 보호**: 데이터 익명화 및 유사도 분석 완료
- ✅ **시각화**: t-SNE를 통한 데이터 분포 분석 완료

---

**연구자**: Seungmin
**날짜**: 2025년 9월 1일
