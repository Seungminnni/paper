# 프라이버시 보호를 위한 크로스 모달 데이터 파이프라인 연구

이 프로젝트는 스플릿 러닝(Split Learning) 환경을 가정하여, 클라이언트의 의료 데이터를 안전하게 서버로 전송하는 파이프라인을 구축하고, 그 과정에서 발생하는 정보 손실을 측정하는 연구입니다.

## 💡 핵심 아이디어

클라이언트의 민감한 원본 데이터를 직접 전송하는 대신, 다음과 같은 프라이버시 보호 파이프라인을 구현합니다.

1.  **클라이언트**: 원본 텍스트 데이터를 BERT 모델을 통해 **특징 벡터(Smashed Data)**로 변환합니다.
2.  **클라이언트**: 추출된 벡터를 **이미지 배열(Image Array)**로 한번 더 변환하여, 원본 벡터의 숫자 값을 시각적으로 알아보기 힘든 형태로 위장합니다.
3.  **서버**: 전송받은 이미지 배열을 다시 **벡터**로 복원하여 후속 분석에 사용합니다.

이 연구의 주된 목표는 **`벡터 -> 이미지 -> 벡터`** 로 이어지는 파이프라인이 원본 데이터의 정보를 얼마나 잘 보존하는지 정량적으로 측정하는 것입니다.

## 📂 파일 설명

| 파일명                               | 역할                                                                                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `pre_train.py`                       | BERT 모델을 의료 데이터로 사전 학습합니다. (샘플 수 제한 없음)                                                         |
| `fine_tune.py`                       | 사전 학습된 모델을 감염병 예측 태스크에 맞게 미세 조정합니다. (샘플 수 제한 없음)                                       |
| `client_smashed_data_generation.py`  | **(1단계)** 클라이언트 측 원본 벡터(`Client_smashed_data_layer2.csv`)를 생성합니다.                                   |
| `server_dictionary_generation.py`  | **(2단계)** 서버의 비교 기준이 될 벡터 사전(`Dictionary_smashed_data_layer2.csv`)을 생성합니다.                      |
| `generate_config.py`                 | **(3단계)** 벡터를 이미지로 변환하는 데 필요한 공유 설정(`vector_image_config.json`)을 생성합니다.                    |
| `client_side.py`                     | **(4단계)** 클라이언트 원본 벡터를 이미지 배열로 변환하여 `smashed_images.npy` 파일로 저장합니다.                      |
| `server_side.py`                     | **(5단계)** 전송된 `smashed_images.npy`를 다시 벡터로 복원하여 `restored_client_vectors.csv` 파일로 저장합니다. |
| `similarity_analysis.py`             | **(최종 분석)** 두 벡터 파일(.csv)을 비교하여 코사인 유사도를 측정하고, 정보 손실률을 분석합니다.                   |
| `requirements.txt`                   | 필요한 패키지 목록                                                                                                    |
| `run_all.sh`                         | (수정 필요) 전체 파이프라인을 실행하는 쉘 스크립트                                                                    |

## 🚀 실행 순서 (정보 손실률 측정)

연구 목표에 맞춰, 이미지 변환 파이프라인이 데이터에 미치는 영향을 분석하는 전체 실행 순서입니다.

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
# 1. 사전 학습
python pre_train.py

# 2. 미세 조정
python fine_tune.py
```

### 3. 데이터 생성 및 변환 파이프라인 실행

```bash
# 3. 클라이언트 원본 벡터 생성
python client_smashed_data_generation.py

# 4. 이미지 변환 규칙 생성
python generate_config.py

# 5. 클라이언트, 벡터를 이미지로 변환하여 저장
python client_side.py

# 6. 서버, 이미지를 다시 벡터로 복원하여 저장
python server_side.py
```

### 4. 최종 분석 실행

```bash
# 7. 원본 벡터와 복원된 벡터를 비교하여 정보 손실률 분석
python similarity_analysis.py
```

실행이 완료되면, 터미널에서 평균 유사도 등 정량적인 분석 결과를 확인할 수 있으며, `information_loss_distribution.png` 라는 이름으로 유사도 분포 그래프가 저장됩니다.

## 🔧 기술 스택

- **언어**: Python 3.12
- **머신러닝**: PyTorch, Transformers
- **데이터 처리**: Pandas, NumPy

---

**연구자**: Seungmin
**날짜**: 2025년 9월 2일 (업데이트)