# Paper Project - Individual CSV Data Communication Test

본 프로젝트는 CSV 데이터를 텍스트→이미지→텍스트 형태로 통신하는 테스트를 수행합니다.

## 개발환경 설정

### Python 가상환경
프로젝트에는 Python 가상환경이 설정되어 있습니다.

### 패키지 설치
```bash
# 가상환경에서 필요한 패키지들 설치
pip install -r requirements.txt
```

### 스크립트 실행
```bash
# individual_data_test.py 실행
python individual_data_test.py
```

## 프로젝트 구조
- `individual_data_test.py`: 메인 테스트 스크립트
- `ncvaa.csv`, `ncvotera.csv`, `ncvoterb.csv`: 테스트용 CSV 데이터
- `requirements.txt`: Python 의존성 패키지 목록

## 주요 기능
- CSV 데이터의 개별 항목들을 고유한 시각적 패턴으로 변환
- 텍스트→이미지→텍스트 통신 테스트
- 19개 컬럼 타입별 최적화된 패턴 생성
- 정확한 데이터 값 보존 검증
