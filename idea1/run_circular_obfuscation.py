#!/usr/bin/env python3
# 순환 은폐 SL 환경 실행 스크립트
# 사용법: python run_circular_obfuscation.py

import os
import sys
import subprocess

def run_command(command, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️  Warnings:", result.stderr)
        print(f"✅ {description} 완료!")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 실패: {e}")
        print("Error output:", e.stderr)
        return False
    return True

def main():
    print("🎯 순환 은폐 SL 환경 실행")
    print("📋 실행 순서:")
    print("   1. Pre-training (Text→Image→Vector→Image→Text)")
    print("   2. Fine-tuning (순환 구조 적용)")
    print("   3. Server smashed data 생성")
    print("   4. Client smashed data 생성")
    print("   5. 유사도 분석 (은폐 효과 검증)")
    print("=" * 60)

    # 가상환경 활성화 확인
    if not os.path.exists("venv") and not os.path.exists(".venv"):
        print("⚠️  가상환경이 없습니다. 다음 명령어로 설치하세요:")
        print("   python -m venv venv && source venv/bin/activate && pip install -r requirements.txt")
        return

    # 단계별 실행
    steps = [
        {
            "command": "source .venv/bin/activate && python pretrain_voter_model.py",
            "description": "1단계: 순환 은폐 Pre-training"
        },
        {
            "command": "source .venv/bin/activate && python finetune_voter_model.py",
            "description": "2단계: 순환 은폐 Fine-tuning"
        },
        {
            "command": "source .venv/bin/activate && python server_smashed_data_generation.py",
            "description": "3단계: 서버 측 은폐 데이터 생성"
        },
        {
            "command": "source .venv/bin/activate && python client_smashed_data_generation.py",
            "description": "4단계: 클라이언트 측 은폐 데이터 생성"
        },
        {
            "command": "source .venv/bin/activate && python voter_similarity_calculation.py",
            "description": "5단계: 은폐 효과 유사도 분석"
        }
    ]

    for step in steps:
        success = run_command(step["command"], step["description"])
        if not success:
            print(f"\n❌ {step['description']} 실패로 실행 중단")
            break

    print("\n" + "=" * 60)
    print("🎉 순환 은폐 SL 환경 실행 완료!")
    print("📊 생성된 파일들:")
    print("   • Pre-trained_voter_final.pt (학습된 모델)")
    print("   • Fine-tuned_voter_final.pt (미세조정 모델)")
    print("   • Dictionary_smashed_data.csv (서버 은폐 데이터)")
    print("   • Client_smashed_data.csv (클라이언트 은폐 데이터)")
    print("   • similarity_analysis.txt (유사도 분석 결과)")
    print("=" * 60)

    # 결과 파일 존재 확인
    output_files = [
        "Pre-trained_voter_final.pt",
        "Fine-tuned_voter_final.pt",
        "Dictionary_smashed_data.csv",
        "Client_smashed_data.csv"
    ]

    print("\n📁 생성된 파일 확인:")
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB 단위
            print(f"   ✅ {file} ({size:.1f} KB)")
        else:
            print(f"   ❌ {file} (생성되지 않음)")

if __name__ == "__main__":
    main()
