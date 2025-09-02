
import numpy as np
import pandas as pd
import json
import os

def generate_config():
    print("⚙️  Generating configuration for vector-to-image conversion...")

    # 클라이언트가 생성한 smashed data를 기준으로 vmin, vmax를 계산합니다.
    smashed_data_file = "Client_smashed_data_layer2.csv"
    if not os.path.exists(smashed_data_file):
        print(f"❌ Error: '{smashed_data_file}' not found.")
        print("Please run 'client_side.py' first to generate the data.")
        return

    print(f"📊 Loading vectors from '{smashed_data_file}' to determine global scale...")
    train_vectors = pd.read_csv(smashed_data_file).values

    # 1. 전역 vmin, vmax 계산
    vmin = float(train_vectors.min())
    vmax = float(train_vectors.max())

    # 2. 이미지 크기(side) 및 원본 벡터 길이(D) 계산
    D = train_vectors.shape[1]
    side = int(np.ceil(np.sqrt(D)))

    # 3. 설정 파일로 저장
    config = {
        "vmin": vmin,
        "vmax": vmax,
        "original_dim": D,
        "image_side": side
    }

    config_file_path = "vector_image_config.json"
    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"✅ Config file '{config_file_path}' generated successfully!")
    print("📋 Configuration:")
    print(json.dumps(config, indent=4))

if __name__ == "__main__":
    generate_config()
