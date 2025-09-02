#!/usr/bin/env python3
"""
Client-side Smashed Data Generation (Step 2: Image Conversion & Sending)
클라이언트 사이드 Smashed Data 생성 (2단계: 이미지 변환 및 전송)
"""

import os
import pandas as pd
import numpy as np
import json


def def_vector_to_image(v, side=None, vmin=None, vmax=None, robust=False, cmap=None):
    v = np.asarray(v, dtype=np.float32).copy()
    if robust:
        lo, hi = np.percentile(v, [1, 99])
        v = np.clip(v, lo, hi)
        vmin, vmax = lo, hi
    else:
        assert vmin is not None and vmax is not None, "train에서 얻은 vmin/vmax를 넘겨주세요."
    v = (v - vmin) / (vmax - vmin + 1e-8)
    v = np.clip(v, 0.0, 1.0)
    if side is None:
        side = int(np.ceil(np.sqrt(len(v))))
    pad = side*side - len(v)
    if pad > 0:
        v = np.pad(v, (0, pad), constant_values=0.0)
    img = v.reshape(side, side, 1)
    if cmap is not None:
        import matplotlib.cm as cm
        rgb = cm.get_cmap(cmap)(img[..., 0])[..., :3]
        return rgb.astype(np.float32)
    return img

def main():
    print("💻 Starting Client-side Process (Image Conversion & Sending)")
    print("=" * 60)

    # 필요한 파일 확인
    smashed_data_file = "Client_smashed_data_layer2.csv"
    config_file = "vector_image_config.json"

    if not os.path.exists(smashed_data_file):
        print(f"❌ Error: Smashed data file not found at '{smashed_data_file}'!")
        print("Please run 'client_smashed_data_generation.py' first.")
        return

    if not os.path.exists(config_file):
        print(f"❌ Error: Config file not found at '{config_file}'!")
        print("Please run 'generate_config.py' first.")
        return

    # 1. Smashed data 벡터 로드
    print(f"📊 Loading smashed data vectors from '{smashed_data_file}'...")
    smashed_vectors = pd.read_csv(smashed_data_file).values
    print(f"✅ Loaded {len(smashed_vectors)} vectors.")

    # 2. 공유 설정 파일 로드
    print(f"⚙️ Loading vector-to-image config from '{config_file}'...")
    with open(config_file, "r") as f:
        config = json.load(f)
    print("✅ Config loaded.")

    # 3. 벡터를 이미지 배열로 변환 (배치 처리)
    print("\n🖼️  Converting vectors to image arrays...")
    smashed_images = []
    for vector in smashed_vectors:
        img = def_vector_to_image(
            vector,
            side=config["image_side"],
            vmin=config["vmin"],
            vmax=config["vmax"]
        )
        smashed_images.append(img)

    smashed_images = np.array(smashed_images, dtype=np.float32)

    print(f"✅ Smashed data converted to image arrays.")
    print(f"   - Array batch shape: {smashed_images.shape}")
    print(f"   - Data type: {smashed_images.dtype}")

    # 4. 생성된 이미지 배열을 파일로 저장하여 서버로 전달
    output_file = "smashed_images.npy"
    np.save(output_file, smashed_images)
    print(f"\n🎉 Client-side process completed!")
    print(f"🔒 Smashed image arrays saved to '{output_file}' for the server.")

if __name__ == "__main__":
    main()
