#!/usr/bin/env python3
"""
개선 버전: 컨볼루션 Autoencoder 압축 + Residual Attention CNN 학습
CSV 텍스트 → Word2Vec → 모자이크 이미지 → 압축 → CNN → 벡터
- 효율성 및 정확도 평가 시스템 추가
- 10,000개 데이터 중 7,000개 학습, 3,000개 검증
- 최적화된 에포크 및 배치 사이즈
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
try:
    from tensorflow.keras.optimizers import AdamW
except Exception:
    from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
from datetime import datetime
import math

# -----------------------------------
# Word2Vec processor (간단 버전)
# -----------------------------------
from simple_word2vec_mosaic import SimpleWord2VecProcessor

# -----------------------------------
# Residual Block
# -----------------------------------
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    return x

# -----------------------------------
# Attention (Squeeze-and-Excitation)
# -----------------------------------
def se_block(x, reduction=16):
    filters = x.shape[-1]
    if filters is None:
        filters = tf.shape(x)[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // reduction, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([x, se])

# -----------------------------------
# Autoencoder for Compression
# -----------------------------------
def build_autoencoder(input_shape):
    """실제 압축이 가능한 Autoencoder 구축 - 정규화 강화 및 안정성 개선"""
    inputs = Input(shape=input_shape)
    print(f"   🔧 Building autoencoder for input shape: {input_shape}")

    # Encoder - 점진적 압축 (드롭아웃 및 정규화 강화)
    x = Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(inputs)  # /2
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)  # 드롭아웃 추가
    x = residual_block(x, 64)
    x = se_block(x)
    print(f"   📐 After Conv1: {x.shape}")

    x = Conv2D(128, (3, 3), strides=2, padding="same", activation="relu")(x)  # /4
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # 드롭아웃 증가
    x = residual_block(x, 128)
    x = se_block(x)
    print(f"   📐 After Conv2: {x.shape}")

    x = Conv2D(256, (3, 3), strides=2, padding="same", activation="relu")(x)  # /8
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)  # 더 강한 정규화
    x = residual_block(x, 256)
    x = se_block(x)
    print(f"   📐 After Conv3: {x.shape}")

    # 강력한 압축을 위한 추가 레이어 (정규화 강화)
    x = Conv2D(512, (3, 3), strides=1, padding="same", activation="relu")(x)  
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # 최고 수준 정규화
    
    # 압축된 표현 (더 작은 차원으로 강화)
    encoded = Conv2D(128, (3, 3), strides=1, padding="same", activation="relu")(x)  # 256→128로 축소
    print(f"   🗜️ Bottleneck (compressed): {encoded.shape}")
    
    # Decoder - 정확한 복원을 위한 크기 맞춤 (정규화 포함)
    x = Conv2D(256, (3, 3), strides=1, padding="same", activation="relu")(encoded)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Conv2D(512, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2DTranspose(256, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = residual_block(x, 256)
    
    x = Conv2DTranspose(128, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = residual_block(x, 128)
    
    x = Conv2DTranspose(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # 최종 출력 - 원본과 정확히 같은 크기로 맞춤
    decoded = Conv2D(1, (3, 3), padding="same", activation="sigmoid")(x)
    
    # 크기 불일치 시 크롭핑으로 조정
    if decoded.shape[1] != input_shape[0] or decoded.shape[2] != input_shape[1]:
        decoded = Lambda(lambda x: x[:, :input_shape[0], :input_shape[1], :])(decoded)
    
    print(f"   🔄 Reconstructed: {decoded.shape}")

    # Full autoencoder (L2 정규화 추가)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(
        optimizer=AdamW(learning_rate=5e-4, weight_decay=1e-4),
        loss='mse'
    )
    
    # Encoder only (압축용)
    encoder = Model(inputs, encoded)
    
    return autoencoder, encoder

# -----------------------------------
# Residual Attention CNN for Prediction
# -----------------------------------
def build_residual_attention_cnn(input_shape, output_size):
    """단순화된 CNN 모델 - 학습 안정성 향상"""
    inputs = Input(shape=input_shape)

    # 더 간단한 구조로 변경
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), strides=2, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = GlobalAveragePooling2D()(x)
    
    # 더 작은 Dense 레이어
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)  # 타깃 크기와 맞춤
    x = Dropout(0.3)(x)
    outputs = Dense(output_size, activation="linear")(x)

    model = Model(inputs, outputs)
    # 더 높은 학습률로 조정
    model.compile(
        optimizer=AdamW(learning_rate=5e-4, weight_decay=1e-5),
        loss="mse",
        metrics=["mae"]
    )
    return model

# -----------------------------------
# Load & Preprocess Data with Evaluation
# -----------------------------------
def load_text_data(max_samples=10000):
    """환자 데이터를 로드하고 더 풍부한 텍스트 정보 생성"""
    print(f"📊 Loading patient data (max {max_samples} samples)...")
    df = pd.read_csv("patients.csv", nrows=max_samples).dropna(subset=["FIRST", "LAST"])
    
    # 더 풍부한 텍스트 정보 생성
    texts = []
    for _, row in df.iterrows():
        parts = []
        # 이름 정보
        if pd.notna(row.get('FIRST')): parts.append(str(row['FIRST']))
        if pd.notna(row.get('LAST')): parts.append(str(row['LAST']))
        # 지역 정보
        if pd.notna(row.get('CITY')): parts.append(str(row['CITY']))
        if pd.notna(row.get('STATE')): parts.append(str(row['STATE']))
        # 인구통계 정보
        if pd.notna(row.get('RACE')): parts.append(str(row['RACE']))
        if pd.notna(row.get('ETHNICITY')): parts.append(str(row['ETHNICITY']))
        if pd.notna(row.get('GENDER')): parts.append(str(row['GENDER']))
        
        text = ' '.join(parts).lower()
        texts.append(text)
    
    print(f"✅ Loaded {len(texts)} patient records")
    print(f"   Sample text: '{texts[0]}'")
    return texts

def generate_mosaic(texts, processor, vector_size=256):
    """텍스트를 모자이크 이미지(샘플별 작은 이미지)로 변환
    - 반환 X: (N, H, W, C)
    - 반환 vectors: (N, vector_size)
    - processor: 사전에 학습된 SimpleWord2VecProcessor
    """
    print(f"🎨 Generating per-sample mosaic from {len(texts)} texts...")
    if hasattr(processor, "vectorize"):
        vectors = processor.vectorize(texts)
    else:
        print("⚠️ processor.vectorize() not found; falling back to train_and_vectorize(). This may be slower.")
        vectors = processor.train_and_vectorize(texts)

    # 정규화 (샘플별 min-max가 아닌 전체 min-max로 고정)
    vmin, vmax = vectors.min(), vectors.max()
    vectors = (vectors - vmin) / (vmax - vmin + 1e-8)

    # 벡터를 정사각 이미지로 reshape (예: 256D -> 16x16x1)
    side = int(math.sqrt(vector_size))
    assert side * side == vector_size, "vector_size는 완전제곱수여야 합니다. 예: 256=16x16"
    N = vectors.shape[0]
    X = vectors.reshape(N, side, side, 1)
    print(f"   Mosaic batch shape: {X.shape} (N,H,W,C)")

    return X, vectors

# -----------------------------------
# Evaluation Metrics
# -----------------------------------
def calculate_metrics(y_true, y_pred, name=""):
    """포괄적인 평가 메트릭 계산 - 안정성 개선"""
    # 입력 데이터 검증 및 정리
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # NaN 또는 inf 값 제거
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid_mask):
        print(f"   ⚠️ Warning: No valid values found in {name}")
        return {
            'mse': float('inf'),
            'mae': float('inf'),
            'r2': -float('inf'),
            'correlation': 0.0,
            'smape': 100.0
        }
    
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    # 기본 메트릭 계산
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    # 안전한 R² 계산
    try:
        # 분산이 0에 가까운 경우 처리
        y_var = np.var(y_true_clean)
        if y_var < 1e-10:
            r2 = 0.0  # 상수 타깃의 경우
        else:
            r2 = r2_score(y_true_clean, y_pred_clean)
            # R² 값이 비정상적인 경우 클리핑
            if np.isnan(r2) or np.isinf(r2):
                r2 = -1.0  # 최소값으로 설정
            elif r2 < -10:
                r2 = -10.0  # 하한 설정
    except:
        r2 = -1.0
    
    # 안전한 상관관계 계산
    try:
        if len(np.unique(y_true_clean)) < 2 or len(np.unique(y_pred_clean)) < 2:
            correlation = 0.0  # 상수 배열의 경우
        else:
            correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
            if np.isnan(correlation) or np.isinf(correlation):
                correlation = 0.0
    except:
        correlation = 0.0
    
    # 안전한 SMAPE 계산
    try:
        denominator = np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-8
        smape = 2.0 * np.mean(np.abs(y_true_clean - y_pred_clean) / denominator) * 100
        if np.isnan(smape) or np.isinf(smape):
            smape = 100.0
    except:
        smape = 100.0
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'correlation': float(correlation),
        'smape': float(smape)
    }
    
    if name:
        print(f"📊 {name} Metrics:")
        print(f"   • MSE: {mse:.6f}")
        print(f"   • MAE: {mae:.4f}")
        print(f"   • R²: {r2:.4f}")
        print(f"   • Correlation: {correlation:.4f}")
        print(f"   • SMAPE: {smape:.2f}%")
        print(f"   • Valid samples: {len(y_true_clean)}/{len(y_true)}")
    
    return metrics

def plot_results(history_ae, history_cnn, train_metrics, val_metrics):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Autoencoder 학습 곡선
    axes[0, 0].plot(history_ae.history['loss'], label='Train Loss')
    axes[0, 0].plot(history_ae.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Autoencoder Training')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # CNN 학습 곡선
    axes[0, 1].plot(history_cnn.history['loss'], label='Train Loss')
    axes[0, 1].plot(history_cnn.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('CNN Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # MAE 비교
    mae_comparison = [train_metrics['mae'], val_metrics['mae']]
    axes[0, 2].bar(['Training', 'Validation'], mae_comparison, color=['blue', 'orange'])
    axes[0, 2].set_title('MAE Comparison')
    axes[0, 2].set_ylabel('MAE')
    
    # 상관관계 비교
    corr_comparison = [train_metrics['correlation'], val_metrics['correlation']]
    axes[1, 0].bar(['Training', 'Validation'], corr_comparison, color=['green', 'red'])
    axes[1, 0].set_title('Correlation Comparison')
    axes[1, 0].set_ylabel('Correlation')
    
    # R² 비교
    r2_comparison = [train_metrics['r2'], val_metrics['r2']]
    axes[1, 1].bar(['Training', 'Validation'], r2_comparison, color=['purple', 'brown'])
    axes[1, 1].set_title('R² Score Comparison')
    axes[1, 1].set_ylabel('R² Score')
    
    # 성능 요약
    summary_text = f"""Performance Summary:

Training:
• MAE: {train_metrics['mae']:.4f}
• Correlation: {train_metrics['correlation']:.4f}
• R²: {train_metrics['r2']:.4f}

Validation:
• MAE: {val_metrics['mae']:.4f}
• Correlation: {val_metrics['correlation']:.4f}
• R²: {val_metrics['r2']:.4f}

Generalization:
• MAE Ratio: {val_metrics['mae']/train_metrics['mae']:.2f}
• Correlation Drop: {train_metrics['correlation'] - val_metrics['correlation']:.4f}
"""
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title('Performance Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Results visualization saved as 'training_results.png'")

# -----------------------------------
# Main Test Function with Comprehensive Evaluation
# -----------------------------------
# -----------------------------------
# Main Test Function with Comprehensive Evaluation
# -----------------------------------
def fit_word2vec_processor(all_texts, vector_size=256):
    print(f"🧩 Fitting Word2Vec processor on {len(all_texts)} texts (once, shared for train/val)...")
    processor = SimpleWord2VecProcessor(vector_size=vector_size)
    # train_and_vectorize는 벡터를 바로 반환하므로 여기서는 내부 모델만 학습시키기 위해 한 번 호출 후 저장
    _ = processor.train_and_vectorize(all_texts)
    return processor

def test_pipeline():
    """메인 테스트 파이프라인 - 개선된 평가 시스템"""
    print("🚀 Starting Comprehensive Training and Evaluation Pipeline")
    print("="*70)
    
    start_time = time.time()
    
    # 1. 데이터 로딩
    texts = load_text_data(max_samples=10000)

    print(f"\n🧩 Fit shared text→vector processor")
    processor = fit_word2vec_processor(texts, vector_size=256)

    print(f"\n📊 Data Split: 7,000 training / 3,000 validation")
    train_texts, val_texts = train_test_split(
        texts, train_size=7000, test_size=3000, random_state=42, shuffle=True
    )
    print(f"   ✅ Actual split: {len(train_texts)} train, {len(val_texts)} validation")

    print(f"\n🎨 Phase 1: Mosaic Generation (per-sample)")
    train_mosaic, train_vec = generate_mosaic(train_texts, processor=processor, vector_size=256)
    val_mosaic, val_vec = generate_mosaic(val_texts, processor=processor, vector_size=256)

    input_shape = train_mosaic.shape[1:]  # (H,W,C)
    print(f"   Input shape: {input_shape}")
    
    # 3. Autoencoder 압축 학습 (실제 압축이 가능한 버전)
    print(f"\n🤖 Phase 2: Autoencoder Training with Real Compression")
    autoencoder, encoder = build_autoencoder(input_shape)
    
    # 콜백 설정 (조기 종료 강화)
    ae_callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, min_delta=1e-5),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]
    
    # Autoencoder 학습 (실제 압축 최적화)
    print("   Training autoencoder for real compression...")
    history_ae = autoencoder.fit(
        train_mosaic, train_mosaic,
        epochs=40,           # 에포크 감소 (50→40)
        batch_size=32,       # 배치 크기 증가 (16→32)
        validation_data=(val_mosaic, val_mosaic),
        callbacks=ae_callbacks,
        verbose=1
    )
    
    # 4. 실제 압축본 생성 (encoder만 사용)
    print(f"\n🗜️ Phase 3: Generating Real Compressed Representations")
    compressed_train = encoder.predict(train_mosaic, verbose=0)
    compressed_val = encoder.predict(val_mosaic, verbose=0)
    
    print(f"   Compressed train shape: {compressed_train.shape}")
    print(f"   Compressed val shape: {compressed_val.shape}")
    
    # 압축 효율성 계산
    original_size = np.prod(train_mosaic.shape[1:])
    compressed_size = np.prod(compressed_train.shape[1:])
    compression_ratio = original_size / compressed_size
    print(f"   Per-sample sizes → original: {original_size}, compressed: {compressed_size}")
    print(f"   Compression ratio: {compression_ratio:.2f}:1")
    
    # 5. CNN 예측 학습 (데이터 증강 포함)
    print(f"\n🧠 Phase 4: CNN Prediction Training with Data Augmentation")
    vector_size = train_vec.shape[1]
    cnn = build_residual_attention_cnn(compressed_train.shape[1:], output_size=vector_size)

    # 샘플별 타깃: 원래 텍스트 임베딩 그대로 회귀
    y_train = train_vec  # (N, vector_size)
    y_val = val_vec      # (N, vector_size)
    print(f"   Target vector dimension: {vector_size} (per-sample regression to original embedding)")

    # 데이터 증강: 더 부드러운 증강 적용
    print("   Applying gentle data augmentation...")
    augmented_train = [compressed_train]
    augmented_targets = [y_train]
    for i in range(2):
        noise_level = 0.001 * (i + 1)
        augmented_batch = compressed_train + np.random.normal(0, noise_level, compressed_train.shape)
        augmented_train.append(augmented_batch)
        augmented_targets.append(y_train)
    X_train_aug = np.vstack(augmented_train)
    y_train_aug = np.vstack(augmented_targets)
    print(f"   Augmented training data shape: {X_train_aug.shape}, targets: {y_train_aug.shape}")
    
    # CNN 콜백 설정 (더 관대한 조기 종료)
    cnn_callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=1e-6),
        ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=8, min_lr=1e-6, verbose=1)
    ]
    
    # CNN 학습 (더 긴 학습 허용)
    print("   Training simplified CNN...")
    history_cnn = cnn.fit(
        X_train_aug, y_train_aug,
        epochs=120,         # 에포크 증가 
        batch_size=8,       # 원래 배치 크기로 복원
        validation_data=(compressed_val, y_val),
        callbacks=cnn_callbacks,
        verbose=1
    )
    
    # 6. 최종 평가 및 Robustness 테스트
    print(f"\n📊 Phase 5: Comprehensive Evaluation with Robustness Analysis")

    # 예측 수행
    train_pred = cnn.predict(compressed_train, verbose=0)
    val_pred = cnn.predict(compressed_val, verbose=0)

    # 메트릭 계산
    train_metrics = calculate_metrics(y_train, train_pred, "Training")
    val_metrics = calculate_metrics(y_val, val_pred, "Validation")
    
    # Robustness 테스트 (노이즈 추가된 데이터로 예측 일관성 확인)
    print(f"\n🛡️ Robustness Analysis:")
    robustness_scores = []
    for noise_level in [0.001, 0.005, 0.01]:
        noisy_val = compressed_val + np.random.normal(0, noise_level, compressed_val.shape)
        noisy_pred = cnn.predict(noisy_val, verbose=0)
        consistency = np.corrcoef(val_pred.flatten(), noisy_pred.flatten())[0, 1]
        robustness_scores.append(consistency)
        print(f"   • Noise level {noise_level:.3f}: {consistency:.3f} consistency")
    
    avg_robustness = np.mean(robustness_scores)
    print(f"   • Average robustness: {avg_robustness:.3f}")
    
    if avg_robustness > 0.9:
        robustness_grade = "🛡️ Highly robust"
    elif avg_robustness > 0.8:
        robustness_grade = "👍 Moderately robust"
    else:
        robustness_grade = "⚠️ Low robustness"
    
    print(f"   • Robustness grade: {robustness_grade}")
    
    # 7. 일반화 성능 분석 (개선된 기준)
    print(f"\n🎯 Generalization Analysis:")
    mae_ratio = val_metrics['mae'] / train_metrics['mae']
    corr_drop = train_metrics['correlation'] - val_metrics['correlation']
    
    print(f"   • MAE Ratio (val/train): {mae_ratio:.3f}")
    print(f"   • Correlation Drop: {corr_drop:.4f}")
    print(f"   • Compression Efficiency: {compression_ratio:.1f}:1")
    
    # 성능 등급 결정 (개선된 기준)
    if val_metrics['correlation'] > 0.85 and val_metrics['r2'] > 0.7 and mae_ratio < 1.2:
        grade = "🏆 Excellent"
    elif val_metrics['correlation'] > 0.75 and val_metrics['r2'] > 0.5 and mae_ratio < 1.5:
        grade = "✅ Good"  
    elif val_metrics['correlation'] > 0.6 and val_metrics['r2'] > 0.3:
        grade = "👍 Moderate"
    else:
        grade = "⚠️ Needs Improvement"
    
    print(f"   • Overall Performance: {grade}")
    
    # 일반화 등급
    if mae_ratio < 1.0 and corr_drop < 0.05:
        gen_grade = "🎯 Excellent Generalization"
    elif mae_ratio < 1.2 and corr_drop < 0.1:
        gen_grade = "👍 Good Generalization"
    elif mae_ratio < 1.5:
        gen_grade = "👍 Moderate Generalization" 
    else:
        gen_grade = "⚠️ Poor Generalization"
    
    print(f"   • Generalization: {gen_grade}")
    
    # 8. 시각화 및 결과 저장
    print(f"\n📈 Generating Results Visualization...")
    plot_results(history_ae, history_cnn, train_metrics, val_metrics)
    
    # 실행 시간
    total_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {total_time:.2f} seconds")
    
    # 9. 최종 요약 리포트 (개선된 버전)
    print(f"\n{'='*70}")
    print("🎓 ENHANCED PERFORMANCE REPORT")
    print(f"{'='*70}")
    print(f"📊 Dataset: {len(texts)} samples ({len(train_texts)} train / {len(val_texts)} val)")
    print(f"🗜️ Compression: {compression_ratio:.1f}:1 ratio")
    print(f"⚙️ Optimization: Batch size 8 + Data Augmentation + Low LR (0.0005)")
    print(f"📈 Training Performance:")
    print(f"   • Correlation: {train_metrics['correlation']:.4f} ({train_metrics['correlation']*100:.1f}%)")
    print(f"   • R² Score: {train_metrics['r2']:.4f} ({train_metrics['r2']*100:.1f}%)")
    print(f"   • MAE: {train_metrics['mae']:.4f}")
    print(f"🎯 Validation Performance:")
    print(f"   • Correlation: {val_metrics['correlation']:.4f} ({val_metrics['correlation']*100:.1f}%)")
    print(f"   • R² Score: {val_metrics['r2']:.4f} ({val_metrics['r2']*100:.1f}%)")
    print(f"   • MAE: {val_metrics['mae']:.4f}")
    print(f"📊 Generalization Metrics:")
    print(f"   • MAE Ratio: {mae_ratio:.3f}")
    print(f"   • Correlation Drop: {corr_drop:.4f}")
    print(f"   • Evaluation: {gen_grade}")
    print(f"🛡️ Robustness: {avg_robustness:.3f} ({robustness_grade})")
    print(f"🏆 Overall Grade: {grade}")
    print(f"⏱️ Training Time: {total_time:.1f}s")
    print(f"{'='*70}")
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'compression_ratio': compression_ratio,
        'training_time': total_time,
        'grade': grade,
        'generalization_grade': gen_grade,
        'robustness_score': avg_robustness,
        'robustness_grade': robustness_grade,
        'mae_ratio': mae_ratio
    }

if __name__ == "__main__":
    test_pipeline()
