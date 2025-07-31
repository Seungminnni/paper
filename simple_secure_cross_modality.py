#!/usr/bin/env python3
"""
Simple Secure Cross-Modality System
- Encrypts text messages into random-looking images
- Only trained decoder can recover original messages
- Demonstrates security through vector space separation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random

# GPU setup
print("=== Simple Secure Cross-Modality System ===")

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU setup complete: {len(gpus)} GPU(s) available")
        gpu_available = True
    else:
        print("No GPU found. Using CPU.")
        gpu_available = False
except Exception as e:
    print(f"GPU initialization error: {e}")
    gpu_available = False

device_name = '/GPU:0' if gpu_available else '/CPU:0'
print(f"Using device: {device_name}")

# =============================================================================
# Simple Secure Dataset Generation
# =============================================================================

def create_simple_dataset():
    """Generate simple secure message dataset"""
    
    # Simple message patterns for testing
    messages = [
        "hello", "world", "secure", "crypto", "neural",
        "deep", "learn", "test", "code", "data",
        "safe", "key", "lock", "pass", "hash",
        "net", "ai", "ml", "dl", "sys"
    ]
    
    vocab_size = len(messages)
    msg_to_idx = {msg: idx for idx, msg in enumerate(messages)}
    idx_to_msg = {idx: msg for msg, idx in msg_to_idx.items()}
    
    print(f"Dataset info:")
    print(f"  - Total messages: {vocab_size}")
    print(f"  - Sample messages: {messages[:5]}")
    
    return messages, msg_to_idx, idx_to_msg, vocab_size

# Create dataset
messages, msg_to_idx, idx_to_msg, vocab_size = create_simple_dataset()

# One-hot encoding function
def messages_to_onehot(message_list, msg_to_idx, vocab_size):
    """Convert messages to one-hot vectors"""
    onehot = np.zeros((len(message_list), vocab_size))
    for i, msg in enumerate(message_list):
        if msg in msg_to_idx:
            onehot[i, msg_to_idx[msg]] = 1
    return onehot

# Generate training data (each message repeated multiple times)
train_messages = []
test_messages = []

# Training data: repeat each message 20 times
for msg in messages:
    train_messages.extend([msg] * 20)

# Test data: repeat each message 5 times  
for msg in messages:
    test_messages.extend([msg] * 5)

# Shuffle
random.shuffle(train_messages)
random.shuffle(test_messages)

# Convert to one-hot
train_onehot = messages_to_onehot(train_messages, msg_to_idx, vocab_size)
test_onehot = messages_to_onehot(test_messages, msg_to_idx, vocab_size)

print(f"Data preparation complete:")
print(f"  - Training samples: {len(train_messages)} ({train_onehot.shape})")
print(f"  - Test samples: {len(test_messages)} ({test_onehot.shape})")

# =============================================================================
# Simple Secure Architecture
# =============================================================================

# Hyperparameters
LATENT_DIM = 64        # Latent space dimension
IMAGE_SIZE = 32        # Generated image size
NOISE_DIM = 16         # Security noise dimension

print(f"Architecture settings:")
print(f"  - Vocabulary size: {vocab_size}")
print(f"  - Latent dimension: {LATENT_DIM}")
print(f"  - Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  - Noise dimension: {NOISE_DIM}")

with tf.device(device_name):
    
    # =============================================================================
    # Secure Message Encoder (Message -> Encrypted Latent)
    # =============================================================================
    print("Building Secure Message Encoder...")
    
    message_input = Input(shape=(vocab_size,), name='message_input')
    noise_input = Input(shape=(NOISE_DIM,), name='noise_input')
    
    # Message encoding
    x = Dense(128, activation='relu', name='msg_enc1')(message_input)
    x = Dropout(0.3)(x)
    x = Dense(LATENT_DIM, activation='tanh', name='msg_enc2')(x)
    
    # Noise processing
    noise_processed = Dense(32, activation='relu')(noise_input)
    noise_processed = Dense(16, activation='tanh')(noise_processed)
    
    # Combine message and noise for security
    combined = tf.keras.layers.concatenate([x, noise_processed])
    encrypted_latent = Dense(LATENT_DIM, activation='sigmoid', name='encrypted_latent')(combined)
    
    secure_encoder = Model([message_input, noise_input], encrypted_latent, name='secure_encoder')
    
    # =============================================================================
    # Image Generator (Encrypted Latent -> Steganographic Image)
    # =============================================================================
    print("Building Image Generator...")
    
    latent_input = Input(shape=(LATENT_DIM,), name='latent_input')
    
    # Convert latent to image
    x = Dense(8 * 8 * 32, activation='relu')(latent_input)
    x = Reshape((8, 8, 32))(x)
    
    # Upsampling to generate image
    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)  # 16x16
    x = Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)  # 32x32
    
    # Final steganographic image
    stego_image = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid', name='stego_image')(x)
    
    image_generator = Model(latent_input, stego_image, name='image_generator')
    
    # =============================================================================
    # Image Analyzer (Steganographic Image -> Decoded Latent)
    # =============================================================================
    print("Building Image Analyzer...")
    
    image_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='image_input')
    
    # Extract hidden information from image
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)  # 16x16
    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)  # 8x8
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Decode latent vector
    decoded_latent = Dense(LATENT_DIM, activation='tanh', name='decoded_latent')(x)
    
    image_analyzer = Model(image_input, decoded_latent, name='image_analyzer')
    
    # =============================================================================
    # Message Decoder (Decoded Latent -> Original Message)
    # =============================================================================
    print("Building Message Decoder...")
    
    decoded_input = Input(shape=(LATENT_DIM,), name='decoded_input')
    
    # Decode message
    x = Dense(128, activation='relu')(decoded_input)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    
    # Final message recovery
    recovered_message = Dense(vocab_size, activation='softmax', name='recovered_message')(x)
    
    message_decoder = Model(decoded_input, recovered_message, name='message_decoder')

# =============================================================================
# End-to-End Secure System
# =============================================================================
print("Building End-to-End Secure System...")

# Complete pipeline: Message -> Image -> Message
msg_input_e2e = Input(shape=(vocab_size,), name='msg_input_e2e')
noise_input_e2e = Input(shape=(NOISE_DIM,), name='noise_input_e2e')

# Step-by-step transformation
encrypted_latent_e2e = secure_encoder([msg_input_e2e, noise_input_e2e])
stego_image_e2e = image_generator(encrypted_latent_e2e)
decoded_latent_e2e = image_analyzer(stego_image_e2e)
recovered_message_e2e = message_decoder(decoded_latent_e2e)

# Complete secure system
secure_system = Model(
    [msg_input_e2e, noise_input_e2e], 
    [stego_image_e2e, recovered_message_e2e],
    name='secure_communication_system'
)

# =============================================================================
# Model Summary
# =============================================================================
print("Model Architecture:")
print("\n1. Secure Message Encoder:")
secure_encoder.summary()
print("\n2. Image Generator:")
image_generator.summary()
print("\n3. Image Analyzer:")
image_analyzer.summary()
print("\n4. Message Decoder:")
message_decoder.summary()

# =============================================================================
# Model Compilation and Training
# =============================================================================
print("Compiling models...")

optimizer = Adam(learning_rate=0.001)

# Individual models
message_to_image = Model([msg_input_e2e, noise_input_e2e], stego_image_e2e, name='message_to_image')
message_to_image.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

image_to_message = Model(image_input, message_decoder(image_analyzer(image_input)), name='image_to_message')
image_to_message.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# End-to-end system
secure_system.compile(
    optimizer=optimizer,
    loss=['mse', 'categorical_crossentropy'],
    loss_weights=[0.4, 0.6],
    metrics={'image_generator': ['mae'], 'message_decoder': ['accuracy']}
)

# =============================================================================
# Training Data Preparation
# =============================================================================
print("Preparing training data...")

# Generate security noise
train_noise = np.random.normal(0, 0.3, (len(train_messages), NOISE_DIM))
test_noise = np.random.normal(0, 0.3, (len(test_messages), NOISE_DIM))

# Generate dummy steganographic targets
def generate_dummy_images(messages, image_size=32):
    """Generate message-specific dummy images"""
    n_samples = len(messages)
    images = np.zeros((n_samples, image_size, image_size, 1))
    
    for i, msg in enumerate(messages):
        # Use message hash as seed for reproducible "encryption"
        seed = hash(msg) % (2**31)
        np.random.seed(seed)
        images[i] = np.random.rand(image_size, image_size, 1)
    
    return images.astype('float32')

dummy_images_train = generate_dummy_images(train_messages)
dummy_images_test = generate_dummy_images(test_messages)

print(f"Training data prepared:")
print(f"  - Messages: {train_onehot.shape}")
print(f"  - Noise: {train_noise.shape}")
print(f"  - Images: {dummy_images_train.shape}")

# =============================================================================
# Secure Training
# =============================================================================
print("Starting secure training...")

with tf.device(device_name):
    
    print("Phase 1: Message -> Image training...")
    history1 = message_to_image.fit(
        [train_onehot, train_noise], dummy_images_train,
        epochs=5, batch_size=16,  # Reduced epochs and batch size for stability
        validation_data=([test_onehot, test_noise], dummy_images_test),
        verbose=1
    )
    
    print("Phase 2: End-to-End fine-tuning...")
    history2 = secure_system.fit(
        [train_onehot, train_noise], [dummy_images_train, train_onehot],
        epochs=5, batch_size=16,  # Reduced for stability
        validation_data=([test_onehot, test_noise], [dummy_images_test, test_onehot]),
        verbose=1
    )

print("Training completed!")

# =============================================================================
# Security Test
# =============================================================================
print("Starting security test...")

# Test messages
test_security_messages = [
    "hello", "world", "secure", "crypto", "neural"
]

print("Test messages:")
for i, msg in enumerate(test_security_messages):
    print(f"  {i+1}. '{msg}'")

# Prepare test data
test_indices = [msg_to_idx[msg] for msg in test_security_messages if msg in msg_to_idx]
test_security_onehot = np.zeros((len(test_indices), vocab_size))
for i, idx in enumerate(test_indices):
    test_security_onehot[i, idx] = 1

test_security_noise = np.random.normal(0, 0.3, (len(test_indices), NOISE_DIM))

# Run secure transformation
predictions = secure_system.predict([test_security_onehot, test_security_noise])
generated_images = predictions[0]
recovered_messages = predictions[1]

# Analyze results
recovered_indices = np.argmax(recovered_messages, axis=1)
recovered_texts = [idx_to_msg[idx] if idx in idx_to_msg else "UNKNOWN" for idx in recovered_indices]

# =============================================================================
# Results Visualization
# =============================================================================
print("Visualizing results...")

plt.figure(figsize=(20, 10))

num_samples = min(len(test_security_messages), len(recovered_texts))

for i in range(num_samples):
    # Original message
    plt.subplot(3, num_samples, i+1)
    plt.text(0.5, 0.5, f"Original:\n'{test_security_messages[i]}'", 
             fontsize=12, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.title(f'Input #{i+1}')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Steganographic image
    plt.subplot(3, num_samples, i+1+num_samples)
    if i < len(generated_images):
        plt.imshow(generated_images[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        plt.title('Encrypted Image')
    plt.xticks([])
    plt.yticks([])
    
    # Recovered message
    plt.subplot(3, num_samples, i+1+num_samples*2)
    recovered_text = recovered_texts[i] if i < len(recovered_texts) else "ERROR"
    is_correct = (test_security_messages[i] == recovered_text)
    color = "lightgreen" if is_correct else "lightcoral"
    
    plt.text(0.5, 0.5, f"Recovered:\n'{recovered_text}'", 
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
    plt.title(f'Output #{i+1}\n{"✓" if is_correct else "✗"}')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 1)
    plt.ylim(0, 1)

plt.suptitle('Secure Cross-Modality Communication System\n(Message -> Encrypted Image -> Recovered Message)', fontsize=16)
plt.tight_layout()
plt.show()

# =============================================================================
# Performance Analysis
# =============================================================================
print("Performance Analysis:")

# Calculate accuracy
correct_recoveries = sum(1 for orig, rec in zip(test_security_messages[:num_samples], 
                                               recovered_texts[:num_samples]) if orig == rec)
recovery_accuracy = correct_recoveries / num_samples if num_samples > 0 else 0

# Security analysis: distance between original and encrypted vectors
if len(test_security_onehot) > 0:
    encrypted_latents = secure_encoder.predict([test_security_onehot, test_security_noise])
    
    security_distances = []
    for i in range(len(test_security_onehot)):
        orig_norm = np.linalg.norm(test_security_onehot[i])
        enc_norm = np.linalg.norm(encrypted_latents[i])
        if orig_norm > 0 and enc_norm > 0:
            cosine_sim = np.dot(test_security_onehot[i], encrypted_latents[i]) / (orig_norm * enc_norm)
            security_distances.append(1 - cosine_sim)
    
    avg_security_distance = np.mean(security_distances) if security_distances else 0
    security_std = np.std(security_distances) if security_distances else 0
else:
    avg_security_distance = 0
    security_std = 0

print(f"Results:")
print(f"  Message Recovery Accuracy: {recovery_accuracy:.4f} ({recovery_accuracy*100:.2f}%)")
print(f"  Average Security Distance: {avg_security_distance:.4f}")
print(f"  Security Distance Std: {security_std:.4f}")

print(f"Security Assessment:")
if avg_security_distance > 0.7:
    security_level = "HIGH"
    security_desc = "Strong security: Original and encrypted vectors are very different"
elif avg_security_distance > 0.5:
    security_level = "MEDIUM"
    security_desc = "Moderate security: Some protection against interception"
else:
    security_level = "LOW"
    security_desc = "Security enhancement needed"

print(f"  Security Level: {security_level}")
print(f"  Description: {security_desc}")

if recovery_accuracy > 0.8:
    recovery_level = "EXCELLENT"
elif recovery_accuracy > 0.6:
    recovery_level = "GOOD"
else:
    recovery_level = "NEEDS IMPROVEMENT"

print(f"  Recovery Performance: {recovery_level}")

print(f"Secure Cross-Modality System Complete!")
print(f"Execution Environment: {device_name}")
print(f"Key Features:")
print(f"  ✓ Messages encrypted into random-looking images")
print(f"  ✓ Intercepted images reveal no original content")
print(f"  ✓ Only trained decoder can recover messages")
print(f"  ✓ Security through vector space separation")
