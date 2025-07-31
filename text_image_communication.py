#!/usr/bin/env python3
"""
Text-Image Communication System
- Client: Text → Image (Encoder)
- Server: Image → Text (Decoder)
- Vector spaces are completely different but models can interpret each other
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random

# GPU setup
print("=== Text-Image Communication System ===")

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU ready: {len(gpus)} GPU(s)")
        gpu_available = True
    else:
        print("Using CPU")
        gpu_available = False
except Exception as e:
    print(f"GPU error: {e}")
    gpu_available = False

device_name = '/GPU:0' if gpu_available else '/CPU:0'
print(f"Device: {device_name}")

# =============================================================================
# Simple Message Dataset
# =============================================================================

def create_messages():
    """Create simple message dataset"""
    
    messages = [
        "hello", "world", "cat", "dog", "good", 
        "bad", "yes", "no", "start", "stop",
        "go", "come", "help", "thank", "sorry"
    ]
    
    vocab_size = len(messages)
    msg_to_idx = {msg: idx for idx, msg in enumerate(messages)}
    idx_to_msg = {idx: msg for msg, idx in msg_to_idx.items()}
    
    print(f"Messages: {vocab_size} words")
    print(f"Examples: {messages[:5]}")
    
    return messages, msg_to_idx, idx_to_msg, vocab_size

messages, msg_to_idx, idx_to_msg, vocab_size = create_messages()

# One-hot encoding
def text_to_onehot(texts, msg_to_idx, vocab_size):
    onehot = np.zeros((len(texts), vocab_size))
    for i, text in enumerate(texts):
        if text in msg_to_idx:
            onehot[i, msg_to_idx[text]] = 1
    return onehot

# Generate training data
train_texts = []
test_texts = []

# Each message repeated many times
for msg in messages:
    train_texts.extend([msg] * 50)  # 50 times each
    test_texts.extend([msg] * 10)   # 10 times each

random.shuffle(train_texts)
random.shuffle(test_texts)

train_onehot = text_to_onehot(train_texts, msg_to_idx, vocab_size)
test_onehot = text_to_onehot(test_texts, msg_to_idx, vocab_size)

print(f"Training data: {len(train_texts)} samples")
print(f"Test data: {len(test_texts)} samples")

# =============================================================================
# Model Parameters
# =============================================================================

LATENT_DIM = 32        # Shared latent space
IMAGE_SIZE = 28        # Image size 28x28

print(f"Architecture:")
print(f"  Vocab size: {vocab_size}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

with tf.device(device_name):
    
    # =============================================================================
    # CLIENT SIDE: Text → Image Model
    # =============================================================================
    print("Building CLIENT model (Text → Image)...")
    
    # Text Encoder
    text_input = Input(shape=(vocab_size,), name='text_input')
    x = Dense(64, activation='relu')(text_input)
    x = Dense(LATENT_DIM, activation='relu')(x)
    text_encoded = Dense(LATENT_DIM, activation='sigmoid', name='text_latent')(x)
    
    text_encoder = Model(text_input, text_encoded, name='text_encoder')
    
    # Image Generator (from text latent)
    latent_input = Input(shape=(LATENT_DIM,), name='latent_input')
    x = Dense(7 * 7 * 16, activation='relu')(latent_input)
    x = Reshape((7, 7, 16))(x)
    x = Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)  # 14x14
    x = Conv2DTranspose(8, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)   # 28x28
    generated_image = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(x)
    
    image_generator = Model(latent_input, generated_image, name='image_generator')
    
    # Complete CLIENT model: Text → Image
    client_model = Model(text_input, image_generator(text_encoder(text_input)), name='client_text_to_image')
    
    # =============================================================================
    # SERVER SIDE: Image → Text Model  
    # =============================================================================
    print("Building SERVER model (Image → Text)...")
    
    # Image Encoder
    image_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='image_input')
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(image_input)
    x = Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)  # 14x14
    x = Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)  # 7x7
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    image_encoded = Dense(LATENT_DIM, activation='sigmoid', name='image_latent')(x)
    
    image_encoder = Model(image_input, image_encoded, name='image_encoder')
    
    # Text Decoder (from image latent)
    latent_to_text_input = Input(shape=(LATENT_DIM,), name='latent_to_text_input')
    x = Dense(64, activation='relu')(latent_to_text_input)
    x = Dense(vocab_size, activation='softmax')(x)
    
    text_decoder = Model(latent_to_text_input, x, name='text_decoder')
    
    # Complete SERVER model: Image → Text
    server_model = Model(image_input, text_decoder(image_encoder(image_input)), name='server_image_to_text')

print("Model Summary:")
print("\nCLIENT - Text Encoder:")
text_encoder.summary()
print("\nCLIENT - Image Generator:")
image_generator.summary()
print("\nSERVER - Image Encoder:")
image_encoder.summary()
print("\nSERVER - Text Decoder:")
text_decoder.summary()

# =============================================================================
# Model Compilation
# =============================================================================
print("Compiling models...")

client_model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
server_model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# =============================================================================
# Generate Target Images (Different vector space)
# =============================================================================
print("Generating target images...")

def generate_target_images(texts, image_size=28):
    """Generate unique images for each text (completely different vector space)"""
    n_samples = len(texts)
    images = np.zeros((n_samples, image_size, image_size, 1))
    
    for i, text in enumerate(texts):
        # Use text hash as seed for consistent but different image
        seed = hash(text) % (2**31)
        np.random.seed(seed)
        
        # Generate unique pattern for this text
        base_image = np.random.rand(image_size, image_size, 1)
        
        # Add some structure to make it more "image-like"
        for x in range(0, image_size, 4):
            for y in range(0, image_size, 4):
                if np.random.random() > 0.5:
                    base_image[x:x+4, y:y+4] = np.random.random()
        
        images[i] = base_image
    
    return images.astype('float32')

# Generate target images for training
train_target_images = generate_target_images(train_texts)
test_target_images = generate_target_images(test_texts)

print(f"Target images generated: {train_target_images.shape}")

# =============================================================================
# Training Phase
# =============================================================================
print("Starting training...")

with tf.device(device_name):
    
    print("Phase 1: Training CLIENT (Text → Image)...")
    client_history = client_model.fit(
        train_onehot, train_target_images,
        epochs=10, batch_size=32,
        validation_data=(test_onehot, test_target_images),
        verbose=1
    )
    
    print("Phase 2: Training SERVER (Image → Text)...")
    server_history = server_model.fit(
        train_target_images, train_onehot,
        epochs=10, batch_size=32, 
        validation_data=(test_target_images, test_onehot),
        verbose=1
    )

print("Training completed!")

# =============================================================================
# Test Communication System
# =============================================================================
print("Testing communication system...")

# Test messages - ALL "hello" to test if it's just memorization
test_messages = ["hello", "hello", "hello", "hello", "hello"]
print(f"Test messages (all same): {test_messages}")

# Also test with more samples - all "hello"
extended_test_messages = ["hello"] * 20  # 20 times "hello"
print(f"Extended test: {len(extended_test_messages)} samples of 'hello'")

# Prepare test data - all "hello"
test_indices = [msg_to_idx[msg] for msg in test_messages if msg in msg_to_idx]
test_input_onehot = np.zeros((len(test_indices), vocab_size))
for i, idx in enumerate(test_indices):
    test_input_onehot[i, idx] = 1

# Prepare extended test data
extended_test_indices = [msg_to_idx["hello"]] * len(extended_test_messages)
extended_test_onehot = np.zeros((len(extended_test_messages), vocab_size))
for i, idx in enumerate(extended_test_indices):
    extended_test_onehot[i, idx] = 1

# Step 1: CLIENT converts text to images (all "hello")
print("CLIENT: Converting all 'hello' to images...")
client_generated_images = client_model.predict(test_input_onehot)

# Extended test with 20 "hello" samples
print("CLIENT: Converting 20 'hello' samples to images...")
extended_client_images = client_model.predict(extended_test_onehot)

# Step 2: SERVER interprets images back to text
print("SERVER: Interpreting images to text...")
server_predictions = server_model.predict(client_generated_images)
server_predicted_indices = np.argmax(server_predictions, axis=1)
server_predicted_texts = [idx_to_msg[idx] for idx in server_predicted_indices]

# Extended server predictions
print("SERVER: Interpreting 20 image samples...")
extended_server_predictions = server_model.predict(extended_client_images)
extended_predicted_indices = np.argmax(extended_server_predictions, axis=1)
extended_predicted_texts = [idx_to_msg[idx] for idx in extended_predicted_indices]

# Check if all generated images are identical (they should be if it's just memorization)
print("\nAnalyzing generated images...")
image_similarity_check = []
base_image = client_generated_images[0].flatten()
for i in range(1, len(client_generated_images)):
    similarity = np.corrcoef(base_image, client_generated_images[i].flatten())[0, 1]
    image_similarity_check.append(similarity)

avg_image_similarity = np.mean(image_similarity_check) if image_similarity_check else 1.0
print(f"Average image similarity between 'hello' samples: {avg_image_similarity:.6f}")
print(f"(1.0 = identical, <0.99 = different images)")

# Extended image similarity check
extended_similarity_check = []
extended_base = extended_client_images[0].flatten()
for i in range(1, len(extended_client_images)):
    similarity = np.corrcoef(extended_base, extended_client_images[i].flatten())[0, 1]
    extended_similarity_check.append(similarity)

extended_avg_similarity = np.mean(extended_similarity_check) if extended_similarity_check else 1.0
print(f"Extended test - image similarity: {extended_avg_similarity:.6f}")

# =============================================================================
# Results Visualization
# =============================================================================
print("Visualizing results...")

plt.figure(figsize=(15, 10))

num_samples = len(test_messages)

for i in range(num_samples):
    # Original text
    plt.subplot(3, num_samples, i+1)
    plt.text(0.5, 0.5, f"Original:\n'{test_messages[i]}'", 
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round", facecolor="lightblue"))
    plt.title(f'CLIENT Input')
    plt.axis('off')
    
    # Generated image (CLIENT output)
    plt.subplot(3, num_samples, i+1+num_samples)
    plt.imshow(client_generated_images[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('CLIENT Output\n(Generated Image)')
    plt.axis('off')
    
    # Predicted text (SERVER output)
    plt.subplot(3, num_samples, i+1+num_samples*2)
    predicted_text = server_predicted_texts[i]
    is_correct = (test_messages[i] == predicted_text)
    color = "lightgreen" if is_correct else "lightcoral"
    
    plt.text(0.5, 0.5, f"Predicted:\n'{predicted_text}'", 
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round", facecolor=color))
    plt.title(f'SERVER Output\n{"✓" if is_correct else "✗"}')
    plt.axis('off')

plt.suptitle('Text-Image Communication System\n(CLIENT: Text→Image, SERVER: Image→Text)', fontsize=16)
plt.tight_layout()
plt.show()

# =============================================================================
# Performance Analysis
# =============================================================================
print("Performance Analysis:")

# Calculate accuracy
correct_predictions = sum(1 for orig, pred in zip(test_messages, server_predicted_texts) if orig == pred)
accuracy = correct_predictions / len(test_messages)

# Extended accuracy for 20 samples
extended_correct = sum(1 for orig, pred in zip(extended_test_messages, extended_predicted_texts) if orig == pred)
total_samples = len(test_messages) + len(extended_test_messages)
total_correct = correct_predictions + extended_correct
overall_accuracy = total_correct / total_samples

print(f"Standard Test Accuracy: {accuracy:.2%} ({correct_predictions}/{len(test_messages)})")
print(f"Extended Test Accuracy: {extended_correct/len(extended_test_messages):.2%} ({extended_correct}/{len(extended_test_messages)})")
print(f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_samples})")

# Analysis of results
print("\n" + "="*50)
print("MEMORIZATION vs LEARNING ANALYSIS")
print("="*50)

if avg_image_similarity > 0.999:
    print("🔍 FINDING: All 'hello' inputs generate IDENTICAL images")
    print("   → This confirms the system is using deterministic hash-based generation")
    print("   → Same input always produces same 'random' image")
    
    if overall_accuracy == 1.0:
        print("🎯 CONCLUSION: 100% accuracy is due to MEMORIZATION, not learning")
        print("   → System acts like a lookup table: text → fixed image → text")
        print("   → This is NOT true cross-modality learning")
    else:
        print("🤔 INTERESTING: Despite identical images, accuracy < 100%")
        print("   → This suggests some randomness in the decoding process")
else:
    print("🔍 FINDING: 'hello' inputs generate DIFFERENT images")
    print("   → System has some randomness in image generation")
    
    if overall_accuracy == 1.0:
        print("🎯 CONCLUSION: High accuracy with different images suggests true learning")
        print("   → System may have learned meaningful cross-modality mapping")
    else:
        print("🎯 CONCLUSION: Variable accuracy suggests learning with noise")
        print("   → System shows realistic learning behavior")

print(f"\nImage similarity metrics:")
print(f"  • Standard test similarity: {avg_image_similarity:.6f}")
print(f"  • Extended test similarity: {extended_avg_similarity:.6f}")
print(f"  • (1.0 = identical, <0.99 = different images)")

print("\nNext step recommendation:")
if avg_image_similarity > 0.999 and overall_accuracy == 1.0:
    print("→ Implement truly random image generation for genuine cross-modality learning")
else:
    print("→ Current system shows promise for real cross-modality communication")

# Analyze vector differences
print("Vector Space Analysis:")

# Get text vectors (original)
text_vectors = test_input_onehot

# Get image vectors (after CLIENT conversion)
client_latents = text_encoder.predict(test_input_onehot)
image_vectors = client_generated_images.reshape(len(client_generated_images), -1)

# Calculate vector distances
vector_distances = []
for i in range(len(text_vectors)):
    text_norm = np.linalg.norm(text_vectors[i])
    image_norm = np.linalg.norm(image_vectors[i])
    
    if text_norm > 0 and image_norm > 0:
        # Cosine distance
        cosine_sim = np.dot(text_vectors[i], image_vectors[i][:len(text_vectors[i])]) / (text_norm * image_norm)
        vector_distances.append(1 - cosine_sim)

avg_distance = np.mean(vector_distances) if vector_distances else 0

print(f"Average Vector Distance: {avg_distance:.4f}")
print(f"Vector Space Separation: {'HIGH' if avg_distance > 0.8 else 'MEDIUM' if avg_distance > 0.5 else 'LOW'}")

print("Detailed Results:")
for i, (orig, pred) in enumerate(zip(test_messages, server_predicted_texts)):
    status = "✓" if orig == pred else "✗"
    print(f"  {i+1}. '{orig}' → '{pred}' {status}")

print(f"\nCommunication System Complete!")
print(f"Key Points:")
print(f"  • Text and Image vectors are in completely different spaces")
print(f"  • CLIENT converts text to images")
print(f"  • SERVER interprets images back to text") 
print(f"  • Models learn to communicate through shared latent space")
print(f"  • Vector distance: {avg_distance:.4f} (different spaces)")
print(f"  • Communication accuracy: {accuracy:.2%}")
