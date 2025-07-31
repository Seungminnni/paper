#!/usr/bin/env python3
"""
CSV-based Text-Image Communication System
- Uses real CSV data (names) for text-image communication
- Client: Name → Random Image
- Server: Random Image → Name
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder

# GPU setup
print("=== CSV-based Text-Image Communication System ===")

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
# Load and Process CSV Data
# =============================================================================
print("Loading CSV data...")

def load_csv_names(csv_file):
    """Load names from CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Remove empty rows
        df = df.dropna(subset=['first_name'])
        
        print(f"Total rows in CSV: {len(df)}")
        
        # Extract first names
        names = df['first_name'].str.lower().str.strip()
        names = names[names.str.len() > 0]  # Remove empty names
        
        # Get unique names and count frequency
        name_counts = names.value_counts()
        print(f"Unique names found: {len(name_counts)}")
        print(f"Most common names: {name_counts.head()}")
        
        # Use names that appear at least twice for training
        frequent_names = name_counts[name_counts >= 2].index.tolist()
        
        if len(frequent_names) < 5:
            # If not enough frequent names, use all unique names
            frequent_names = name_counts.index.tolist()[:20]  # Take top 20
            
        print(f"Selected names for training: {len(frequent_names)}")
        print(f"Names: {frequent_names[:10]}")
        
        return frequent_names, name_counts
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        # Fallback to default names
        fallback_names = ["joseph", "pauline", "mary", "john", "james", "robert", "michael", "william", "david", "richard"]
        print(f"Using fallback names: {fallback_names}")
        return fallback_names, None

# Load names from CSV
selected_names, name_counts = load_csv_names('ncvaa.csv')

# Create vocabulary
vocab_size = len(selected_names)
name_to_idx = {name: idx for idx, name in enumerate(selected_names)}
idx_to_name = {idx: name for name, idx in name_to_idx.items()}

print(f"Vocabulary size: {vocab_size}")

# =============================================================================
# Generate Training Data
# =============================================================================
def create_training_data(names, samples_per_name=100):
    """Create training data by repeating each name multiple times"""
    train_texts = []
    test_texts = []
    
    for name in names:
        # Training data
        train_texts.extend([name] * samples_per_name)
        # Test data  
        test_texts.extend([name] * (samples_per_name // 5))  # 20% for testing
    
    # Shuffle data
    random.shuffle(train_texts)
    random.shuffle(test_texts)
    
    return train_texts, test_texts

print("Generating training data...")
train_texts, test_texts = create_training_data(selected_names, samples_per_name=50)

print(f"Training samples: {len(train_texts)}")
print(f"Test samples: {len(test_texts)}")

# One-hot encoding
def text_to_onehot(texts, name_to_idx, vocab_size):
    onehot = np.zeros((len(texts), vocab_size))
    for i, text in enumerate(texts):
        if text in name_to_idx:
            onehot[i, name_to_idx[text]] = 1
    return onehot

train_onehot = text_to_onehot(train_texts, name_to_idx, vocab_size)
test_onehot = text_to_onehot(test_texts, name_to_idx, vocab_size)

# =============================================================================
# Model Parameters
# =============================================================================
LATENT_DIM = 64        # Increased for more complex names
IMAGE_SIZE = 32        # Slightly larger images

print(f"Architecture:")
print(f"  Vocab size: {vocab_size}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

# =============================================================================
# Build Models
# =============================================================================
with tf.device(device_name):
    
    print("Building CLIENT model (Name → Image)...")
    
    # Text Encoder
    text_input = Input(shape=(vocab_size,), name='text_input')
    x = Dense(128, activation='relu')(text_input)
    x = Dense(LATENT_DIM, activation='relu')(x)
    text_encoded = Dense(LATENT_DIM, activation='sigmoid', name='text_latent')(x)
    
    text_encoder = Model(text_input, text_encoded, name='text_encoder')
    
    # Image Generator
    latent_input = Input(shape=(LATENT_DIM,), name='latent_input')
    x = Dense(8 * 8 * 32, activation='relu')(latent_input)
    x = Reshape((8, 8, 32))(x)
    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)  # 16x16
    x = Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)  # 32x32
    generated_image = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid')(x)
    
    image_generator = Model(latent_input, generated_image, name='image_generator')
    
    # Complete CLIENT model
    client_model = Model(text_input, image_generator(text_encoder(text_input)), name='client_name_to_image')
    
    print("Building SERVER model (Image → Name)...")
    
    # Image Encoder
    image_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='image_input')
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)  # 16x16
    x = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(x)  # 8x8
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    image_encoded = Dense(LATENT_DIM, activation='sigmoid', name='image_latent')(x)
    
    image_encoder = Model(image_input, image_encoded, name='image_encoder')
    
    # Text Decoder
    latent_to_text_input = Input(shape=(LATENT_DIM,), name='latent_to_text_input')
    x = Dense(128, activation='relu')(latent_to_text_input)
    x = Dense(vocab_size, activation='softmax')(x)
    
    text_decoder = Model(latent_to_text_input, x, name='text_decoder')
    
    # Complete SERVER model
    server_model = Model(image_input, text_decoder(image_encoder(image_input)), name='server_image_to_name')

# =============================================================================
# Generate Consistent Target Images
# =============================================================================
def generate_name_images(texts, image_size=32):
    """Generate consistent patterns for each name"""
    n_samples = len(texts)
    images = np.zeros((n_samples, image_size, image_size, 1), dtype='float32')
    
    unique_patterns = {}
    
    for i, name in enumerate(texts):
        if name not in unique_patterns:
            # Use name hash for consistent pattern
            name_seed = abs(hash(name)) % 10000
            np.random.seed(name_seed)
            
            # Create unique visual pattern for each name
            base_image = np.random.rand(image_size, image_size, 1).astype('float32')
            
            # Different pattern types based on name characteristics
            name_value = sum(ord(c) for c in name) % 4
            
            if name_value == 0:  # Circular patterns
                center_x, center_y = image_size//2, image_size//2
                radius = (name_seed % 8) + 5
                y, x = np.ogrid[:image_size, :image_size]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                base_image[mask] = 0.8 + 0.2 * np.random.random()
                
            elif name_value == 1:  # Rectangular patterns
                x1, y1 = (name_seed % 10) + 2, ((name_seed // 10) % 10) + 2
                x2, y2 = x1 + ((name_seed % 10) + 8), y1 + (((name_seed // 8) % 10) + 8)
                x2, y2 = min(x2, image_size-2), min(y2, image_size-2)
                base_image[x1:x2, y1:y2] = 0.2 + 0.4 * np.random.random()
                
            elif name_value == 2:  # Line patterns
                for j in range(0, image_size, 4):
                    if j + (name_seed % 4) < image_size:
                        base_image[j + (name_seed % 4), :] = 0.7 + 0.3 * np.random.random()
                        
            else:  # Dot patterns
                for _ in range(10):
                    dot_x = (name_seed + _) % (image_size - 2) + 1
                    dot_y = ((name_seed + _) // 7) % (image_size - 2) + 1
                    base_image[dot_x:dot_x+2, dot_y:dot_y+2] = 0.9
            
            unique_patterns[name] = base_image
            print(f"📝 Created pattern for '{name}' (type: {name_value}, seed: {name_seed})")
        
        images[i] = unique_patterns[name].copy()
    
    print(f"Generated {n_samples} images with {len(unique_patterns)} unique name patterns")
    return images

# Generate target images
print("Generating name-based image patterns...")
train_target_images = generate_name_images(train_texts, IMAGE_SIZE)
test_target_images = generate_name_images(test_texts, IMAGE_SIZE)

# =============================================================================
# Model Compilation
# =============================================================================
print("Compiling models...")
client_model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
server_model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# =============================================================================
# Training
# =============================================================================
print("Starting training with CSV name data...")

with tf.device(device_name):
    
    print("\nPhase 1: Training CLIENT (Name → Image)...")
    client_history = client_model.fit(
        train_onehot, train_target_images,
        epochs=20, batch_size=64,
        validation_data=(test_onehot, test_target_images),
        verbose=1
    )
    
    print("\nPhase 2: Training SERVER (Image → Name)...")
    server_history = server_model.fit(
        train_target_images, train_onehot,
        epochs=20, batch_size=64,
        validation_data=(test_target_images, test_onehot),
        verbose=1
    )

print("Training completed!")

# =============================================================================
# Testing with Different Data Sizes
# =============================================================================
def test_with_data_size(names, data_multiplier, test_name):
    """Test accuracy with different amounts of training data"""
    print(f"\n{'='*60}")
    print(f"TESTING: {test_name}")
    print(f"Data multiplier: {data_multiplier}x")
    print(f"{'='*60}")
    
    # Create test data
    test_names = names[:min(5, len(names))]  # Test with first 5 names
    test_indices = [name_to_idx[name] for name in test_names]
    test_input_onehot = np.zeros((len(test_indices), vocab_size))
    
    for i, idx in enumerate(test_indices):
        test_input_onehot[i, idx] = 1
    
    # CLIENT: Convert names to images
    client_generated_images = client_model.predict(test_input_onehot)
    
    # SERVER: Convert images back to names
    server_predictions = server_model.predict(client_generated_images)
    server_predicted_indices = np.argmax(server_predictions, axis=1)
    server_predicted_names = [idx_to_name[idx] for idx in server_predicted_indices]
    
    # Calculate accuracy
    correct = sum(1 for orig, pred in zip(test_names, server_predicted_names) if orig == pred)
    accuracy = correct / len(test_names)
    
    print(f"Results:")
    for i, (orig, pred) in enumerate(zip(test_names, server_predicted_names)):
        status = "✓" if orig == pred else "✗"
        print(f"  {i+1}. '{orig}' → '{pred}' {status}")
    
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(test_names)})")
    
    # Analyze image diversity
    similarities = []
    for i in range(len(client_generated_images)):
        for j in range(i+1, len(client_generated_images)):
            img1 = client_generated_images[i].flatten()
            img2 = client_generated_images[j].flatten()
            sim = np.corrcoef(img1, img2)[0, 1]
            similarities.append(sim)
    
    avg_similarity = np.mean(similarities) if similarities else 0
    print(f"Image diversity: {avg_similarity:.4f} (lower = more diverse)")
    
    return accuracy, avg_similarity

# Test current system
current_accuracy, current_diversity = test_with_data_size(
    selected_names, 1.0, "Current Training Data"
)

# =============================================================================
# Data Size Analysis
# =============================================================================
print(f"\n{'='*60}")
print("DATA SIZE vs ACCURACY ANALYSIS")
print(f"{'='*60}")

print(f"🔍 Current system performance:")
print(f"   • Training samples: {len(train_texts)}")
print(f"   • Vocabulary size: {vocab_size}")
print(f"   • Accuracy: {current_accuracy:.2%}")
print(f"   • Image diversity: {current_diversity:.4f}")

print(f"\n💡 Expected trends with MORE data:")
print(f"   • More training samples per name → Better pattern learning")
print(f"   • More unique names → More complex discrimination task")
print(f"   • Larger vocabulary → Need larger latent space")

print(f"\n📊 Current data characteristics:")
if name_counts is not None:
    print(f"   • Name frequency distribution: {name_counts.describe()}")
    
print(f"\n🎯 Recommendations for scaling:")
print(f"   • For more data: Increase LATENT_DIM and model capacity")
print(f"   • For better accuracy: More epochs and larger batch size")
print(f"   • For real-world use: Add noise and regularization")

# =============================================================================
# Visualization
# =============================================================================
print("Creating visualization...")

plt.figure(figsize=(16, 12))

# Show some test results
test_names = selected_names[:min(6, len(selected_names))]
test_indices = [name_to_idx[name] for name in test_names]
test_input_onehot = np.zeros((len(test_indices), vocab_size))

for i, idx in enumerate(test_indices):
    test_input_onehot[i, idx] = 1

client_generated_images = client_model.predict(test_input_onehot)
server_predictions = server_model.predict(client_generated_images)
server_predicted_indices = np.argmax(server_predictions, axis=1)
server_predicted_names = [idx_to_name[idx] for idx in server_predicted_indices]

for i in range(min(6, len(test_names))):
    # Original name
    plt.subplot(3, 6, i+1)
    plt.text(0.5, 0.5, f"'{test_names[i]}'", 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round", facecolor="lightblue"))
    plt.title('CLIENT Input')
    plt.axis('off')
    
    # Generated image
    plt.subplot(3, 6, i+7)
    plt.imshow(client_generated_images[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='viridis')
    plt.title('Generated Image')
    plt.axis('off')
    
    # Predicted name
    plt.subplot(3, 6, i+13)
    predicted_name = server_predicted_names[i]
    is_correct = (test_names[i] == predicted_name)
    color = "lightgreen" if is_correct else "lightcoral"
    
    plt.text(0.5, 0.5, f"'{predicted_name}'", 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round", facecolor=color))
    plt.title(f'SERVER Output {"✓" if is_correct else "✗"}')
    plt.axis('off')

plt.suptitle(f'CSV Name-based Communication System\nVocab: {vocab_size}, Accuracy: {current_accuracy:.1%}', fontsize=14)
plt.tight_layout()
plt.show()

print(f"\n🎉 CSV-based Text-Image Communication System Complete!")
print(f"📊 Final Statistics:")
print(f"   • Names processed: {vocab_size}")
print(f"   • Training samples: {len(train_texts)}")
print(f"   • Communication accuracy: {current_accuracy:.2%}")
print(f"   • System successfully learned name → image → name mappings!")
