#!/usr/bin/env python3
"""
Word2Vec-based Mosaic Image Classification System
1. Vectorize data using Word2Vec.
2. Convert vectors to mosaic images (smashed data).
3. Train a decoder to classify the images.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== Word2Vec Mosaic Classification System ===")
print("1. Text -> Word2Vec -> Mosaic Image")
print("2. Train Decoder on Mosaic Images for Classification")

class Word2VecProcessor:
    """Handles data vectorization using Word2Vec."""
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.w2v_model = None
        self.label_encoder = LabelEncoder()

    def train_word2vec(self, sentences):
        """Train Word2Vec model on the given sentences."""
        print(f"🧠 Training Word2Vec model with vector_size={self.vector_size}...")
        self.w2v_model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=4)
        print("✅ Word2Vec model trained.")

    def vectorize_data(self, df, text_column, label_column):
        """Vectorize text data and encode labels."""
        print(f"🔄 Vectorizing data from column '{text_column}'...")
        
        # Prepare sentences for Word2Vec
        sentences = [row.split() for row in df[text_column].astype(str)]
        
        # Debug: Show first 3 sentences after tokenization
        print("\n🔍 First 3 tokenized sentences:")
        for i in range(min(3, len(sentences))):
            print(f"  {i+1}. {sentences[i]}")
        
        self.train_word2vec(sentences)
        
        # Vectorize each sentence
        vectors = []
        for sentence in sentences:
            sentence_vector = np.mean([self.w2v_model.wv[word] for word in sentence if word in self.w2v_model.wv], axis=0)
            if np.isnan(sentence_vector).any():
                sentence_vector = np.zeros(self.vector_size)
            vectors.append(sentence_vector)
        
        vectors = np.array(vectors)
        
        # Debug: Show first vector example
        print(f"\n🔍 First vector (first 10 dimensions): {vectors[0][:10]}")
        print(f"🔍 Vector statistics - Min: {vectors.min():.4f}, Max: {vectors.max():.4f}, Mean: {vectors.mean():.4f}")
        
        # Encode labels
        labels = self.label_encoder.fit_transform(df[label_column])
        
        print(f"✅ Vectorization complete. Vector shape: {vectors.shape}")
        print(f"✅ Labels encoded. Number of classes: {len(self.label_encoder.classes_)}")
        
        return vectors, labels, self.label_encoder

class MosaicGenerator:
    """Converts vectors into mosaic images."""
    def __init__(self, image_size=32):
        self.image_size = image_size

    def vectors_to_mosaics(self, vectors):
        """Convert a list of vectors to a list of mosaic images."""
        print(f"🖼️  Generating {len(vectors)} mosaic images...")
        mosaics = []
        for vector in vectors:
            mosaic = self.vector_to_mosaic(vector)
            mosaics.append(mosaic)
        
        mosaics = np.array(mosaics)
        print(f"✅ Mosaics generated. Shape: {mosaics.shape}")
        return mosaics

    def vector_to_mosaic(self, vector):
        """Convert a single vector to a mosaic image."""
        # Simple reshaping for now, can be made more complex
        vector_len = len(vector)
        # Pad or truncate vector to fit into the image
        target_len = self.image_size * self.image_size
        if vector_len < target_len:
            padded_vector = np.pad(vector, (0, target_len - vector_len), 'constant')
        else:
            padded_vector = vector[:target_len]
            
        # Reshape to a grayscale image
        mosaic = padded_vector.reshape((self.image_size, self.image_size, 1))
        return mosaic

class MosaicClassifier:
    """Decoder model to classify mosaic images."""
    def __init__(self, image_size=32, num_classes=2):
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """Build the CNN-based decoder/classifier."""
        print("🏗️  Building mosaic classification model...")
        
        input_shape = (self.image_size, self.image_size, 1)
        
        image_input = Input(shape=input_shape, name='mosaic_input')
        
        # 더 깊고 복잡한 CNN 구조로 고해상도 이미지 처리
        x = Conv2D(64, (5, 5), activation='relu', padding='same')(image_input)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        output = Dense(self.num_classes, activation='softmax', name='class_output')(x)
        
        model = Model(image_input, output, name='mosaic_classifier')
        
        model.compile(optimizer=Adam(learning_rate=0.0001),  # 더 작은 학습률로 안정적 학습
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        print("✅ Model built and compiled.")
        model.summary()
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train the classifier."""
        print("🚀 Training the mosaic classifier...")
        
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_val, y_val),
                                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        
        print("✅ Training complete.")
        return history

def plot_history(history):
    """Plot training and validation history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function."""
    # 1. Load and prepare data
    print("\n--- Step 1: Data Loading and Preparation ---")
    try:
        df = pd.read_csv('ncvotera.csv', encoding='latin1')  # 모든 데이터 로드 (nrows 제한 제거)
        print(f"✅ Loaded {len(df)} records from ncvotera.csv")
        print(f"✅ Available columns: {list(df.columns)}")
        
        # Use ALL available columns except the target column for more information
        # Create a meaningful text column for Word2Vec using ALL relevant columns
        all_text_cols = [col for col in df.columns if col not in ['gender', 'age', 'zip_code']]  # Exclude target and numeric columns
        print(f"✅ Using {len(all_text_cols)} columns for text: {all_text_cols}")
        
        df['full_text'] = df[all_text_cols].astype(str).agg(' '.join, axis=1)
        
        # Debug: Show first 3 examples of full_text
        print("\n🔍 First 3 examples of input text:")
        for i in range(min(3, len(df))):
            print(f"  {i+1}. '{df.iloc[i]['full_text']}'")
        
        # Define the column for vectorization and the column for classification
        text_column = 'full_text'
        label_column = None # 전체 데이터의 정보 보존 테스트 (분류 없음)
        
        # Drop rows with missing labels
        initial_count = len(df)
        df.dropna(subset=[label_column], inplace=True)
        print(f"✅ After dropping missing labels: {len(df)} records (dropped {initial_count - len(df)})")
        
        # Balance the dataset - take equal numbers of male and female
        print(f"\n🔍 Original label distribution:")
        print(df[label_column].value_counts())
        
        # Only keep male and female (remove 'u' - unknown)
        df_balanced = df[df[label_column].isin(['m', 'f'])].copy()
        
        # Get equal samples for each gender - 더 많은 샘플 사용
        min_samples = df_balanced[label_column].value_counts().min()
        # 최대 20,000개까지 사용 (각 성별당 최대 20,000개)
        samples_to_use = min(min_samples, 20000)
        
        df_male = df_balanced[df_balanced[label_column] == 'm'].sample(n=samples_to_use, random_state=42)
        df_female = df_balanced[df_balanced[label_column] == 'f'].sample(n=samples_to_use, random_state=42)
        
        # Combine balanced dataset
        df = pd.concat([df_male, df_female], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n🔍 Balanced dataset:")
        print(f"  Total samples: {len(df)}")
        print(df[label_column].value_counts())
        
    except FileNotFoundError:
        print("❌ Error: ncvotera.csv not found. Please ensure the file is in the correct directory.")
        return
    except Exception as e:
        print(f"❌ An error occurred during data loading: {e}")
        return

    # 2. Vectorize data using Word2Vec
    print("\n--- Step 2: Vectorization ---")
    w2v_processor = Word2VecProcessor(vector_size=1024) # 매우 큰 벡터label_column = 'gender' 크기로 풍부한 의미 정보 활용
    vectors, labels, label_encoder = w2v_processor.vectorize_data(df, text_column, label_column)

    # 3. Generate mosaic images
    print("\n--- Step 3: Mosaic Generation ---")
    mosaic_gen = MosaicGenerator(image_size=64) # 64x64 = 4096 크기의 고해상도 모자이크 이미지
    mosaics = mosaic_gen.vectors_to_mosaics(vectors)

    # 4. Train the classifier
    print("\n--- Step 4: Model Training ---")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(mosaics, labels, test_size=0.2, random_state=42, stratify=labels)
    
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes to predict: {num_classes} ({label_encoder.classes_})")
    
    classifier = MosaicClassifier(image_size=64, num_classes=num_classes)
    
    history = classifier.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=16)  # 더 작은 배치 크기와 더 많은 에포크로 세밀한 학습

    # 5. Evaluate the model
    print("\n--- Step 5: Evaluation ---")
    loss, accuracy = classifier.model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Test Loss: {loss:.4f}")
    
    # 6. Detailed Analysis of Predictions
    print("\n--- Step 6: Detailed Prediction Analysis ---")
    
    # Get predictions
    y_pred_proba = classifier.model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Show class mapping
    class_names = label_encoder.classes_
    print(f"🔍 Class mapping: {dict(enumerate(class_names))}")
    
    # Test set distribution
    print(f"\n🔍 Test set distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  Class {cls} ({class_names[cls]}): {count} samples ({count/len(y_test)*100:.1f}%)")
    
    # Prediction distribution
    print(f"\n🔍 Prediction distribution:")
    unique, counts = np.unique(y_pred, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  Class {cls} ({class_names[cls]}): {count} predictions ({count/len(y_pred)*100:.1f}%)")
    
    # Detailed classification report
    print(f"\n🔍 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Confusion Matrix
    print(f"\n🔍 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("Actual \\ Predicted:", end="")
    for i, name in enumerate(class_names):
        print(f"\t{name}", end="")
    print()
    for i, (actual_class, row) in enumerate(zip(class_names, cm)):
        print(f"{actual_class}\t\t", end="")
        for val in row:
            print(f"\t{val}", end="")
        print()
    
    # Calculate accuracy for each class
    print(f"\n🔍 Per-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum((y_pred == i) & class_mask) / np.sum(class_mask)
            print(f"  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    # Show some examples of correct and incorrect predictions
    print(f"\n🔍 Example predictions:")
    correct_indices = np.where(y_pred == y_test)[0][:3]
    incorrect_indices = np.where(y_pred != y_test)[0][:3]
    
    print("✅ Correct predictions:")
    for idx in correct_indices:
        actual = class_names[y_test[idx]]
        predicted = class_names[y_pred[idx]]
        confidence = y_pred_proba[idx][y_pred[idx]]
        print(f"  Sample {idx}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.4f}")
    
    print("❌ Incorrect predictions:")
    for idx in incorrect_indices:
        actual = class_names[y_test[idx]]
        predicted = class_names[y_pred[idx]]
        confidence = y_pred_proba[idx][y_pred[idx]]
        print(f"  Sample {idx}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.4f}")

    # Plot results
    plot_history(history)

if __name__ == "__main__":
    main()
