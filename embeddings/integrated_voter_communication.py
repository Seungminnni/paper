#!/usr/bin/env python3
"""
Integrated 19-Column Voter Data Communication System
- Processes complete voter records with all 19 features as integrated data
- Tests text→image→text communication for complete voter profiles
- Each record contains: ID, names, demographics, location, dates as single unit
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import random
import json
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib and seaborn style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# MPS (Apple Silicon M4 GPU) setup
print("=== Integrated 19-Column Voter Data Communication System ===")
print(f"🍎 Running on Apple Silicon M4 with optimized MPS backend")

try:
    # Enhanced MPS detection for M4
    physical_devices = tf.config.list_physical_devices()
    print(f"Available devices: {[device.name for device in physical_devices]}")
    
    # Check for MPS (Metal Performance Shaders) support
    if tf.config.list_physical_devices('GPU'):
        gpus = tf.config.list_physical_devices('GPU')
        print(f"✅ GPU devices detected: {len(gpus)}")
        
        # Enable memory growth for MPS
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled for MPS")
        except:
            print("⚠️  Memory growth setting not available")
            
        device_name = '/GPU:0'
        gpu_available = True
        print("🚀 MPS (Apple Silicon M4 GPU) is ready for training!")
        
    else:
        # Fallback: try to enable MPS manually
        try:
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
                result = tf.reduce_sum(test_tensor)
                print(f"✅ MPS test successful: {result.numpy()}")
            device_name = '/GPU:0'
            gpu_available = True
            print("🍎 MPS (Apple Silicon M4 GPU) manually enabled!")
        except Exception as e:
            print(f"❌ MPS not available: {e}")
            device_name = '/CPU:0'
            gpu_available = False
            
except Exception as e:
    print(f"🔴 GPU/MPS setup error: {e}")
    print("🔄 Falling back to CPU")
    device_name = '/CPU:0'
    gpu_available = False

print(f"🎯 Selected device: {device_name}")
print(f"📊 TensorFlow version: {tf.__version__}")

# M4-optimized MPS settings
if gpu_available and device_name == '/GPU:0':
    try:
        # Enable optimizations for M4
        tf.config.experimental.enable_mlir_graph_optimization()
        print("✅ MLIR graph optimization enabled for M4")
    except:
        print("⚠️  Some MLIR optimizations not available (this is normal)")
        
    try:
        # Additional M4 optimizations
        tf.config.experimental.enable_mlir_bridge()
        print("✅ MLIR bridge enabled for enhanced performance")
    except:
        print("ℹ️  MLIR bridge not available (optional feature)")
        pass

def load_integrated_csv_data(csv_file, max_rows=1000):
    """Load CSV data as integrated records with all 19 features per row"""
    try:
        print(f"Loading integrated CSV data from {csv_file}...")
        
        # Try different encodings to handle various CSV formats
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings_to_try:
            try:
                if max_rows is None:
                    print(f"📊 Loading ENTIRE dataset (no row limit) with {encoding} encoding")
                    df = pd.read_csv(csv_file, encoding=encoding)
                else:
                    print(f"📊 Loading {max_rows} rows with {encoding} encoding")
                    df = pd.read_csv(csv_file, nrows=max_rows, encoding=encoding)
                print(f"✅ Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"❌ Failed with {encoding} encoding, trying next...")
                continue
        
        if df is None:
            raise Exception("Could not read CSV with any of the tried encodings")
            
        print(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Expected columns for voter data
        expected_columns = [
            'voter_id', 'voter_reg_num', 'name_prefix', 'first_name', 'middle_name', 
            'last_name', 'name_suffix', 'age', 'gender', 'race', 'ethnic', 
            'street_address', 'city', 'state', 'zip_code', 'full_phone_num', 
            'birth_place', 'register_date', 'download_month'
        ]
        
        # Check available columns
        available_columns = [col for col in expected_columns if col in df.columns]
        print(f"Available columns: {len(available_columns)}/19 - {available_columns}")
        
        # Clean and prepare data
        processed_df = df[available_columns].copy()
        
        # Fill missing values with appropriate defaults
        for col in available_columns:
            if col in ['voter_id', 'voter_reg_num']:
                processed_df[col] = processed_df[col].fillna('unknown_id').astype(str)
            elif col in ['name_prefix', 'name_suffix']:
                processed_df[col] = processed_df[col].fillna('').astype(str).str.lower().str.strip()
            elif col in ['first_name', 'middle_name', 'last_name']:
                processed_df[col] = processed_df[col].fillna('unknown').astype(str).str.lower().str.strip()
            elif col == 'age':
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
            elif col in ['gender', 'race', 'ethnic']:
                processed_df[col] = processed_df[col].fillna('unknown').astype(str).str.lower().str.strip()
            elif col in ['street_address', 'city']:
                processed_df[col] = processed_df[col].fillna('unknown').astype(str).str.lower().str.strip()
            elif col == 'state':
                processed_df[col] = processed_df[col].fillna('unknown').astype(str).str.upper().str.strip()
            elif col in ['zip_code', 'full_phone_num']:
                processed_df[col] = processed_df[col].fillna('00000').astype(str).str.strip()
            elif col == 'birth_place':
                processed_df[col] = processed_df[col].fillna('unknown').astype(str).str.upper().str.strip()
            elif col in ['register_date', 'download_month']:
                processed_df[col] = processed_df[col].fillna('unknown').astype(str).str.strip()
        
        # Remove rows with too many missing values
        processed_df = processed_df.dropna(thresh=len(available_columns)//2)
        
        print(f"✅ Cleaned data: {len(processed_df):,} rows ready for integrated training")
        
        return processed_df, available_columns
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, None

def create_integrated_record_vectors(df, available_columns):
    """Create 19×1 vector structure: each of 19 features gets 1 normalized value"""
    print(f"Creating 19×1 vector structure for {len(df)} records with {len(available_columns)} features...")
    
    # Ensure we have exactly 19 features (pad with dummy if needed)
    expected_columns = [
        'voter_id', 'voter_reg_num', 'name_prefix', 'first_name', 'middle_name', 
        'last_name', 'name_suffix', 'age', 'gender', 'race', 'ethnic', 
        'street_address', 'city', 'state', 'zip_code', 'full_phone_num', 
        'birth_place', 'register_date', 'download_month'
    ]
    
    # Create normalization mappings for each column
    normalizations = {}
    
    for col in expected_columns:
        print(f"Processing column: {col}")
        
        if col not in available_columns:
            # Missing column - will use 0.0
            normalizations[col] = {'type': 'missing'}
            continue
            
        if col in ['voter_id', 'voter_reg_num']:
            # Numeric IDs - normalize by hash
            unique_vals = df[col].unique()
            max_hash = max([hash(str(val)) % 10000 for val in unique_vals])
            normalizations[col] = {'type': 'hash', 'max_val': max_hash}
            
        elif col in ['name_prefix', 'first_name', 'middle_name', 'last_name', 'name_suffix']:
            # Names - normalize by string hash
            unique_vals = df[col].unique()
            max_hash = max([hash(str(val)) % 10000 for val in unique_vals])
            normalizations[col] = {'type': 'hash', 'max_val': max_hash}
            
        elif col == 'age':
            # Age - direct normalization (0-100)
            max_age = df[col].max() if pd.notna(df[col].max()) else 100
            normalizations[col] = {'type': 'numeric', 'max_val': max_age}
            
        elif col in ['gender', 'race', 'ethnic', 'state', 'birth_place']:
            # Categorical data - assign numeric values
            unique_vals = df[col].unique()
            categories = {str(val): i for i, val in enumerate(unique_vals)}
            normalizations[col] = {'type': 'categorical', 'categories': categories, 'max_val': len(categories)}
            
        elif col in ['street_address', 'city']:
            # Location data - normalize by hash
            unique_vals = df[col].unique()
            max_hash = max([hash(str(val)) % 10000 for val in unique_vals])
            normalizations[col] = {'type': 'hash', 'max_val': max_hash}
            
        elif col in ['zip_code', 'full_phone_num']:
            # Code patterns - extract first 3 digits and normalize
            if col == 'zip_code':
                df[col + '_prefix'] = df[col].astype(str).str[:3]
                max_val = 999  # max 3-digit number
            else:  # phone numbers
                df[col + '_prefix'] = df[col].astype(str).str.replace(r'\D', '', regex=True).str[:3]
                max_val = 999
            
            normalizations[col] = {'type': 'numeric_prefix', 'max_val': max_val}
            
        elif col in ['register_date', 'download_month']:
            # Date patterns - extract numeric part and normalize
            if col == 'register_date':
                df[col + '_year'] = df[col].astype(str).str.extract(r'(\d{4})').astype(float)
                max_val = df[col + '_year'].max() if pd.notna(df[col + '_year'].max()) else 2025
                min_val = df[col + '_year'].min() if pd.notna(df[col + '_year'].min()) else 1900
            else:
                df[col + '_month'] = df[col].astype(str).str.split('.').str[1].astype(float, errors='ignore')
                max_val = 12
                min_val = 1
                
            normalizations[col] = {'type': 'date', 'max_val': max_val, 'min_val': min_val}
    
    print(f"Created normalizations for {len(normalizations)} columns")
    
    # Create 19×1 vector structure
    integrated_vectors = []
    integrated_text_labels = []
    
    for idx, row in df.iterrows():
        # Create 19-dimensional vector for this record
        record_vector = np.zeros(19, dtype='float32')
        text_parts = []
        
        for feature_idx, col in enumerate(expected_columns):
            if col not in available_columns:
                # Missing column - use 0.0
                record_vector[feature_idx] = 0.0
                text_parts.append(f"{col}:missing")
                continue
                
            # Get normalized value for this feature
            norm_info = normalizations[col]
            
            if norm_info['type'] == 'missing':
                val = 0.0
                val_str = "missing"
                
            elif norm_info['type'] == 'hash':
                val_str = str(row[col])
                val = (hash(val_str) % 10000) / norm_info['max_val']
                
            elif norm_info['type'] == 'numeric':
                try:
                    val = float(row[col]) / norm_info['max_val']
                    val_str = str(row[col])
                except:
                    val = 0.0
                    val_str = "0"
                    
            elif norm_info['type'] == 'categorical':
                val_str = str(row[col])
                cat_idx = norm_info['categories'].get(val_str, 0)
                val = cat_idx / norm_info['max_val']
                
            elif norm_info['type'] == 'numeric_prefix':
                if col + '_prefix' in row:
                    try:
                        prefix_val = float(row[col + '_prefix'])
                        val = prefix_val / norm_info['max_val']
                        val_str = str(row[col + '_prefix'])
                    except:
                        val = 0.0
                        val_str = "0"
                else:
                    val = 0.0
                    val_str = "0"
                    
            elif norm_info['type'] == 'date':
                if col == 'register_date' and col + '_year' in row:
                    try:
                        year_val = float(row[col + '_year'])
                        val = (year_val - norm_info['min_val']) / (norm_info['max_val'] - norm_info['min_val'])
                        val_str = str(int(year_val))
                    except:
                        val = 0.0
                        val_str = "unknown"
                elif col == 'download_month' and col + '_month' in row:
                    try:
                        month_val = float(row[col + '_month'])
                        val = (month_val - norm_info['min_val']) / (norm_info['max_val'] - norm_info['min_val'])
                        val_str = str(int(month_val))
                    except:
                        val = 0.0
                        val_str = "unknown"
                else:
                    val = 0.0
                    val_str = "unknown"
            else:
                val = 0.0
                val_str = "unknown"
            
            # Ensure value is between 0 and 1
            val = max(0.0, min(1.0, val))
            record_vector[feature_idx] = val
            text_parts.append(f"{col}:{val_str}")
        
        integrated_vectors.append(record_vector)
        integrated_text_labels.append(" | ".join(text_parts))
    
    integrated_vectors = np.array(integrated_vectors)
    
    print(f"✅ Created {len(integrated_vectors)} integrated vectors")
    print(f"   Vector shape: {integrated_vectors.shape} (batch_size, 19_features)")
    print(f"   Sample record: {integrated_text_labels[0][:100]}...")
    
    return integrated_vectors, integrated_text_labels, normalizations, 19

def generate_integrated_record_images(df, available_columns, image_size=64):
    """Generate unique visual patterns for complete voter records (all 19 features)"""
    n_records = len(df)
    images = np.zeros((n_records, image_size, image_size, 3), dtype='float32')  # RGB for richer encoding
    
    print(f"Creating integrated visual patterns for {n_records} complete voter records...")
    
    for idx, (_, row) in enumerate(df.iterrows()):
        # Create unique pattern based on ALL features of the record
        record_seed = 0
        
        # Combine all features to create unique seed
        for col in available_columns:
            val_str = str(row[col]).lower()
            record_seed += sum(ord(c) for c in val_str if c.isalpha())
        
        record_seed = record_seed % 1000000
        np.random.seed(record_seed)
        
        # Initialize base image
        base_image = np.random.rand(image_size, image_size, 3).astype('float32') * 0.1
        
        # Layer 1: ID Information (voter_id, voter_reg_num) - Blue channel
        if 'voter_id' in available_columns:
            voter_id = str(row['voter_id'])
            id_hash = sum(ord(c) for c in voter_id if c.isdigit()) % 20
            # Create ID barcode pattern
            for x in range(0, image_size, 3):
                if (x + id_hash) % 4 == 0:
                    height = 8 + (id_hash % 12)
                    start_y = max(0, image_size//2 - height//2)
                    end_y = min(image_size, start_y + height)
                    base_image[start_y:end_y, x, 2] = 0.3 + (id_hash % 5) * 0.1
        
        # Layer 2: Name Information - Green channel
        name_parts = []
        for name_col in ['name_prefix', 'first_name', 'middle_name', 'last_name', 'name_suffix']:
            if name_col in available_columns and pd.notna(row[name_col]):
                name_parts.append(str(row[name_col]))
        
        if name_parts:
            full_name = " ".join(name_parts).lower()
            name_hash = sum(ord(c) for c in full_name) % 16
            
            # Create name pattern - geometric shapes
            if name_hash % 4 == 0:  # Circle
                center = image_size // 2
                radius = 8 + (name_hash % 8)
                y, x = np.ogrid[:image_size, :image_size]
                mask = (x - center)**2 + (y - center)**2 <= radius**2
                base_image[mask, 1] = 0.4 + (name_hash % 6) * 0.08
            elif name_hash % 4 == 1:  # Rectangle
                size = 10 + (name_hash % 10)
                start = image_size//2 - size//2
                end = start + size
                base_image[start:end, start:end, 1] = 0.5 + (name_hash % 5) * 0.1
            elif name_hash % 4 == 2:  # Cross
                mid = image_size // 2
                thickness = 2 + (name_hash % 4)
                base_image[mid-thickness:mid+thickness, :, 1] = 0.6
                base_image[:, mid-thickness:mid+thickness, 1] = 0.6
            else:  # Diagonal pattern
                for i in range(image_size):
                    j = (i + name_hash) % image_size
                    base_image[i, j, 1] = 0.7
        
        # Layer 3: Demographics (age, gender, race, ethnic) - Red channel  
        demo_hash = 0
        if 'age' in available_columns:
            try:
                age = float(row['age'])
                demo_hash += int(age)
            except:
                demo_hash += 50
                
        for demo_col in ['gender', 'race', 'ethnic']:
            if demo_col in available_columns and pd.notna(row[demo_col]):
                demo_hash += sum(ord(c) for c in str(row[demo_col]).lower())
        
        demo_hash = demo_hash % 24
        
        # Create demographic pattern - wave-like
        amplitude = 4 + (demo_hash % 8)
        frequency = 0.1 + (demo_hash % 6) * 0.05
        
        for i in range(image_size):
            wave_y = int(image_size//2 + amplitude * np.sin(frequency * i))
            wave_y = max(0, min(wave_y, image_size-1))
            
            # Primary wave
            if 0 <= wave_y < image_size:
                base_image[wave_y, i, 0] = 0.5 + (demo_hash % 5) * 0.08
            
            # Secondary wave
            wave_y2 = int(image_size//2 + (amplitude//2) * np.cos(frequency * 1.5 * i))
            wave_y2 = max(0, min(wave_y2, image_size-1))
            if 0 <= wave_y2 < image_size:
                base_image[wave_y2, i, 0] = 0.3 + (demo_hash % 4) * 0.1
        
        # Layer 4: Location Information - Combined channels
        location_hash = 0
        for loc_col in ['street_address', 'city', 'state', 'zip_code']:
            if loc_col in available_columns and pd.notna(row[loc_col]):
                location_hash += sum(ord(c) for c in str(row[loc_col]).lower() if c.isalpha())
        
        location_hash = location_hash % 30
        
        # Create location pattern - grid-like
        grid_size = 4 + (location_hash % 6)
        for i in range(0, image_size, grid_size):
            for j in range(0, image_size, grid_size):
                if (i//grid_size + j//grid_size + location_hash) % 3 == 0:
                    end_i = min(i + grid_size//2, image_size)
                    end_j = min(j + grid_size//2, image_size)
                    intensity = 0.2 + (location_hash % 6) * 0.05
                    base_image[i:end_i, j:end_j, :] += intensity
        
        # Layer 5: Temporal Information (dates) - Overlay pattern
        temp_hash = 0
        for date_col in ['register_date', 'download_month']:
            if date_col in available_columns and pd.notna(row[date_col]):
                temp_hash += sum(ord(c) for c in str(row[date_col]) if c.isdigit())
        
        temp_hash = temp_hash % 40
        
        # Create temporal pattern - spiral
        center = image_size // 2
        for angle in range(0, 360, 15):
            radius = (angle / 60) + (temp_hash % 8)
            x = int(center + radius * np.cos(np.radians(angle)))
            y = int(center + radius * np.sin(np.radians(angle)))
            if 0 <= x < image_size and 0 <= y < image_size:
                channel = temp_hash % 3
                base_image[x, y, channel] = min(1.0, base_image[x, y, channel] + 0.4)
        
        # Normalize and apply final adjustments
        base_image = np.clip(base_image, 0.0, 1.0)
        
        # Add slight noise for uniqueness
        noise_level = 0.05
        base_image += np.random.normal(0, noise_level, base_image.shape)
        base_image = np.clip(base_image, 0.0, 1.0)
        
        images[idx] = base_image
        
        if idx < 5:  # Show progress for first few
            print(f"  ✓ Created integrated pattern for record {idx+1} (seed: {record_seed})")
    
    if n_records > 5:
        print(f"  ... and {n_records - 5} more integrated patterns created")
    
    print(f"✅ Generated {n_records} integrated RGB images ({image_size}x{image_size}x3)")
    
    return images

def build_integrated_record_models(input_dim=19, batch_size=32, latent_dim=128, learning_rate=0.0005):
    """Build models for mosaic-style batch processing with M4 MPS optimization"""
    
    print(f"Building mosaic batch processing models on device: {device_name}")
    print(f"Input: (batch_size={batch_size}, features={input_dim})")
    print(f"Output: Single mosaic image containing all batch data")
    
    with tf.device(device_name):
        
        # CLIENT: Batch of 19D vectors → Single Mosaic Image
        # Input: (batch_size, 19)
        batch_input = Input(shape=(input_dim,), name='batch_19d_input')
        
        # Encode each individual record
        x = Dense(64, activation='relu')(batch_input)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        encoded_record = Dense(16, activation='sigmoid')(x)  # Each record → 16D
        
        # This model processes one record at a time, but will be applied to entire batch
        record_encoder = Model(batch_input, encoded_record, name='record_encoder')
        
        # Mosaic Image Generator: Takes encoded batch and creates single image
        # Input will be (batch_size, 16) → reshape to mosaic image
        mosaic_input = Input(shape=(batch_size, 16), name='mosaic_input')
        
        # Treat the (batch_size, 16) as a 2D "image" and apply Conv2D
        # Pad to make it square-ish for better conv operations
        x = tf.keras.layers.ZeroPadding2D(padding=((0, 64-batch_size), (0, 48)))(mosaic_input)  # Make it 64x64
        x = tf.expand_dims(x, axis=-1)  # Add channel dimension: (64, 64, 1)
        
        # Apply Conv2D to create richer representation
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) 
        x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # RGB output
        
        mosaic_image = x  # (64, 64, 3) - Single mosaic image containing all batch data
        
        mosaic_generator = Model(mosaic_input, mosaic_image, name='mosaic_generator')
        
        # Complete CLIENT model: batch of records → single mosaic image
        # This will process the batch through record_encoder, then create mosaic
        def create_mosaic_from_batch(batch_records):
            # Apply record encoder to each item in batch
            encoded_batch = record_encoder(batch_records)  # (batch_size, 16)
            # Reshape for mosaic generation
            encoded_batch = tf.expand_dims(encoded_batch, 0)  # (1, batch_size, 16)
            mosaic = mosaic_generator(encoded_batch)  # (1, 64, 64, 3)
            return mosaic[0]  # Return (64, 64, 3)
        
        # SERVER: Single Mosaic Image → Batch of 19D vectors
        # Input: Single mosaic image containing batch data
        mosaic_image_input = Input(shape=(64, 64, 3), name='mosaic_image_input')
        
        # Extract features from mosaic image
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(mosaic_image_input)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        
        # Crop back to the useful region (batch_size, 16)
        x = x[:, :batch_size, :16, :]  # Extract (batch_size, 16, 16) region
        x = tf.reduce_mean(x, axis=-1)  # Average over channels: (batch_size, 16)
        
        # Decode each record from 16D back to 19D
        decoded_records = []
        for i in range(batch_size):
            record_features = x[:, i, :]  # (batch_size=1, 16)
            # Decode this record
            decoded = Dense(32, activation='relu')(record_features)
            decoded = Dropout(0.2)(decoded)
            decoded = Dense(64, activation='relu')(decoded)
            decoded = Dense(input_dim, activation='sigmoid')(decoded)  # Back to 19D
            decoded_records.append(decoded)
        
        # Stack all decoded records
        batch_output = tf.stack(decoded_records, axis=1)  # (1, batch_size, 19)
        batch_output = tf.squeeze(batch_output, axis=0)    # (batch_size, 19)
        
        mosaic_decoder = Model(mosaic_image_input, batch_output, name='mosaic_decoder')
        
        # Compile models
        record_encoder.compile(
            optimizer=Adam(learning_rate), 
            loss='mse', 
            metrics=['mae']
        )
        
        mosaic_generator.compile(
            optimizer=Adam(learning_rate), 
            loss='mse', 
            metrics=['mae']
        )
        
        mosaic_decoder.compile(
            optimizer=Adam(learning_rate), 
            loss='mse', 
            metrics=['mae']
        )
        
        print(f"✅ Mosaic batch processing models built successfully on {device_name}")
        print(f"   Record encoder parameters: {record_encoder.count_params():,}")
        print(f"   Mosaic generator parameters: {mosaic_generator.count_params():,}")
        print(f"   Mosaic decoder parameters: {mosaic_decoder.count_params():,}")
        print(f"   🍎 M4-optimized learning rate: {learning_rate}")
        print(f"   📊 Batch processing: {batch_size} records → 1 mosaic image → {batch_size} records")
        
        return record_encoder, mosaic_generator, mosaic_decoder, create_mosaic_from_batch

def test_integrated_communication(client_model, server_model, test_vectors, test_images, text_labels, vocabularies, available_columns, num_tests=5):
    """Test the complete integrated communication system"""
    
    print(f"\n🧪 Testing integrated 19-column communication with {num_tests} samples...")
    
    # Randomly select test samples
    test_indices = np.random.choice(len(test_vectors), num_tests, replace=False)
    
    results = []
    
    for i, test_idx in enumerate(test_indices):
        print(f"\n--- Test {i+1}/{num_tests} ---")
        
        # Original data
        original_vector = test_vectors[test_idx:test_idx+1]
        original_image = test_images[test_idx:test_idx+1]
        original_label = text_labels[test_idx]
        
        print(f"Original record: {original_label[:150]}...")
        
        # CLIENT: Text → Image
        generated_image = client_model.predict(original_vector, verbose=0)
        
        # SERVER: Image → Text  
        reconstructed_vector = server_model.predict(generated_image, verbose=0)
        
        # Calculate similarity
        vector_similarity = np.dot(original_vector.flatten(), reconstructed_vector.flatten()) / (
            np.linalg.norm(original_vector.flatten()) * np.linalg.norm(reconstructed_vector.flatten())
        )
        
        # Decode reconstructed vector to readable format
        reconstructed_features = []
        offset = 0
        for col in available_columns:
            if col in vocabularies:
                col_size = len(vocabularies[col])
                col_probs = reconstructed_vector[0][offset:offset+col_size]
                best_idx = np.argmax(col_probs)
                confidence = col_probs[best_idx]
                
                # Find the value corresponding to this index
                for val, idx in vocabularies[col].items():
                    if idx == best_idx:
                        reconstructed_features.append(f"{col}:{val}({confidence:.3f})")
                        break
                
                offset += col_size
        
        reconstructed_label = " | ".join(reconstructed_features)
        print(f"Reconstructed: {reconstructed_label[:150]}...")
        print(f"Vector similarity: {vector_similarity:.4f}")
        
        # Analyze feature-level accuracy
        feature_accuracy = 0
        offset = 0
        for col in available_columns:
            if col in vocabularies:
                col_size = len(vocabularies[col])
                original_feature = np.argmax(original_vector[0][offset:offset+col_size])
                reconstructed_feature = np.argmax(reconstructed_vector[0][offset:offset+col_size])
                
                if original_feature == reconstructed_feature:
                    feature_accuracy += 1
                
                offset += col_size
        
        feature_accuracy = feature_accuracy / len(vocabularies)
        print(f"Feature accuracy: {feature_accuracy:.4f}")
        
        results.append({
            'test_idx': test_idx,
            'vector_similarity': vector_similarity,
            'feature_accuracy': feature_accuracy,
            'original_label': original_label,
            'reconstructed_label': reconstructed_label
        })
    
    # Summary
    avg_vector_similarity = np.mean([r['vector_similarity'] for r in results])
    avg_feature_accuracy = np.mean([r['feature_accuracy'] for r in results])
    
    print(f"\n📊 INTEGRATED COMMUNICATION TEST RESULTS:")
    print(f"   Average vector similarity: {avg_vector_similarity:.4f}")
    print(f"   Average feature accuracy: {avg_feature_accuracy:.4f}")
    
    return results, avg_vector_similarity, avg_feature_accuracy

def create_training_visualization(client_history, server_history, test_results):
    """Create comprehensive training and testing visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Integrated 19-Column Voter Data Training Results', fontsize=16, fontweight='bold')
    
    # CLIENT training history
    axes[0, 0].plot(client_history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    axes[0, 0].plot(client_history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    axes[0, 0].set_title('CLIENT Model: Text → Image')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # SERVER training history
    axes[0, 1].plot(server_history.history['accuracy'], label='Training Accuracy', color='green', linewidth=2)
    axes[0, 1].plot(server_history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
    axes[0, 1].set_title('SERVER Model: Image → Text')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning curves comparison
    axes[0, 2].plot(client_history.history['loss'], label='CLIENT Loss', color='blue', alpha=0.7)
    axes[0, 2].plot(server_history.history['accuracy'], label='SERVER Accuracy', color='green', alpha=0.7)
    axes[0, 2].set_title('Training Progress Comparison')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Test results visualization
    if test_results:
        vector_similarities = [r['vector_similarity'] for r in test_results]
        feature_accuracies = [r['feature_accuracy'] for r in test_results]
        
        axes[1, 0].bar(range(len(vector_similarities)), vector_similarities, 
                      color='skyblue', alpha=0.7, edgecolor='navy')
        axes[1, 0].set_title('Vector Similarity by Test Sample')
        axes[1, 0].set_xlabel('Test Sample')
        axes[1, 0].set_ylabel('Similarity')
        axes[1, 0].set_ylim(0, 1)
        
        axes[1, 1].bar(range(len(feature_accuracies)), feature_accuracies, 
                      color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        axes[1, 1].set_title('Feature Accuracy by Test Sample')
        axes[1, 1].set_xlabel('Test Sample')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim(0, 1)
        
        # Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""Training Summary:

CLIENT Model:
• Final Loss: {client_history.history['loss'][-1]:.6f}
• Best Loss: {min(client_history.history['loss']):.6f}
• Total Epochs: {len(client_history.history['loss'])}

SERVER Model:
• Final Accuracy: {server_history.history['accuracy'][-1]:.4f}
• Best Accuracy: {max(server_history.history['accuracy']):.4f}

Communication Test:
• Avg Vector Similarity: {np.mean(vector_similarities):.4f}
• Avg Feature Accuracy: {np.mean(feature_accuracies):.4f}
• Test Samples: {len(test_results)}

Status: {'✅ EXCELLENT' if np.mean(feature_accuracies) > 0.8 else '🟡 GOOD' if np.mean(feature_accuracies) > 0.6 else '🔴 NEEDS IMPROVEMENT'}
"""
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"integrated_voter_training_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training visualization saved: {plot_filename}")
    return plot_filename

def save_integrated_results(df, available_columns, vocabularies, test_results, avg_vector_similarity, avg_feature_accuracy):
    """Save comprehensive results to files"""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON results
    json_filename = f"integrated_voter_results_{timestamp}.json"
    json_data = {
        "timestamp": timestamp,
        "device_used": device_name,
        "dataset_info": {
            "total_records": len(df),
            "available_columns": len(available_columns),
            "columns": available_columns,
            "vocabulary_sizes": {col: len(vocab) for col, vocab in vocabularies.items()},
            "total_vocab_size": sum(len(vocab) for vocab in vocabularies.values())
        },
        "training_results": {
            "avg_vector_similarity": float(avg_vector_similarity),
            "avg_feature_accuracy": float(avg_feature_accuracy),
            "test_samples": len(test_results)
        },
        "detailed_test_results": [
            {
                "test_idx": int(r['test_idx']),
                "vector_similarity": float(r['vector_similarity']),
                "feature_accuracy": float(r['feature_accuracy']),
                "original_sample": r['original_label'][:200],
                "reconstructed_sample": r['reconstructed_label'][:200]
            } for r in test_results
        ]
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Results saved to: {json_filename}")
    return json_filename

# =============================================================================
# Main Execution - Integrated 19-Column Training
# =============================================================================

def main():
    print("\n🚀 STARTING INTEGRATED 19-COLUMN VOTER DATA TRAINING")
    print("="*80)
    
    # Load integrated data
    df, available_columns = load_integrated_csv_data('ncvaa.csv', max_rows=1000)
    
    if df is None:
        print("❌ Failed to load integrated data")
        return
    
    print(f"\n📊 Integrated data loaded:")
    print(f"   Records: {len(df):,}")
    print(f"   Available columns: {len(available_columns)}/19")
    print(f"   Columns: {available_columns}")
    
    # Create integrated feature vectors and images
    print("\n🔧 Processing integrated records...")
    integrated_vectors, text_labels, vocabularies, total_vocab_size = create_integrated_record_vectors(df, available_columns)
    integrated_images = generate_integrated_record_images(df, available_columns, image_size=64)
    
    print(f"\n📊 Integrated data prepared:")
    print(f"   Feature vectors: {integrated_vectors.shape}")
    print(f"   Images: {integrated_images.shape}")
    print(f"   Total vocabulary size: {total_vocab_size}")
    
    # Build integrated models
    print("\n🏗️ Building integrated models...")
    client_model, server_model, text_encoder, image_generator, image_encoder, text_decoder = build_integrated_record_models(
        vocab_size=total_vocab_size,
        latent_dim=128,
        image_size=64,
        learning_rate=0.0005
    )
    
    # Training setup
    print("\n🎯 Starting integrated training...")
    batch_size = 32
    epochs = 50
    
    # Split data
    split_idx = int(0.8 * len(integrated_vectors))
    train_vectors = integrated_vectors[:split_idx]
    train_images = integrated_images[:split_idx]
    test_vectors = integrated_vectors[split_idx:]
    test_images = integrated_images[split_idx:]
    
    print(f"Training data: {len(train_vectors)} records")
    print(f"Test data: {len(test_vectors)} records")
    
    # Train CLIENT model (Text → Image)
    print("\n🔧 Training CLIENT model (Text → Image)...")
    client_callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.7, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
    ]
    
    client_history = client_model.fit(
        train_vectors, train_images,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=client_callbacks,
        verbose=1
    )
    
    # Train SERVER model (Image → Text)
    print("\n🔧 Training SERVER model (Image → Text)...")
    server_callbacks = [
        ReduceLROnPlateau(monitor='accuracy', factor=0.7, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True, verbose=1)
    ]
    
    server_history = server_model.fit(
        train_images, train_vectors,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=server_callbacks,
        verbose=1
    )
    
    # Test the integrated communication
    print("\n🧪 Testing integrated communication...")
    test_results, avg_vector_similarity, avg_feature_accuracy = test_integrated_communication(
        client_model, server_model, test_vectors, test_images, 
        [text_labels[split_idx + i] for i in range(len(test_vectors))],
        vocabularies, available_columns, num_tests=10
    )
    
    # Create visualization
    print("\n📊 Creating comprehensive visualization...")
    plot_filename = create_training_visualization(client_history, server_history, test_results)
    
    # Save results
    json_filename = save_integrated_results(df, available_columns, vocabularies, test_results, avg_vector_similarity, avg_feature_accuracy)
    
    print(f"\n✅ INTEGRATED TRAINING COMPLETE!")
    print(f"   CLIENT Loss: {client_history.history['loss'][-1]:.6f}")
    print(f"   SERVER Accuracy: {server_history.history['accuracy'][-1]:.6f}")
    print(f"   Communication Similarity: {avg_vector_similarity:.4f}")
    print(f"   Feature Accuracy: {avg_feature_accuracy:.4f}")
    print(f"   📁 Visualization: {plot_filename}")
    print(f"   📁 Results: {json_filename}")

if __name__ == "__main__":
    main()
