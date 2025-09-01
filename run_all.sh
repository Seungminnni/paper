#!/bin/bash
"""
Batch script to run all medical data privacy research pipeline
의료 데이터 프라이버시 연구 파이프라인 배치 실행 스크립트
"""

echo "🚀 Starting Medical Data Privacy Research Pipeline"
echo "=================================================="

# Step 1: Pre-training
echo ""
echo "Step 1: Pre-training BERT Model"
echo "-------------------------------"
python pre_train.py

# Step 2: Fine-tuning
echo ""
echo "Step 2: Fine-tuning BERT Model"
echo "------------------------------"
python fine_tune.py

# Step 3: Server-side Smashed Data Generation
echo ""
echo "Step 3: Server-side Smashed Data Generation"
echo "-------------------------------------------"
python server_side.py

# Step 4: Client-side Smashed Data Generation
echo ""
echo "Step 4: Client-side Smashed Data Generation"
echo "-------------------------------------------"
python client_side.py

echo ""
echo "🎉 All steps completed successfully!"
echo "📁 Check the generated files:"
echo "   - Pre_train_epoch*_BERT_Based.pt"
echo "   - Fine_tuned_epoch*_BERT_Based.pt"
echo "   - Dictionary_smashed_data_layer2.csv"
echo "   - Client_smashed_data_layer2.csv"
