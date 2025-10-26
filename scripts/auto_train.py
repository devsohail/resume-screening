#!/usr/bin/env python3
"""
Automated model training pipeline
Automatically trains model when enough new data is available
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ml.embeddings.bedrock_embedder import get_embedder
from backend.ml.classifier.trainer import train_model
from backend.core.config import settings


async def auto_train_pipeline(
    input_csv: str,
    min_samples: int = 100,
    force: bool = False
):
    """
    Automated training pipeline
    
    Args:
        input_csv: Input CSV with training data
        min_samples: Minimum samples required to train
        force: Force training even if not enough samples
    """
    print("🤖 Automated Training Pipeline")
    print("=" * 50)
    
    # Step 1: Load data
    print(f"\n📊 Step 1: Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"   Total samples: {len(df)}")
    print(f"   Positive: {(df['label'] == 1).sum()}")
    print(f"   Negative: {(df['label'] == 0).sum()}")
    
    # Check if enough data
    if len(df) < min_samples and not force:
        print(f"\n⚠️  Not enough data to train!")
        print(f"   Need: {min_samples} samples")
        print(f"   Have: {len(df)} samples")
        print(f"   Missing: {min_samples - len(df)} samples")
        print(f"\n💡 Upload more resumes or use --force to train anyway")
        return False
    
    # Step 2: Generate embeddings
    print(f"\n🔌 Step 2: Connecting to AWS Bedrock...")
    try:
        embedder = get_embedder()
        print("   ✅ Connected!")
    except Exception as e:
        print(f"   ❌ Failed to connect to AWS Bedrock: {e}")
        print(f"\n💡 Training requires AWS Bedrock for embeddings")
        print(f"   Configure AWS credentials in .env file")
        return False
    
    print(f"\n🧮 Step 3: Generating embeddings...")
    resume_embeddings = []
    job_embeddings = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"   Processing {idx}/{len(df)}...")
        
        try:
            resume_emb = await embedder.embed_text(str(row['resume_text']))
            job_emb = await embedder.embed_text(str(row['job_description']))
            resume_embeddings.append(resume_emb)
            job_embeddings.append(job_emb)
        except Exception as e:
            print(f"   ⚠️  Skipping row {idx}: {e}")
            continue
    
    resume_embeddings = np.array(resume_embeddings)
    job_embeddings = np.array(job_embeddings)
    labels = df['label'].values[:len(resume_embeddings)]
    
    print(f"   ✅ Generated {len(resume_embeddings)} embeddings")
    
    # Step 4: Split data
    print(f"\n✂️  Step 4: Splitting data...")
    
    # Train: 70%, Val: 15%, Test: 15%
    resume_train_val, resume_test, job_train_val, job_test, y_train_val, y_test = train_test_split(
        resume_embeddings, job_embeddings, labels,
        test_size=0.15,
        random_state=42,
        stratify=labels
    )
    
    resume_train, resume_val, job_train, job_val, y_train, y_val = train_test_split(
        resume_train_val, job_train_val, y_train_val,
        test_size=0.176,  # 0.15 / (1 - 0.15) ≈ 0.176
        random_state=42,
        stratify=y_train_val
    )
    
    print(f"   Train: {len(y_train)} samples")
    print(f"   Val: {len(y_val)} samples")
    print(f"   Test: {len(y_test)} samples")
    
    # Step 5: Save processed data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    processed_path = f"training_data/processed_{timestamp}.pt"
    
    print(f"\n💾 Step 5: Saving processed data...")
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'train': {
            'resume_embeddings': torch.FloatTensor(resume_train),
            'job_embeddings': torch.FloatTensor(job_train),
            'labels': torch.FloatTensor(y_train)
        },
        'val': {
            'resume_embeddings': torch.FloatTensor(resume_val),
            'job_embeddings': torch.FloatTensor(job_val),
            'labels': torch.FloatTensor(y_val)
        },
        'test': {
            'resume_embeddings': torch.FloatTensor(resume_test),
            'job_embeddings': torch.FloatTensor(job_test),
            'labels': torch.FloatTensor(y_test)
        }
    }, processed_path)
    
    print(f"   ✅ Saved to: {processed_path}")
    
    # Step 6: Train model
    print(f"\n🏋️  Step 6: Training model...")
    model_path = f"models/auto_trained_{timestamp}.pt"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Train with reasonable defaults
        train_model(
            data_path=processed_path,
            output_path=model_path,
            epochs=30,
            batch_size=32,
            learning_rate=0.001,
            patience=5
        )
        
        print(f"\n✅ Training complete!")
        print(f"   Model saved to: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Deploy model
    print(f"\n🚀 Step 7: Deploying model...")
    print(f"\n📋 To deploy this model:")
    print(f"   1. Upload to S3:")
    print(f"      aws s3 cp {model_path} s3://octa-resume-screening-data/models/classifier/auto_{timestamp}/model.pt")
    print(f"\n   2. Register in database:")
    print(f"      python scripts/register_model.py \\")
    print(f"        --model-path s3://octa-resume-screening-data/models/classifier/auto_{timestamp}/model.pt \\")
    print(f"        --version auto_{timestamp} \\")
    print(f"        --activate")
    
    print(f"\n✅ PIPELINE COMPLETE!")
    print(f"\n🎯 Next steps:")
    print(f"   - Review model performance in MLflow")
    print(f"   - Deploy model to production if metrics are good")
    print(f"   - Continue collecting more data for next training run")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated model training')
    parser.add_argument('--data', required=True, help='Input CSV file with training data')
    parser.add_argument('--min-samples', type=int, default=100, help='Minimum samples to train')
    parser.add_argument('--force', action='store_true', help='Force training even without enough data')
    
    args = parser.parse_args()
    
    # Run async pipeline
    success = asyncio.run(auto_train_pipeline(
        args.data,
        args.min_samples,
        args.force
    ))
    
    sys.exit(0 if success else 1)

