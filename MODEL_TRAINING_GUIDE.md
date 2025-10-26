# 🧠 Model Training Guide

## Complete Guide to Training Your Resume Screening Classifier

---

## 📋 **Overview**

The system uses **two approaches** for resume screening:

1. **Similarity-Based** (No Training Needed) ✅
   - Uses AWS Bedrock embeddings + TF-IDF
   - Works immediately without training
   - Good baseline performance

2. **Binary Classifier** (Requires Training) 🎯
   - PyTorch neural network
   - Learns from your labeled data
   - Better accuracy with enough data
   - **This guide covers training this classifier**

---

## 🎯 **Step 1: Collect Training Data**

### What You Need:

**Minimum**: 1,000 labeled examples (500 SHORTLIST, 500 REJECT)
**Recommended**: 10,000+ labeled examples for production

### Data Collection Methods:

#### **Method A: Use Existing Screening Results** (Easiest)

After using the system for a while, export your screening results:

```bash
# Export screening results from database
python scripts/export_training_data.py \
  --output training_data/labeled_resumes.csv \
  --min-confidence 0.8
```

This exports resumes where human reviewers confirmed the decision.

#### **Method B: Manual Labeling**

Create a CSV file: `training_data/labeled_resumes.csv`

```csv
resume_text,job_description,label,candidate_name,skills,experience_years
"John Doe, Senior Python Developer with 8 years experience...","Looking for Python developer with FastAPI...","1","John Doe","python,fastapi,aws,docker",8
"Fresh graduate with no experience...","Looking for Senior Python Developer...","0","Jane Smith","python",0
```

**Label Guide:**
- `1` = SHORTLIST (good fit)
- `0` = REJECT (not a good fit)

#### **Method C: Use Your Uploaded Resumes**

Export all uploaded resumes and label them:

```bash
python scripts/export_resumes_for_labeling.py \
  --output training_data/unlabeled_resumes.csv

# Then manually add 'label' column (0 or 1)
```

---

## 🔄 **Step 2: Prepare Training Dataset**

### Install Training Dependencies:

```bash
cd /Applications/MAMP/htdocs/resume-screening
pip install torch torchvision scikit-learn pandas tqdm
```

### Create Data Preparation Script:

Save as: `scripts/prepare_training_data.py`

```python
#!/usr/bin/env python3
"""
Prepare training data for classifier
Generates embeddings and creates train/val/test splits
"""

import asyncio
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ml.embeddings.bedrock_embedder import get_embedder
from backend.core.config import settings


async def prepare_training_data(
    input_csv: str,
    output_path: str,
    test_size: float = 0.15,
    val_size: float = 0.15
):
    """
    Prepare training data from CSV
    
    Args:
        input_csv: Path to labeled CSV file
        output_path: Path to save processed data
        test_size: Fraction for test set
        val_size: Fraction for validation set
    """
    print(f"📊 Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Total samples: {len(df)}")
    print(f"Positive (SHORTLIST): {(df['label'] == 1).sum()}")
    print(f"Negative (REJECT): {(df['label'] == 0).sum()}")
    
    # Initialize embedder
    print("\n🔌 Connecting to AWS Bedrock...")
    embedder = get_embedder()
    
    # Generate embeddings
    print("\n🧮 Generating embeddings...")
    resume_embeddings = []
    job_embeddings = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        
        # Generate embeddings
        resume_emb = await embedder.embed_text(row['resume_text'])
        job_emb = await embedder.embed_text(row['job_description'])
        
        resume_embeddings.append(resume_emb)
        job_embeddings.append(job_emb)
    
    # Convert to numpy arrays
    resume_embeddings = np.array(resume_embeddings)
    job_embeddings = np.array(job_embeddings)
    labels = df['label'].values
    
    print(f"\n✅ Generated embeddings:")
    print(f"  Resume embeddings: {resume_embeddings.shape}")
    print(f"  Job embeddings: {job_embeddings.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Split data
    print("\n✂️  Splitting data...")
    
    # First split: train+val vs test
    resume_train_val, resume_test, job_train_val, job_test, y_train_val, y_test = train_test_split(
        resume_embeddings, job_embeddings, labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    resume_train, resume_val, job_train, job_val, y_train, y_val = train_test_split(
        resume_train_val, job_train_val, y_train_val,
        test_size=val_size_adjusted,
        random_state=42,
        stratify=y_train_val
    )
    
    print(f"  Train: {len(y_train)} samples")
    print(f"  Val: {len(y_val)} samples")
    print(f"  Test: {len(y_test)} samples")
    
    # Save as PyTorch tensors
    print(f"\n💾 Saving to {output_path}...")
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
        },
        'metadata': {
            'total_samples': len(df),
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'test_samples': len(y_test),
            'embedding_dim': resume_embeddings.shape[1],
            'positive_samples': int((labels == 1).sum()),
            'negative_samples': int((labels == 0).sum())
        }
    }, output_path)
    
    print("\n✅ Data preparation complete!")
    print(f"📁 Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.15, help='Validation set size')
    
    args = parser.parse_args()
    
    # Run async function
    asyncio.run(prepare_training_data(
        args.input,
        args.output,
        args.test_size,
        args.val_size
    ))
```

### Run Data Preparation:

```bash
chmod +x scripts/prepare_training_data.py

python scripts/prepare_training_data.py \
  --input training_data/labeled_resumes.csv \
  --output training_data/processed_training.pt
```

---

## 🏋️ **Step 3: Train the Model Locally**

### Option A: Using Training Script

The training script is already in your project: `backend/ml/classifier/trainer.py`

```bash
python -m backend.ml.classifier.trainer \
  --data training_data/processed_training.pt \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output models/classifier_v1.pt \
  --mlflow-tracking
```

**Training Configuration:**
- **Epochs**: 50 (adjust based on convergence)
- **Batch Size**: 32 (reduce if GPU memory issues)
- **Learning Rate**: 0.001 (with scheduler)
- **Early Stopping**: Patience 5 epochs
- **Validation**: After each epoch

**Expected Output:**
```
Epoch 1/50: train_loss=0.693, val_loss=0.685, val_acc=0.532
Epoch 2/50: train_loss=0.612, val_loss=0.598, val_acc=0.687
...
Epoch 23/50: train_loss=0.234, val_loss=0.287, val_acc=0.892
✅ Best model achieved val_acc=0.892 at epoch 23
📁 Model saved to: models/classifier_v1.pt
```

### Option B: Using Jupyter Notebook

Already created: `notebooks/02_model_training.ipynb`

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

---

## ☁️ **Step 4: Train on AWS SageMaker** (Production Scale)

### Why Use SageMaker?

- **Faster**: GPU instances (P3, P4)
- **Scalable**: Train on large datasets
- **Managed**: Automatic checkpointing, monitoring
- **Cost-effective**: Pay only for training time

### Training on SageMaker:

#### **A. Upload Data to S3:**

```bash
# Upload processed training data
aws s3 cp training_data/processed_training.pt \
  s3://octa-resume-screening-data/data/training/processed_training.pt
```

#### **B. Run SageMaker Training Job:**

Save as: `scripts/train_on_sagemaker.py`

```python
#!/usr/bin/env python3
"""
Launch SageMaker training job
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole'  # Update this

# Training job configuration
timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
job_name = f'resume-classifier-{timestamp}'

# Create PyTorch estimator
estimator = PyTorch(
    entry_point='training_script.py',
    source_dir='backend/sagemaker',
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU instance
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'epochs': 50,
        'batch-size': 64,  # Larger on GPU
        'learning-rate': 0.001,
        'hidden-dims': '512,256,128',
        'dropout': 0.3
    },
    output_path=f's3://octa-resume-screening-data/models/training-jobs/{job_name}',
    base_job_name='resume-classifier'
)

# Start training
print(f"🚀 Starting SageMaker training job: {job_name}")

estimator.fit({
    'training': f's3://octa-resume-screening-data/data/training/processed_training.pt'
})

print(f"✅ Training complete!")
print(f"📁 Model saved to: {estimator.model_data}")
```

Run it:

```bash
python scripts/train_on_sagemaker.py
```

**Monitor Training:**
```bash
# In AWS Console -> SageMaker -> Training Jobs
# Or use CloudWatch logs
```

---

## 📦 **Step 5: Deploy the Trained Model**

### Upload Model to S3:

```bash
# Upload local model
aws s3 cp models/classifier_v1.pt \
  s3://octa-resume-screening-data/models/classifier/v1.0.0/model.pt
```

### Register Model in Database:

```bash
python scripts/register_model.py \
  --model-path s3://octa-resume-screening-data/models/classifier/v1.0.0/model.pt \
  --version v1.0.0 \
  --model-type binary_classifier \
  --metrics '{"accuracy": 0.892, "precision": 0.885, "recall": 0.901, "f1": 0.893}' \
  --activate
```

This will:
1. Save model metadata to PostgreSQL
2. Set it as the active model
3. Make it available for inference

### Test the Model:

```bash
# Upload a test resume through the UI
# It should now use the trained classifier!
```

---

## 🔍 **Step 6: Evaluate Model Performance**

### Run Evaluation:

```bash
python -m backend.ml.classifier.evaluate \
  --model models/classifier_v1.pt \
  --test-data training_data/processed_training.pt
```

**Output:**
```
📊 Model Evaluation Results:

Confusion Matrix:
              Predicted
              0       1
Actual  0   [142     8]
        1   [ 12   138]

Metrics:
- Accuracy:  0.933
- Precision: 0.945 (of predicted SHORTLIST, 94.5% were correct)
- Recall:    0.920 (found 92% of good candidates)
- F1-Score:  0.932

ROC AUC: 0.967

✅ Model performs well!
```

### **Target Metrics:**
- **Accuracy**: ≥ 85%
- **Precision**: ≥ 82% (avoid false positives)
- **Recall**: ≥ 88% (don't miss good candidates)
- **F1-Score**: ≥ 85%

---

## 🔄 **Step 7: Continuous Improvement**

### A. Collect More Data

As you use the system:
1. Humans review AI decisions
2. Corrections are logged
3. Export new training data
4. Retrain with larger dataset

### B. Automated Retraining

Set up periodic retraining:

```bash
# Cron job (weekly)
0 0 * * 0 /path/to/scripts/auto_retrain.sh
```

`scripts/auto_retrain.sh`:
```bash
#!/bin/bash

# Export new data
python scripts/export_training_data.py \
  --output training_data/labeled_resumes_$(date +%Y%m%d).csv

# Prepare data
python scripts/prepare_training_data.py \
  --input training_data/labeled_resumes_$(date +%Y%m%d).csv \
  --output training_data/processed_training_$(date +%Y%m%d).pt

# Train
python scripts/train_on_sagemaker.py \
  --data training_data/processed_training_$(date +%Y%m%d).pt \
  --version v1.$(date +%Y%m%d)

# If metrics improved, deploy
python scripts/deploy_if_better.py
```

---

## 💡 **Quick Start (Local Training)**

**Minimal example to train your first model:**

```bash
# 1. Prepare sample data (you need at least 100 labeled examples)
cat > training_data/sample_data.csv << EOF
resume_text,job_description,label
"Senior Python Dev with 8 years exp...","Looking for Python developer...",1
"Junior dev fresh grad...","Looking for Senior developer...",0
EOF

# 2. Process data
python scripts/prepare_training_data.py \
  --input training_data/sample_data.csv \
  --output training_data/sample_processed.pt

# 3. Train model
python -m backend.ml.classifier.trainer \
  --data training_data/sample_processed.pt \
  --epochs 20 \
  --output models/my_first_model.pt

# 4. Deploy
aws s3 cp models/my_first_model.pt \
  s3://octa-resume-screening-data/models/classifier/v1.0.0/model.pt

python scripts/register_model.py \
  --model-path s3://octa-resume-screening-data/models/classifier/v1.0.0/model.pt \
  --version v1.0.0 \
  --activate
```

---

## 🎯 **Summary**

**Without Trained Model** (Current State):
- ✅ Similarity-based screening works
- ✅ Good baseline performance
- ⚠️ No personalized learning

**With Trained Model** (After Training):
- ✅ All of the above
- ✅ Learns from your specific needs
- ✅ Better accuracy (85-95%)
- ✅ Continuous improvement

**Training is optional but recommended for production!**

---

## 📞 **Need Help?**

- Check logs: `tail -f logs/training.log`
- Review metrics in MLflow: `mlflow ui`
- Debug issues: Add `--debug` flag to scripts

**The system works great without training, but gets even better with it!** 🚀



