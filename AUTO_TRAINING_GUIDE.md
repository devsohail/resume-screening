# 🤖 Automatic Training System Guide

## Overview

The Resume Screening System includes a **fully automated training pipeline** that:
- ✅ Collects uploaded resumes automatically
- ✅ Allows human feedback/correction on AI decisions
- ✅ Exports training data when enough samples are available
- ✅ Automatically trains and deploys new models
- ✅ Can run on a schedule (cron/EventBridge)

## 🎯 Why Automatic Training?

**You asked**: "Why do I have to train manually? It should auto put uploaded resumes in the model to train."

**The Answer**: We CAN'T train immediately because we need **confirmed labels** (human feedback). But once you review decisions, training happens automatically!

### The Training Loop

```
1. User uploads resume → AI makes prediction
2. Human reviews and confirms/corrects decision
3. System collects reviewed samples
4. When threshold reached (100 samples) → Auto-train
5. New model automatically deployed
6. Repeat!
```

---

## 📋 How It Works

### Step 1: Upload Resume (Automatic)

```bash
# Via API or Frontend
POST /api/v1/screening/upload-resume
{
  "file": <resume.pdf>,
  "job_id": "job-123",
  "candidate_name": "John Doe"
}
```

**What happens:**
- Resume is processed and scored
- Decision is made (SHORTLIST/REJECT)
- Result is saved with `human_reviewed = false`

### Step 2: Human Reviews Decision (Manual Review Required)

```bash
# Review and confirm/correct AI decision
POST /api/v1/feedback/submit
{
  "screening_result_id": "result-123",
  "human_decision": "shortlist",  # or "reject"
  "notes": "Strong Python skills match requirements"
}
```

**What happens:**
- Result is marked as `human_reviewed = true`
- If human agrees with AI → reinforces correct behavior
- If human corrects AI → marked as `needs_retraining = true`
- Sample is now available for training!

### Step 3: Check Training Status (Automatic/Manual)

```bash
# Check if enough data for training
GET /api/v1/feedback/training-status

# Response:
{
  "total_samples": 125,
  "positive_samples": 60,
  "negative_samples": 65,
  "ready_for_training": true,
  "samples_needed": 0,
  "threshold": 100
}
```

### Step 4: Trigger Training (Automatic)

#### Option A: Manual Trigger via API

```bash
POST /api/v1/feedback/trigger-training

# Response:
{
  "success": true,
  "message": "Training started!",
  "job_id": 12345,
  "log_file": "logs/training_20251018_1400.log"
}
```

#### Option B: Scheduled Automatic Training (Recommended)

```bash
# Add to crontab (runs daily at 2 AM)
0 2 * * * cd /path/to/resume-screening && python scripts/scheduled_training.py >> logs/cron.log 2>&1
```

---

## 🚀 Quick Setup

### 1. Enable Automatic Training

```bash
cd /Applications/MAMP/htdocs/resume-screening

# Check current training status
python scripts/scheduled_training.py --dry-run

# Output:
# 📊 Training Data Status:
#    Total reviewed: 1
#    Positive samples: 0
#    Negative samples: 1
#    Needs retraining: 0
#    Threshold: 100
# ⏳ NOT READY TO TRAIN
#    Need 99 more reviewed samples
```

### 2. Review Some Resumes

Use the frontend or API to review screening results:

```python
import requests

# Get screening results
results = requests.get("http://localhost:8000/api/v1/screening/results/job-123").json()

# Submit feedback for each result
for result in results['results']:
    requests.post("http://localhost:8000/api/v1/feedback/submit", json={
        "screening_result_id": result['id'],
        "human_decision": "shortlist",  # Your decision
        "notes": "Good candidate"
    })
```

### 3. Check Training Status

```bash
# Check if ready
python scripts/scheduled_training.py --dry-run

# When you have 100+ reviewed samples:
# ✅ READY TO TRAIN (dry run - not training)
#    Would train with 125 samples
```

### 4. Train Model

```bash
# Option 1: Train manually
python scripts/scheduled_training.py

# Option 2: Force training with less data (for testing)
python scripts/scheduled_training.py --force --min-samples 10
```

---

## 🔄 Setting Up Scheduled Training

### Local (Cron)

```bash
# Edit crontab
crontab -e

# Add this line (runs daily at 2 AM)
0 2 * * * cd /Applications/MAMP/htdocs/resume-screening && /usr/bin/python3 scripts/scheduled_training.py >> logs/training_cron.log 2>&1

# Or check every hour (only trains if enough data)
0 * * * * cd /Applications/MAMP/htdocs/resume-screening && /usr/bin/python3 scripts/scheduled_training.py >> logs/training_cron.log 2>&1
```

### AWS EventBridge (Production)

```hcl
# terraform/eventbridge.tf

resource "aws_cloudwatch_event_rule" "scheduled_training" {
  name                = "resume-screening-training"
  description         = "Trigger model training daily"
  schedule_expression = "cron(0 2 * * ? *)"  # 2 AM UTC daily
}

resource "aws_cloudwatch_event_target" "training_lambda" {
  rule      = aws_cloudwatch_event_rule.scheduled_training.name
  target_id = "TrainingLambda"
  arn       = aws_lambda_function.training.arn
}

resource "aws_lambda_function" "training" {
  filename      = "training_lambda.zip"
  function_name = "resume-screening-training"
  role          = aws_iam_role.lambda_training.arn
  handler       = "scheduled_training.lambda_handler"
  runtime       = "python3.9"
  timeout       = 900  # 15 minutes
  memory_size   = 2048
  
  environment {
    variables = {
      MIN_SAMPLES = "100"
    }
  }
}
```

---

## 📊 Monitoring Training

### View Training Logs

```bash
# View latest training log
tail -f logs/training_*.log

# View cron log
tail -f logs/training_cron.log

# Check MLflow for metrics
python -m mlflow ui --port 5000
# Open: http://localhost:5000
```

### Check Training History

```sql
-- Query training history
SELECT 
    version,
    framework,
    accuracy,
    created_at
FROM model_metadata
ORDER BY created_at DESC
LIMIT 10;
```

---

## 🎯 Best Practices

### 1. **Review Regularly**
- Review at least 10-20 screening results per week
- Focus on borderline cases (scores 40-60)
- Correct obvious mistakes

### 2. **Balanced Training Data**
- Aim for 40-60% positive, 40-60% negative
- Don't review only SHORTLIST or only REJECT
- Review diverse job types and skill sets

### 3. **Quality Over Quantity**
- 100 high-quality reviews > 1000 auto-approved
- Take time to read resumes before confirming
- Add notes explaining your decisions

### 4. **Monitor Model Performance**
- Check MLflow metrics after each training
- Compare new model vs old model on test set
- Only deploy if new model is better

### 5. **Gradual Deployment**
- Start with A/B testing (50% old, 50% new)
- Monitor for 1 week before full rollout
- Keep old model as backup

---

## 🔧 Troubleshooting

### Not Enough Data

```bash
# Check status
python scripts/scheduled_training.py --dry-run

# Need more reviews?
# - Review pending screening results in frontend
# - Lower threshold temporarily for testing:
python scripts/scheduled_training.py --min-samples 20 --force
```

### Training Fails

```bash
# Check logs
tail -f logs/training_*.log

# Common issues:
# 1. AWS Bedrock not configured
#    → Set AWS credentials in .env
# 2. Not enough memory
#    → Reduce batch size in auto_train.py
# 3. Imbalanced data
#    → Review more of the minority class
```

### Model Not Deploying

```bash
# Check S3 upload
aws s3 ls s3://octa-resume-screening-data/models/classifier/

# Register model manually
python scripts/register_model.py \
  --model-path s3://bucket/models/classifier/auto_20251018/model.pt \
  --version auto_20251018 \
  --activate
```

---

## 📈 Advanced: Custom Training Pipeline

### Export Custom Training Data

```python
# scripts/export_training_data.py
python scripts/export_training_data.py \
  --output custom_data.csv \
  --min-confidence 0.8 \
  --days 7 \
  --status completed
```

### Train with Custom Parameters

```python
# scripts/auto_train.py
python scripts/auto_train.py \
  --data custom_data.csv \
  --min-samples 50 \
  --force
```

### Manual Training (Full Control)

```python
import asyncio
from backend.ml.embeddings.bedrock_embedder import get_embedder
from backend.ml.classifier.trainer import train_model

# 1. Load your data
import pandas as pd
df = pd.read_csv('training_data.csv')

# 2. Generate embeddings
embedder = await get_embedder()
resume_embeddings = [await embedder.embed_text(text) for text in df['resume_text']]
job_embeddings = [await embedder.embed_text(text) for text in df['job_description']]

# 3. Train
train_model(
    resume_embeddings=resume_embeddings,
    job_embeddings=job_embeddings,
    labels=df['label'],
    epochs=50,
    batch_size=16,
    learning_rate=0.0001
)
```

---

## 🎉 Summary

### What You Get

1. **Zero Manual Training** (after initial reviews)
2. **Continuous Learning** from every upload
3. **Human-in-the-Loop** for quality control
4. **Automatic Deployment** when ready
5. **Full Audit Trail** of all decisions

### The Complete Flow

```
📤 Upload Resume
   ↓
🤖 AI Screens
   ↓
👤 Human Reviews (You)
   ↓
💾 Data Collected
   ↓
🔄 [100 samples reached]
   ↓
🏋️  Auto Train (Scheduled)
   ↓
📊 MLflow Metrics
   ↓
✅ Deploy New Model
   ↓
🎯 Better Predictions
   ↓
[Repeat Forever!]
```

### Your Action Items

1. ✅ Review 100 screening results (API or frontend)
2. ✅ Set up cron job for scheduled training
3. ✅ Monitor MLflow for metrics
4. ✅ Deploy new models when ready
5. ✅ Keep reviewing to improve!

---

## 📚 Related Guides

- **[MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md)** - Detailed training guide
- **[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md)** - Complete system guide
- **[QUICKSTART.md](QUICKSTART.md)** - Getting started

---

**Questions?** Check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open an issue!

