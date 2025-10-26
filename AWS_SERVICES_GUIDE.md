# 🔧 AWS Services & Pinecone - Complete Guide

## Overview

Your Resume Screening System uses several cloud services, each with a **specific purpose**. Here's what each one does and why you need it.

---

## 🧠 AWS Bedrock (Embeddings)

### What It Does
**AWS Bedrock provides AI foundation models** - specifically, we use **Titan Text Embeddings** to convert text into numerical vectors (embeddings).

### Why You Need It
- **Semantic Understanding**: Converts resumes and job descriptions into vectors that capture meaning
- **Similarity Search**: Enables finding candidates with similar skills/experience
- **Pre-trained**: No training needed - works out of the box
- **Enterprise-grade**: Secure, scalable, managed by AWS

### How It Works

```python
# When a resume is uploaded:
resume_text = "5 years Python, AWS, Machine Learning..."
job_description = "Looking for Python developer with ML experience..."

# Bedrock converts to vectors (1536 dimensions)
resume_vector = bedrock.embed(resume_text)  # [0.234, -0.567, 0.891, ...]
job_vector = bedrock.embed(job_description)  # [0.198, -0.543, 0.876, ...]

# Now we can measure similarity
similarity = cosine_similarity(resume_vector, job_vector)  # 0.87 (87% match!)
```

### Use Cases in Your System

1. **Resume Upload** → Generate embedding for semantic search
2. **Job Creation** → Generate embedding for matching
3. **Candidate Search** → Find similar candidates using vector similarity
4. **Model Training** → Create training data with embeddings

### Configuration

```bash
# .env
AWS_BEDROCK_REGION=us-east-1
AWS_BEDROCK_MODEL_ID=amazon.titan-embed-text-v1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
```

### Cost
- **$0.0001 per 1,000 tokens** (~$0.10 per 1,000 resumes)
- Very cheap for embeddings!

### When It Runs
- ✅ **Every time** a resume is uploaded
- ✅ **Every time** a job is created
- ✅ **During training** to generate embeddings for model
- ❌ **NOT needed** for inference (once model is trained)

---

## 🤖 AWS SageMaker (Model Training & Deployment)

### What It Does
**SageMaker is AWS's machine learning platform** for training and deploying custom ML models at scale.

### Why You Need It
- **Production ML**: Deploy your trained classifier to a scalable endpoint
- **GPU Training**: Train large models faster with GPU instances
- **Auto-scaling**: Handle variable load automatically
- **Model Versioning**: Track and manage multiple model versions
- **A/B Testing**: Test new models against old ones

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     SageMaker Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. TRAINING JOB                                            │
│     ┌──────────────────────────────────────────┐            │
│     │ • Load data from S3                      │            │
│     │ • Train PyTorch classifier               │            │
│     │ • Validate on test set                   │            │
│     │ • Save model to S3                       │            │
│     └──────────────────────────────────────────┘            │
│                          ↓                                   │
│  2. MODEL REGISTRY                                          │
│     ┌──────────────────────────────────────────┐            │
│     │ • Version: v1.2.3                        │            │
│     │ • Accuracy: 92.5%                        │            │
│     │ • Status: Approved                       │            │
│     └──────────────────────────────────────────┘            │
│                          ↓                                   │
│  3. ENDPOINT DEPLOYMENT                                     │
│     ┌──────────────────────────────────────────┐            │
│     │ • Deploy to ml.t2.medium instance        │            │
│     │ • Auto-scaling: 1-5 instances            │            │
│     │ • Health checks every 30s                │            │
│     │ • API: POST /invocations                 │            │
│     └──────────────────────────────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Use Cases in Your System

#### 1. **Training Models**
```python
# Run training job on SageMaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='backend/sagemaker/training_script.py',
    role='SageMakerRole',
    instance_type='ml.p3.2xlarge',  # GPU instance
    instance_count=1,
    framework_version='2.0'
)

estimator.fit({
    'training': 's3://bucket/training_data/',
    'validation': 's3://bucket/validation_data/'
})
```

#### 2. **Deploying Models**
```python
# Deploy trained model to endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='resume-screening-v1'
)

# Make predictions
result = predictor.predict({
    'resume_embedding': resume_vector,
    'job_embedding': job_vector
})
# → {'prediction': 'shortlist', 'confidence': 0.92}
```

### Configuration

```bash
# .env
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerRole
SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::123456789012:role/SageMakerExecution
SAGEMAKER_ENDPOINT_NAME=resume-screening-classifier
```

### Cost
- **Training**: $3.06/hour for GPU (ml.p3.2xlarge)
- **Inference**: $0.065/hour for CPU (ml.t2.medium)
- **Example**: $10-20 per training run, $50/month for always-on endpoint

### When It Runs
- 🏋️ **Training**: When you run `python scripts/auto_train.py` (manual/scheduled)
- 🚀 **Inference**: Every time a resume is screened (if deployed)
- 📊 **Monitoring**: Continuous health checks and metrics

### Two Deployment Options

#### Option A: **Lambda + API Gateway** (Serverless - Recommended for Start)
- ✅ Pay per request (cheaper for low volume)
- ✅ Auto-scales to zero (no cost when idle)
- ✅ Simpler to deploy
- ❌ Cold start latency (1-2s)
- ❌ 15 min timeout limit

#### Option B: **SageMaker Endpoint** (Enterprise - Recommended for Scale)
- ✅ Always warm (fast predictions)
- ✅ Advanced features (A/B testing, auto-scaling)
- ✅ No timeout limits
- ❌ Always running (costs $$$ even when idle)
- ❌ More complex setup

### Current Setup
Your system is **configured for SageMaker** but can work **without it** initially:
- ✅ **Development**: Uses local model file (`models/classifier.pt`)
- ✅ **Production**: Can deploy to SageMaker when ready
- ✅ **Hybrid**: Falls back to similarity-only if SageMaker unavailable

---

## 📍 Pinecone (Vector Database)

### What It Does
**Pinecone is a managed vector database** for storing and searching high-dimensional embeddings at scale.

### Why You Need It
- **Fast Vector Search**: Find similar resumes in milliseconds
- **Scalability**: Store millions of resume embeddings
- **Metadata Filtering**: Search with filters (e.g., "Python + 5+ years")
- **Real-time Updates**: Add new resumes instantly
- **Managed Service**: No infrastructure to maintain

### How It Works

```python
# 1. Store resume embeddings
pinecone.upsert(
    id="resume-123",
    vector=resume_embedding,  # 1536 dimensions from Bedrock
    metadata={
        "candidate_name": "John Doe",
        "skills": ["Python", "AWS", "ML"],
        "experience_years": 5
    }
)

# 2. Search for similar candidates
results = pinecone.query(
    vector=job_embedding,  # Vector for job description
    top_k=10,  # Top 10 matches
    filter={"experience_years": {"$gte": 3}}  # 3+ years exp
)

# Results:
# [
#   {'id': 'resume-123', 'score': 0.92, 'metadata': {...}},
#   {'id': 'resume-456', 'score': 0.89, 'metadata': {...}},
#   ...
# ]
```

### Use Cases in Your System

#### 1. **Semantic Resume Search**
```bash
# Find candidates similar to a job description
GET /api/v1/screening/search?job_id=123&top_k=20

# Returns: Top 20 most similar candidates from all uploaded resumes
```

#### 2. **Duplicate Detection**
```python
# Check if candidate already applied
similar = pinecone.query(resume_embedding, top_k=1)
if similar[0]['score'] > 0.98:
    print("Duplicate resume detected!")
```

#### 3. **Skills-based Search**
```python
# Find Python developers with ML experience
results = pinecone.query(
    vector=skill_embedding,
    filter={
        "skills": {"$in": ["Python", "Machine Learning"]},
        "experience_years": {"$gte": 3}
    }
)
```

### Configuration

```bash
# .env
PINECONE_API_KEY=your-api-key-here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=resume-embeddings
```

### Cost
- **Free Tier**: 1M vectors, 1 index (good for testing!)
- **Paid**: $0.096/hour per pod (~$70/month)
- **Serverless**: Pay per operation (cheaper for low volume)

### When It Runs
- ✅ **Every resume upload**: Stores embedding for future search
- ✅ **Search queries**: Finds similar candidates
- ✅ **Batch operations**: Update multiple resumes at once
- ⚠️ **Optional**: System works without Pinecone (slower search)

### Current Setup
Your system has **graceful degradation**:
- ✅ **If Pinecone configured**: Full vector search enabled
- ✅ **If Pinecone missing**: Falls back to PostgreSQL + text search
- ✅ **No crashes**: System works either way

---

## 📊 How They Work Together

### Complete Flow: Resume Upload → Screening → Training

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RESUME SCREENING FLOW                        │
└─────────────────────────────────────────────────────────────────────┘

1. USER UPLOADS RESUME
   ↓
   📄 resume.pdf → S3 Bucket (storage)
   
2. EXTRACT TEXT
   ↓
   📝 "5 years Python, AWS, Machine Learning..."
   
3. GENERATE EMBEDDING (AWS Bedrock)
   ↓
   🧠 [0.234, -0.567, 0.891, ...] (1536 dimensions)
   
4. STORE IN VECTOR DB (Pinecone)
   ↓
   📍 Indexed for fast similarity search
   
5. CALCULATE SIMILARITY
   ↓
   📊 Compare resume_vector vs job_vector → 87% match
   
6. PREDICT WITH CLASSIFIER (SageMaker - Optional)
   ↓
   🤖 ML Model: "SHORTLIST" (92% confidence)
   
7. COMBINE SCORES (Hybrid)
   ↓
   🎯 Final: 89.5 = (0.6 × 87) + (0.4 × 92)
   
8. SAVE RESULT (PostgreSQL)
   ↓
   💾 Decision: SHORTLIST, Score: 89.5
   
9. HUMAN REVIEWS
   ↓
   👤 Confirm or correct decision
   
10. COLLECT TRAINING DATA
    ↓
    📊 100+ reviewed samples → Export CSV
    
11. TRAIN NEW MODEL (SageMaker)
    ↓
    🏋️ GPU training on ml.p3.2xlarge
    
12. DEPLOY UPDATED MODEL
    ↓
    🚀 New endpoint with better accuracy!
```

---

## 🎯 Service Comparison Table

| Service | Purpose | Required? | Cost | Alternative |
|---------|---------|-----------|------|-------------|
| **AWS Bedrock** | Text → Embeddings | ✅ Yes (for semantic) | $0.10/1K resumes | sentence-transformers (local) |
| **SageMaker** | ML Training/Deploy | ⚠️ Optional | $50-100/month | Local PyTorch + Lambda |
| **Pinecone** | Vector Search | ⚠️ Optional | Free tier OK | PostgreSQL pgvector |
| **S3** | File Storage | ✅ Yes | $0.023/GB | Local filesystem (dev) |
| **RDS PostgreSQL** | Main Database | ✅ Yes | $15-50/month | Local PostgreSQL |
| **Lambda** | Serverless API | ⚠️ Optional | Pay per use | Always-on server |

---

## 🚀 Deployment Strategies

### Strategy 1: **Minimal Setup** (Start Here)
```
✅ AWS Bedrock (embeddings only)
✅ S3 (resume storage)
✅ RDS PostgreSQL (data)
❌ SageMaker (use local model)
❌ Pinecone (use PostgreSQL search)
❌ Lambda (use EC2/local server)

Cost: ~$30/month
Best for: MVP, testing, low volume
```

### Strategy 2: **Recommended Production**
```
✅ AWS Bedrock (embeddings)
✅ S3 (storage)
✅ RDS PostgreSQL (data)
✅ SageMaker Endpoint (classifier)
✅ Pinecone Serverless (vector search)
✅ Lambda + API Gateway (API)

Cost: ~$150/month
Best for: 100-1000 resumes/day
```

### Strategy 3: **Enterprise Scale**
```
✅ AWS Bedrock (embeddings)
✅ S3 (storage)
✅ RDS PostgreSQL (data)
✅ SageMaker Multi-instance (HA)
✅ Pinecone Enterprise (multi-pod)
✅ ECS Fargate (containers)
✅ CloudFront (CDN)
✅ WAF (security)

Cost: $500-2000/month
Best for: 10,000+ resumes/day
```

---

## 🔧 Local Development

You can develop **locally** without most cloud services:

```bash
# .env for local development
AWS_BEDROCK_REGION=us-east-1  # Required for embeddings
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# Optional (system works without these)
PINECONE_API_KEY=  # Leave empty - uses PostgreSQL
SAGEMAKER_ENDPOINT_NAME=  # Leave empty - uses local model

# Local services
DATABASE_URL=postgresql://user:pass@localhost:5432/resume_screening
S3_BUCKET_RESUMES=octa-resume-screening-data  # Still use S3
```

**What works locally:**
- ✅ Resume upload (to S3)
- ✅ Embeddings (via Bedrock API)
- ✅ Similarity scoring
- ✅ Local classifier model
- ✅ PostgreSQL storage
- ✅ Full API

**What doesn't work locally:**
- ❌ SageMaker training (use local training instead)
- ❌ SageMaker inference (use local model)
- ❌ Pinecone search (PostgreSQL fallback works)

---

## 📈 When to Add Each Service

### Start with Bedrock + S3 + RDS
```
Week 1-2: Get basic system working
- Upload resumes
- Generate embeddings
- Similarity scoring
- PostgreSQL storage
```

### Add SageMaker for Better Predictions
```
Week 3-4: Once you have 100+ reviewed resumes
- Train custom classifier
- Deploy to SageMaker endpoint
- A/B test against similarity-only
```

### Add Pinecone for Fast Search
```
Month 2: When search becomes slow (1000+ resumes)
- Migrate embeddings to Pinecone
- Enable vector search
- Add advanced filtering
```

---

## 🎓 Summary

**AWS Bedrock**
- 🎯 Purpose: Convert text → vectors (embeddings)
- 💰 Cost: Very cheap ($0.10/1K)
- ⚡ When: Every upload/search
- 🔧 Required: Yes (for semantic matching)

**AWS SageMaker**
- 🎯 Purpose: Train & deploy ML models
- 💰 Cost: Medium ($50-200/month)
- ⚡ When: Training (weekly/monthly), Inference (always)
- 🔧 Required: No (can use local model)

**Pinecone**
- 🎯 Purpose: Fast vector similarity search
- 💰 Cost: Free tier → $70/month
- ⚡ When: Search queries, duplicate detection
- 🔧 Required: No (PostgreSQL fallback)

**The Magic**: These services work together to create an intelligent system that learns from every resume and gets better over time!

---

## 📚 Next Steps

1. ✅ **Current**: Basic system with Bedrock + S3 + RDS
2. 📊 **Next**: Upload 100 resumes, review decisions
3. 🏋️ **Then**: Train first model with `python scripts/auto_train.py`
4. 🚀 **Finally**: Deploy to SageMaker when ready for production

---

**Questions?** Check:
- [AUTO_TRAINING_GUIDE.md](AUTO_TRAINING_GUIDE.md) - How training works
- [MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md) - Detailed training
- [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md) - Complete system overview



