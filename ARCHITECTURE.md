# System Architecture

## Overview

The Intelligent Resume Screening System is a production-grade MLOps application that combines semantic similarity, machine learning classification, and enterprise infrastructure to automate candidate evaluation.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND LAYER                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  React + TypeScript Dashboard                                     │  │
│  │  - Job Management  - Resume Upload  - Analytics  - Results View  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTPS/REST API
┌────────────────────────────────▼────────────────────────────────────────┐
│                          API LAYER (FastAPI)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │  Jobs API   │  │ Screening   │  │ Analytics   │  │   Health     │  │
│  │  Routes     │  │  Routes     │  │  Routes     │  │   Checks     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘  │
│         │                  │                │                           │
│         └──────────────────┼────────────────┘                           │
└────────────────────────────┼────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Hybrid Screening Engine                        │  │
│  │  ┌─────────────────────┐         ┌──────────────────────┐        │  │
│  │  │ Similarity Scorer   │         │ Binary Classifier    │        │  │
│  │  │ - Semantic (60%)    │         │ - PyTorch NN        │        │  │
│  │  │ - Skills (30%)      │  +      │ - Probability Output │        │  │
│  │  │ - Experience (10%)  │         │ - Threshold: 0.7    │        │  │
│  │  └─────────────────────┘         └──────────────────────┘        │  │
│  │           ▲                                    ▲                   │  │
│  └───────────┼────────────────────────────────────┼──────────────────┘  │
│              │                                    │                      │
│  ┌───────────┼────────────────┐    ┌─────────────┼─────────────────┐  │
│  │  Text Preprocessing        │    │  Feature Extraction           │  │
│  │  - Cleaning                │    │  - Skills Detection          │  │
│  │  - Normalization           │    │  - Experience Parsing        │  │
│  │  - Section Extraction      │    │  - TF-IDF Features          │  │
│  └────────────────────────────┘    └──────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────┐
│                      ML/AI SERVICES LAYER                                │
│  ┌─────────────────────┐         ┌──────────────────────────────────┐  │
│  │  AWS Bedrock        │         │  PyTorch Classifier              │  │
│  │  Titan Embeddings   │         │  - Input: 2x1536 embeddings     │  │
│  │  - Dimension: 1536  │         │  - Hidden: 512→256→128          │  │
│  │  - Semantic Vectors │         │  - Output: Binary Probability    │  │
│  └─────────────────────┘         └──────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────────┐
│                       STORAGE LAYER                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │  PostgreSQL │  │   Pinecone   │  │   AWS S3    │  │   MLflow     │ │
│  │  - Jobs     │  │  - Embeddings│  │  - Resumes  │  │  - Models    │ │
│  │  - Resumes  │  │  - Vectors   │  │  - Models   │  │  - Metrics   │ │
│  │  - Results  │  │  - Metadata  │  │  - Data     │  │  - Artifacts │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘ │
└───────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS                                 │
│  ┌────────────────────────┐         ┌────────────────────────────┐    │
│  │  AWS Lambda            │         │  SageMaker                 │    │
│  │  - Serverless          │    OR   │  - Enterprise Scale        │    │
│  │  - API Gateway         │         │  - Auto-scaling            │    │
│  │  - Cost-effective      │         │  - Batch Processing        │    │
│  └────────────────────────┘         └────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend Layer

**Technology**: React 18 + TypeScript + Material-UI

**Components**:
- **Dashboard**: Overview with KPIs and metrics
- **Job Management**: CRUD operations for job descriptions
- **Resume Upload**: Drag-drop file upload with progress tracking
- **Screening Results**: Paginated table with filtering and sorting
- **Analytics**: Charts and visualizations for insights

**State Management**: Zustand for global state, React Query for API caching

### 2. API Layer

**Technology**: FastAPI + Pydantic

**Routes**:
- `/api/v1/health` - Health checks
- `/api/v1/jobs` - Job management endpoints
- `/api/v1/screening` - Resume screening operations
- `/api/v1/analytics` - Metrics and analytics

**Features**:
- Async request handling
- Request validation with Pydantic
- JWT authentication
- CORS configuration
- Request/response logging
- Exception handling

### 3. ML Pipeline

#### A. Text Preprocessing
```python
Resume → Clean Text → Extract Sections → Feature Extraction → Embeddings
```

**Operations**:
- Remove PII (emails, phones)
- Normalize unicode
- Extract skills, education, experience
- Generate TF-IDF features

#### B. Similarity Scoring

**Formula**:
```
similarity_score = 
    0.5 × semantic_similarity +
    0.3 × skills_match_ratio +
    0.2 × experience_match_score
```

**Components**:
- Semantic similarity using cosine distance on Bedrock embeddings
- Skills matching with required vs preferred weighting
- Experience matching with range validation

#### C. Binary Classification

**Architecture**:
```
Input: [resume_embedding + job_embedding] (3072 dims)
  ↓
Linear(3072 → 512) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
  ↓
Linear(128 → 1) + Sigmoid
  ↓
Output: Probability [0-1]
```

**Training**:
- Loss: Binary Cross Entropy
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=5

#### D. Hybrid Decision Logic

```python
# Weighted ensemble
final_score = 0.4 × (similarity/100) + 0.6 × classifier_prob

# Decision rules
if similarity >= 70 AND classifier_prob >= 0.7:
    decision = "SHORTLIST"
elif similarity >= 70 OR classifier_prob >= 0.7:
    decision = "REVIEW"  # Conflicting signals
else:
    decision = "REJECT"
```

### 4. Storage Architecture

#### PostgreSQL Schema

```sql
-- Jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    company VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    required_skills JSONB,
    status VARCHAR(20),
    created_at TIMESTAMP,
    INDEX idx_status (status)
);

-- Resumes table
CREATE TABLE resumes (
    id UUID PRIMARY KEY,
    candidate_name VARCHAR(200),
    file_path VARCHAR(500),
    skills JSONB,
    experience_years FLOAT,
    created_at TIMESTAMP,
    INDEX idx_created (created_at)
);

-- Screening results
CREATE TABLE screening_results (
    id UUID PRIMARY KEY,
    resume_id UUID REFERENCES resumes(id),
    job_id UUID REFERENCES jobs(id),
    decision VARCHAR(20),
    final_score FLOAT,
    explanation TEXT,
    created_at TIMESTAMP,
    INDEX idx_job_decision (job_id, decision)
);
```

#### Pinecone Vector Store

```python
# Index configuration
{
    "dimension": 1536,
    "metric": "cosine",
    "pods": 1,
    "replicas": 1,
    "pod_type": "p1.x1"
}

# Vector structure
{
    "id": "resume_uuid",
    "values": [0.1, 0.2, ...],  # 1536 dims
    "metadata": {
        "candidate_name": "John Doe",
        "skills": ["python", "ml"],
        "experience_years": 5
    }
}
```

### 5. MLOps Pipeline

#### Model Training Flow

```
1. Data Collection → S3
   ↓
2. SageMaker Training Job
   - Load data from S3
   - Train PyTorch model
   - Track with MLflow
   - Save checkpoint
   ↓
3. Model Evaluation
   - Validation metrics
   - Compare with baseline
   ↓
4. Model Registry
   - Version in S3
   - Update metadata
   - Set as active
   ↓
5. Deployment
   - Update Lambda layer
   - Update SageMaker endpoint
```

#### Auto-Retraining Trigger

```
S3 Event (new labeled data)
   ↓
Lambda Function
   ↓
Trigger SageMaker Training Job
   ↓
Evaluate New Model
   ↓
If metrics improved:
   → Promote to production
   → Update endpoints
Else:
   → Keep current model
```

### 6. Deployment Architecture

#### Lambda Deployment

```
┌──────────────┐
│ API Gateway  │
│   REST API   │
└──────┬───────┘
       │
┌──────▼────────────────────┐
│  Lambda Function          │
│  - Runtime: Python 3.11   │
│  - Memory: 1024 MB        │
│  - Timeout: 30s           │
│  - Layers: PyTorch CPU    │
│                           │
│  ┌─────────────────────┐ │
│  │ Cold Start Init:    │ │
│  │ - Load model        │ │
│  │ - Init embedder     │ │
│  │ - Connect to DB     │ │
│  └─────────────────────┘ │
│                           │
│  ┌─────────────────────┐ │
│  │ Request Handler:    │ │
│  │ - Parse request     │ │
│  │ - Generate embed    │ │
│  │ - Run inference     │ │
│  │ - Return result     │ │
│  └─────────────────────┘ │
└───────────────────────────┘
```

#### SageMaker Deployment

```
┌──────────────────────────┐
│ SageMaker Endpoint       │
│ - Instance: ml.m5.xlarge │
│ - Auto-scaling: 1-10     │
│ - Model: PyTorch         │
└──────────────────────────┘
```

### 7. Monitoring & Observability

#### CloudWatch Metrics

- **Lambda**: Invocations, Duration, Errors, Throttles
- **SageMaker**: ModelLatency, Invocations, ModelInvocations4xx/5xx
- **Custom**: InferenceAccuracy, ProcessingTime, QueueDepth

#### Logging Strategy

```
Application Logs → CloudWatch Logs → CloudWatch Insights
                                    → Alarms
                                    → Dashboards
```

#### MLflow Tracking

```
Experiment Tracking:
  - Hyperparameters
  - Training metrics (loss, accuracy, F1)
  - Validation metrics
  - Training duration
  
Model Registry:
  - Model artifacts
  - Model versions
  - Performance comparison
  - Deployment status
```

## Security Architecture

### Authentication & Authorization

- JWT tokens for API authentication
- AWS IAM roles with least privilege
- Secrets stored in AWS Secrets Manager
- API rate limiting with API Gateway

### Data Security

- TLS 1.3 for data in transit
- S3 encryption at rest (AES-256)
- RDS encryption at rest
- VPC isolation for compute resources

### PII Protection

- Resume text scrubbing (remove emails, phones)
- Separate storage for PII and processed data
- Audit logging for data access

## Scalability Considerations

### Horizontal Scaling

- Lambda: Auto-scales with concurrent executions (up to 1000)
- SageMaker: Auto-scaling based on invocations/latency
- RDS: Read replicas for analytics queries
- Pinecone: Distributed vector index

### Performance Optimization

- Embedding caching (Redis/ElastiCache)
- Database connection pooling
- Async request processing
- Batch operations for bulk screening

### Cost Optimization

- Lambda: Pay per invocation
- S3: Lifecycle policies for old data
- SageMaker: Stop endpoint when not in use
- Spot instances for training

## Disaster Recovery

### Backup Strategy

- RDS: Automated daily backups (retention: 7 days)
- S3: Versioning enabled
- Model artifacts: Cross-region replication

### High Availability

- Multi-AZ deployment for RDS
- API Gateway in multiple regions
- CloudFront for frontend distribution

---

**Last Updated**: 2025-10-17
**Version**: 1.0.0

