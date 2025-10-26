# 🏗️ Visual Architecture Diagram

## System Architecture with AWS Services

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          RESUME SCREENING SYSTEM                                     │
│                         (Intelligent AI-Powered Platform)                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│  👤 USER INTERFACE (React + TypeScript)                                             │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  • Upload Resumes    • View Results    • Review Decisions    • Analytics Dashboard  │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ↓
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🔌 API LAYER (FastAPI)                                                              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  POST /upload-resume   GET /results   POST /feedback   GET /analytics               │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ↓                    ↓                    ↓
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │   📄 S3 BUCKET   │  │  💾 POSTGRESQL   │  │ 🧠 AWS BEDROCK   │
         │   (Storage)      │  │   (Database)     │  │  (Embeddings)    │
         ├──────────────────┤  ├──────────────────┤  ├──────────────────┤
         │                  │  │                  │  │                  │
         │ • Resume PDFs    │  │ • Jobs           │  │ • Titan Model    │
         │ • Model files    │  │ • Resumes        │  │ • Text → Vector  │
         │ • Training data  │  │ • Results        │  │ • 1536 dims      │
         │                  │  │ • Feedback       │  │                  │
         │ Cost: $0.02/GB   │  │ Cost: $30/month  │  │ Cost: $0.1/1K    │
         └──────────────────┘  └──────────────────┘  └──────────────────┘
                                                                │
                                                                ↓
         ┌────────────────────────────────────────────────────────────────┐
         │                    EMBEDDING VECTORS                           │
         │                  [0.234, -0.567, 0.891, ...]                   │
         └────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ↓                    ↓                    ↓
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │  📍 PINECONE     │  │ 🤖 SAGEMAKER     │  │  🎯 HYBRID       │
         │  (Vector DB)     │  │  (ML Platform)   │  │   SCORING        │
         ├──────────────────┤  ├──────────────────┤  ├──────────────────┤
         │                  │  │                  │  │                  │
         │ • Store vectors  │  │ • Train models   │  │ • Similarity     │
         │ • Fast search    │  │ • Deploy models  │  │ • Classifier     │
         │ • Similarity     │  │ • Auto-scale     │  │ • Final score    │
         │ • Metadata       │  │ • Monitoring     │  │                  │
         │                  │  │                  │  │ Weight: 60/40    │
         │ Cost: Free tier  │  │ Cost: $50+/mo    │  └──────────────────┘
         │   or $70/month   │  │   or pay/use     │
         └──────────────────┘  └──────────────────┘
              (Optional)            (Optional)


┌──────────────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PIPELINE                                         │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  1. 📊 Collect Data           2. 🧮 Generate Embeddings      3. 🏋️  Train Model    │
│     ↓                             ↓                              ↓                  │
│  ┌─────────────┐             ┌─────────────┐              ┌─────────────┐          │
│  │ 100+ human  │    →        │ AWS Bedrock │    →         │ SageMaker   │          │
│  │ reviewed    │             │ embeddings  │              │ Training    │          │
│  │ samples     │             │ for each    │              │ GPU         │          │
│  └─────────────┘             └─────────────┘              └─────────────┘          │
│                                                                    ↓                 │
│  4. 📈 Evaluate              5. ✅ Register              6. 🚀 Deploy               │
│     ↓                             ↓                              ↓                  │
│  ┌─────────────┐             ┌─────────────┐              ┌─────────────┐          │
│  │ Test set    │    →        │ Model       │    →         │ SageMaker   │          │
│  │ metrics     │             │ Registry    │              │ Endpoint    │          │
│  │ MLflow      │             │ versioning  │              │ or Lambda   │          │
│  └─────────────┘             └─────────────┘              └─────────────┘          │
│                                                                                      │
│  Frequency: Daily/Weekly automatic check (cron or EventBridge)                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Request Flow: Upload Resume → Get Decision

```
┌──────────┐
│  USER    │  Upload resume.pdf
│ (Browser)│ ────────────────────┐
└──────────┘                     │
                                 ↓
                         ┌────────────────┐
                         │   FastAPI      │
                         │   Backend      │
                         └────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ↓                ↓                ↓
       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
       │     S3      │  │  PostgreSQL │  │   Bedrock   │
       │  Save PDF   │  │  Save job   │  │  Generate   │
       │             │  │  metadata   │  │  embedding  │
       └─────────────┘  └─────────────┘  └─────────────┘
                                                 │
                                  ┌──────────────┴───────────────┐
                                  ↓                              ↓
                         ┌─────────────┐              ┌─────────────┐
                         │  Pinecone   │              │  SageMaker  │
                         │  Search for │              │  Predict:   │
                         │  similar    │              │  SHORTLIST? │
                         │  candidates │              │             │
                         └─────────────┘              └─────────────┘
                                  │                              │
                                  └──────────────┬───────────────┘
                                                 ↓
                                       ┌─────────────────┐
                                       │  Hybrid Engine  │
                                       │  Combine scores │
                                       │  Make decision  │
                                       └─────────────────┘
                                                 │
                                                 ↓
                                       ┌─────────────────┐
                                       │   PostgreSQL    │
                                       │  Save result    │
                                       └─────────────────┘
                                                 │
                                                 ↓
                                       ┌─────────────────┐
                                       │  Return to UI   │
                                       │  Show decision  │
                                       │  Score: 89.5    │
                                       └─────────────────┘
```

---

## 📊 Service Interaction Matrix

| When You... | Uses These Services |
|-------------|---------------------|
| **Upload Resume** | S3 (store file) → Bedrock (embedding) → Pinecone (index) → PostgreSQL (metadata) |
| **Screen Resume** | PostgreSQL (get job) → Bedrock (embed) → SageMaker (classify) → Pinecone (search) → PostgreSQL (save result) |
| **Review Decision** | PostgreSQL (update feedback) |
| **Train Model** | PostgreSQL (export data) → Bedrock (embeddings) → SageMaker (training) → S3 (save model) |
| **Deploy Model** | S3 (load model) → SageMaker (create endpoint) |
| **Search Candidates** | Pinecone (vector search) → PostgreSQL (get details) |

---

## 🎯 Service Dependencies

```
Core Services (Required):
  ├─ PostgreSQL ✅ (Main database)
  ├─ S3 ✅ (File storage)
  └─ AWS Bedrock ✅ (Embeddings - REQUIRED for semantic matching)

Optional Services (Enhanced Features):
  ├─ Pinecone ⚠️ (Fast vector search - nice to have)
  └─ SageMaker ⚠️ (ML training/deployment - can use local model)

Minimum Viable System:
  ✅ PostgreSQL + S3 + Bedrock
  → Works with similarity scoring only
  → Cost: ~$30/month

Recommended Production:
  ✅ PostgreSQL + S3 + Bedrock + SageMaker + Pinecone
  → Full ML with fast search
  → Cost: ~$150/month

Enterprise Scale:
  ✅ All services + multi-region + HA
  → High availability, global scale
  → Cost: $500-2000/month
```

---

## 🚦 Deployment Stages

### Stage 1: Development (Current)
```
Frontend: localhost:5173
Backend: localhost:8000
Database: AWS RDS PostgreSQL
Storage: AWS S3
Embeddings: AWS Bedrock
Vector DB: ❌ (disabled)
ML Model: 📁 Local file (models/classifier.pt)

Cost: ~$30/month
```

### Stage 2: Beta Production
```
Frontend: CloudFront + S3
Backend: Lambda + API Gateway
Database: AWS RDS PostgreSQL
Storage: AWS S3
Embeddings: AWS Bedrock
Vector DB: Pinecone Serverless
ML Model: 🚀 SageMaker Endpoint (single instance)

Cost: ~$150/month
```

### Stage 3: Production
```
Frontend: CloudFront + S3
Backend: ECS Fargate (containers)
Database: AWS RDS PostgreSQL (Multi-AZ)
Storage: AWS S3 (versioning + lifecycle)
Embeddings: AWS Bedrock
Vector DB: Pinecone Enterprise (multi-pod)
ML Model: 🚀 SageMaker Endpoint (auto-scaling)
Monitoring: CloudWatch + X-Ray
Security: WAF + GuardDuty

Cost: $500-2000/month
```

---

## 💰 Cost Breakdown Example (100 resumes/day)

```
Service              Usage              Cost/Month    Required?
──────────────────────────────────────────────────────────────
PostgreSQL (RDS)     db.t3.small       $30           ✅ Yes
S3 Storage           10 GB             $0.23         ✅ Yes
AWS Bedrock          3,000 calls       $0.30         ✅ Yes
SageMaker Endpoint   ml.t2.medium      $47           ⚠️  Optional
Pinecone Serverless  3K queries        $10           ⚠️  Optional
Lambda (API)         100K requests     $0.20         ⚠️  Optional
API Gateway          100K requests     $0.35         ⚠️  Optional
CloudWatch Logs      1 GB              $0.50         ✅ Yes
Data Transfer        10 GB             $0.90         ✅ Yes
──────────────────────────────────────────────────────────────
TOTAL (Minimal)                        $32.48        Without SageMaker/Pinecone
TOTAL (Recommended)                    $89.48        With SageMaker + Pinecone
TOTAL (Enterprise)                     $500+         With HA + scaling
```

---

## 🔍 Quick Reference

**Need embeddings?** → AWS Bedrock (Required)
**Need to store files?** → S3 (Required)
**Need to store data?** → PostgreSQL (Required)
**Need fast vector search?** → Pinecone (Optional - PostgreSQL fallback)
**Need ML predictions?** → SageMaker (Optional - local model fallback)

**TL;DR**: You NEED Bedrock + S3 + PostgreSQL. Everything else is optional but recommended for production!

---

See [AWS_SERVICES_GUIDE.md](AWS_SERVICES_GUIDE.md) for detailed explanations of each service.



