# 🚀 AI-Powered Resume Screening System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org/)

An enterprise-grade AI-powered resume screening system that automatically evaluates candidates against job descriptions using semantic similarity and machine learning classification.

## ✨ Key Features

- **🤖 Hybrid AI Decision Engine**: Combines semantic similarity scoring with binary classification
- **☁️ AWS Bedrock Integration**: Leverages Amazon Titan Embeddings for deep semantic understanding
- **📊 Vector Database**: Pinecone for efficient similarity search at scale
- **🧠 PyTorch Classifier**: Custom-trained neural network for candidate evaluation
- **⚡ Dual Deployment**: AWS Lambda (serverless) + SageMaker (enterprise)
- **📈 MLOps Pipeline**: MLflow tracking, model versioning, automated retraining
- **💻 Modern UI**: React + TypeScript dashboard with real-time analytics
- **🏗️ Production-Ready**: Terraform IaC, Docker, comprehensive testing
- **🔄 Full Stack**: FastAPI backend, PostgreSQL database, S3 storage

## 📋 Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Training the Classifier](#training-the-classifier)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

### High-Level Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Frontend   │────▶│   FastAPI    │────▶│   AWS       │
│  (React)    │     │   Backend    │     │   Services  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                     │
                           │                     ├─ Bedrock (Embeddings)
                           │                     ├─ S3 (Storage)
                           │                     ├─ RDS PostgreSQL
                           │                     ├─ Pinecone (Vectors)
                           ▼                     ├─ Lambda
                    ┌──────────────┐            └─ SageMaker
                    │  ML Models   │
                    ├──────────────┤
                    │ Similarity   │
                    │ Classifier   │
                    │ Hybrid       │
                    └──────────────┘
```

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI 0.109+
- **ML**: PyTorch 2.2, scikit-learn
- **Embeddings**: AWS Bedrock Titan
- **Vector DB**: Pinecone
- **Database**: PostgreSQL 15
- **Storage**: AWS S3
- **MLOps**: MLflow, SageMaker

### Frontend
- **Framework**: React 18 + TypeScript
- **UI**: Material-UI (MUI)
- **State**: Zustand
- **Charts**: Recharts
- **Build**: Vite

### Infrastructure
- **IaC**: Terraform
- **Compute**: AWS Lambda, SageMaker
- **CI/CD**: Docker, Docker Compose
- **Monitoring**: CloudWatch

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+** - [Download](https://www.python.org/downloads/)
- **Node.js 18+** - [Download](https://nodejs.org/)
- **Docker & Docker Compose** - [Download](https://www.docker.com/products/docker-desktop)
- **PostgreSQL 15+** (or use Docker Compose) - [Download](https://www.postgresql.org/download/)
- **AWS Account** with credentials configured - [Sign Up](https://aws.amazon.com/)
- **Pinecone API Key** - [Sign Up](https://www.pinecone.io/)

### AWS Services Required

- **AWS Bedrock** (for Titan embeddings)
- **S3** (for storage)
- **RDS** (optional, for production PostgreSQL)
- **Lambda** (optional, for serverless deployment)
- **SageMaker** (optional, for ML training/inference)

## 🚀 Installation

### Option 1: Docker Compose (Recommended for Development)

1. **Clone the repository**
```bash
git clone https://github.com/devsohail/resume-screening.git
cd resume-screening
```

2. **Copy environment file**
```bash
cp .env.example .env
```

3. **Edit `.env` with your credentials**
```bash
# Required: AWS credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Required: Pinecone API key
PINECONE_API_KEY=your_pinecone_api_key

# Required: S3 bucket name (create one in AWS)
S3_BUCKET_RESUMES=your-bucket-name
S3_BUCKET_MODELS=your-bucket-name
S3_BUCKET_DATA=your-bucket-name

# Generate secure keys for production
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

4. **Start all services**
```bash
docker-compose up -d
```

5. **Run database migrations**
```bash
docker-compose exec backend alembic upgrade head
```

6. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/v1/docs
- MLflow: http://localhost:5000

### Option 2: Manual Setup

#### Backend Setup

1. **Create Python virtual environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup database**
```bash
# Start PostgreSQL (if not using Docker)
# Create database
createdb resume_screening

# Run migrations
alembic upgrade head
```

4. **Start backend server**
```bash
uvicorn backend.api.main:app --reload
```

#### Frontend Setup

1. **Install dependencies**
```bash
cd frontend
npm install
```

2. **Start development server**
```bash
npm run dev
```

## ⚙️ Configuration

### Environment Variables

All configuration is managed through environment variables. See `.env.example` for all available options.

#### Essential Configuration

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# S3 Buckets
S3_BUCKET_RESUMES=your-bucket-name
S3_BUCKET_MODELS=your-bucket-name
S3_BUCKET_DATA=your-bucket-name

# Bedrock
BEDROCK_MODEL_ID=amazon.titan-embed-text-v1

# Pinecone
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=resume-embeddings

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/resume_screening

# Security
SECRET_KEY=your-random-secret
JWT_SECRET=your-random-jwt-secret
```

#### Model Configuration

```bash
# Adjust these based on your requirements
SIMILARITY_THRESHOLD=70.0        # 0-100 scale
CLASSIFIER_THRESHOLD=0.7         # 0-1 probability
HYBRID_SIMILARITY_WEIGHT=0.4     # Weight for similarity
HYBRID_CLASSIFIER_WEIGHT=0.6     # Weight for classifier
```

### Setting up AWS Bedrock

1. Enable AWS Bedrock in your account
2. Request access to Titan Embeddings model
3. Ensure your AWS credentials have Bedrock permissions

### Setting up Pinecone

1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Create a new index:
   - Name: `resume-embeddings`
   - Dimensions: `1536` (for Titan embeddings)
   - Metric: `cosine`
   - Cloud: `aws`
   - Region: `us-east-1`

## 🎯 Usage

### Creating a Job

```python
import requests

job_data = {
    "title": "Senior Python Developer",
    "company": "TechCorp",
    "description": "Looking for an experienced Python developer with expertise in FastAPI, AWS, and machine learning.",
    "required_skills": ["python", "fastapi", "aws"],
    "preferred_skills": ["machine learning", "docker"],
    "min_experience_years": 5
}

response = requests.post(
    "http://localhost:8000/api/v1/jobs",
    json=job_data
)
job = response.json()
print(f"Created job: {job['id']}")
```

### Uploading & Screening a Resume

```python
files = {'file': open('resume.pdf', 'rb')}
data = {
    'job_id': job['id'],
    'candidate_name': 'John Doe',
    'candidate_email': 'john@example.com'
}

response = requests.post(
    "http://localhost:8000/api/v1/screening/upload-resume",
    files=files,
    data=data
)

result = response.json()
print(f"Decision: {result['screening_result']['decision']}")
print(f"Score: {result['screening_result']['final_score']}")
print(f"Explanation: {result['screening_result']['explanation']}")
```

### Getting Screening Results

```python
response = requests.get(
    f"http://localhost:8000/api/v1/screening/results/{job['id']}"
)

results = response.json()
for candidate in results['results']:
    print(f"{candidate['candidate_name']}: {candidate['decision']} ({candidate['final_score']})")
```

### Using the Web Interface

1. Navigate to http://localhost:3000
2. Click "Create Job" and fill in the job details
3. Click on a job to view details
4. Upload resumes for screening
5. View results, scores, and explanations
6. Provide feedback to improve the model

## 📚 API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/jobs` | Create a new job |
| GET | `/api/v1/jobs` | List all jobs |
| GET | `/api/v1/jobs/{id}` | Get job details |
| POST | `/api/v1/screening/upload-resume` | Upload and screen resume |
| GET | `/api/v1/screening/results/{job_id}` | Get screening results |
| POST | `/api/v1/feedback/submit` | Submit human feedback |
| GET | `/api/v1/analytics/metrics` | Get analytics metrics |
| GET | `/api/v1/health` | Health check |

## 🚀 Deployment

### AWS Deployment with Terraform

1. **Install Terraform**
```bash
# macOS
brew install terraform

# Or download from https://www.terraform.io/downloads
```

2. **Configure AWS credentials**
```bash
aws configure
```

3. **Initialize and deploy**
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

For detailed deployment instructions, see [AWS_SERVICES_GUIDE.md](AWS_SERVICES_GUIDE.md).

### Docker Deployment

Build and push Docker images:

```bash
# Backend
cd backend
docker build -t your-registry/resume-screening-backend:latest .
docker push your-registry/resume-screening-backend:latest

# Frontend
cd frontend
docker build -t your-registry/resume-screening-frontend:latest .
docker push your-registry/resume-screening-frontend:latest
```

## 🧠 Training the Classifier

### Using Jupyter Notebooks

1. **Start Jupyter**
```bash
cd notebooks
jupyter lab
```

2. **Run notebooks in order**:
   - `01_data_exploration.ipynb` - Explore training data
   - More notebooks available in the `/notebooks` directory

### Training Script

```bash
python scripts/auto_train.py \
    --min-samples 100 \
    --epochs 50 \
    --batch-size 32
```

For detailed training instructions, see [MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md).

## 🧪 Testing

### Run all tests
```bash
cd backend
pytest tests/ -v --cov=backend
```

### Run specific test suites
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage report
pytest tests/ --cov=backend --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
black backend/

# Check linting
flake8 backend/

# Type checking
mypy backend/

# Security scan
bandit -r backend/
```

## 📊 Monitoring

### MLflow

Access MLflow UI at http://localhost:5000 to view:
- Training experiments
- Model versions
- Performance metrics
- Hyperparameters

### Logs

```bash
# Backend logs
docker-compose logs -f backend

# All service logs
docker-compose logs -f
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AWS Bedrock](https://aws.amazon.com/bedrock/) for Titan Embeddings
- [Pinecone](https://www.pinecone.io/) for vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent framework
- [Material-UI](https://mui.com/) for beautiful components
- [MLflow](https://mlflow.org/) for experiment tracking

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/devsohail/resume-screening/issues)
- **Discussions**: [GitHub Discussions](https://github.com/devsohail/resume-screening/discussions)

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Bulk resume processing
- [ ] Interview scheduling integration
- [ ] Custom model fine-tuning UI
- [ ] Mobile application

---

**Built with ❤️ using Python, React, and AWS**

⭐ Star this repo if you find it helpful!
