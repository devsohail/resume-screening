# Quick Start Guide

Get the Resume Screening System running in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.11+ installed
- [ ] Node.js 18+ installed
- [ ] Docker & Docker Compose installed
- [ ] AWS account with credentials configured
- [ ] Pinecone account (free tier works)

## Option 1: Docker Compose (Recommended)

The fastest way to get started:

```bash
# 1. Clone and enter directory
git clone https://github.com/yourusername/resume-screening.git
cd resume-screening

# 2. Copy environment file
cp .env.example .env

# 3. Edit .env with your credentials
nano .env  # Or use your favorite editor

# Required: Add these values to .env
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
# PINECONE_API_KEY=your_pinecone_key

# 4. Start all services
docker-compose up -d

# 5. Check logs
docker-compose logs -f

# 6. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/api/v1/docs
# MLflow: http://localhost:5000
```

## Option 2: Manual Setup

For development:

```bash
# 1. Clone repository
git clone https://github.com/yourusername/resume-screening.git
cd resume-screening

# 2. Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Start PostgreSQL (if not using Docker)
# Install and start PostgreSQL 15

# 5. Initialize database
python -c "from backend.storage.db_handler import get_db_handler; get_db_handler().create_tables()"

# 6. Start backend
cd backend
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# 7. In new terminal, start frontend
cd frontend
npm run dev
```

## First Steps After Installation

### 1. Create Your First Job

Visit http://localhost:3000 or use the API:

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior Python Developer",
    "company": "TechCorp",
    "description": "We are seeking an experienced Python developer with ML expertise...",
    "required_skills": ["python", "fastapi", "aws", "postgresql"],
    "preferred_skills": ["machine learning", "docker"],
    "min_experience_years": 5,
    "location": "Remote"
  }'
```

### 2. Upload and Screen a Resume

```bash
# Save the job_id from step 1
JOB_ID="<job-id-from-response>"

# Upload a resume
curl -X POST http://localhost:8000/api/v1/screening/upload-resume \
  -F "file=@/path/to/resume.pdf" \
  -F "job_id=$JOB_ID" \
  -F "candidate_name=John Doe" \
  -F "candidate_email=john@example.com"
```

### 3. View Results

Visit http://localhost:3000/screening/$JOB_ID or:

```bash
curl http://localhost:8000/api/v1/screening/results/$JOB_ID
```

## Common Issues & Solutions

### Issue: "AWS credentials not found"

**Solution**: Ensure `.env` file has correct AWS keys or run `aws configure`

### Issue: "Pinecone connection failed"

**Solution**:

1. Sign up at https://www.pinecone.io/
2. Get API key from dashboard
3. Add to `.env` file

### Issue: "Database connection error"

**Solution**:

- If using Docker: `docker-compose restart postgres`
- If manual: Ensure PostgreSQL is running on port 5432

### Issue: "Port already in use"

**Solution**: Change ports in `docker-compose.yml` or stop conflicting services

### Issue: "Model not found"

**Solution**: The classifier needs training first (see Training section in README)

## Test the System

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### API Documentation

Visit: http://localhost:8000/api/v1/docs

### Run Tests

```bash
cd backend
pytest tests/ -v
```

## Stopping the Application

### Docker Compose

```bash
docker-compose down
```

### Manual

- Press `Ctrl+C` in each terminal running services

## Next Steps

1. **Train the Classifier**: See `notebooks/03_classifier_training.ipynb`
2. **Customize Scoring**: Edit weights in `.env`
3. **Deploy to AWS**: Run `./scripts/deploy.sh`
4. **Explore Analytics**: Visit http://localhost:3000/analytics
5. **Read Full Docs**: See `README.md` and `ARCHITECTURE.md`

## Getting Help

- **Documentation**: See README.md and ARCHITECTURE.md
- **API Reference**: http://localhost:8000/api/v1/docs
- **Issues**: Create an issue on GitHub
- **Logs**: `docker-compose logs -f` or check `backend/logs/`

## Production Deployment

For AWS deployment:

```bash
# 1. Configure Terraform
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

# 2. Deploy
terraform init
terraform apply

# 3. Or use the deployment script
cd ..
./scripts/deploy.sh
```

---

**You're all set! 🎉**

Start screening resumes intelligently with AI!
