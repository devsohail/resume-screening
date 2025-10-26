# 🎯 SETUP_GUIDE.md

Complete step-by-step guide to set up the Resume Screening System from scratch.

## Table of Contents

1. [AWS Account Setup](#aws-account-setup)
2. [Pinecone Setup](#pinecone-setup)
3. [Local Development Setup](#local-development-setup)
4. [Database Setup](#database-setup)
5. [AWS Services Configuration](#aws-services-configuration)
6. [Troubleshooting](#troubleshooting)

---

## 1. AWS Account Setup

### Create AWS Account

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Follow the signup process
4. Add a payment method (free tier available)

### Create IAM User

**Important**: Don't use your root account credentials!

1. Sign in to AWS Console
2. Go to **IAM** (Identity and Access Management)
3. Click **Users** → **Add users**
4. User details:
   - Username: `resume-screening-user`
   - Access type: ☑️ Programmatic access
5. Click **Next: Permissions**
6. Attach policies:
   - ☑️ `AmazonS3FullAccess`
   - ☑️ `AmazonBedrockFullAccess`
   - ☑️ `AWSLambda_FullAccess` (optional, for serverless)
   - ☑️ `AmazonSageMakerFullAccess` (optional, for ML training)
7. Click **Next** through tags
8. Click **Create user**
9. **IMPORTANT**: Download the CSV with credentials (you can't retrieve the secret key later!)

### Enable AWS Bedrock

1. In AWS Console, search for **Bedrock**
2. Click **Get Started**
3. Go to **Model access** in the left sidebar
4. Click **Manage model access**
5. Find **Titan Text Embeddings** and click **Request access**
6. Wait for approval (usually instant)

### Create S3 Bucket

1. Go to **S3** in AWS Console
2. Click **Create bucket**
3. Bucket settings:
   - Name: `your-unique-bucket-name` (e.g., `mycompany-resume-screening`)
   - Region: `us-east-1` (or your preferred region)
   - Block all public access: ☑️ **Enabled** (for security)
4. Create folder structure:
   ```
   your-bucket/
   ├── resumes/        # Uploaded resumes
   ├── models/         # Trained ML models
   └── data/           # Training data
   ```

---

## 2. Pinecone Setup

Pinecone is our vector database for storing and searching embeddings.

### Create Account

1. Go to [app.pinecone.io](https://app.pinecone.io)
2. Click **Sign Up** (free tier available)
3. Verify your email

### Create Index

1. Click **Create Index**
2. Index settings:
   - Name: `resume-embeddings`
   - Dimensions: `1536` (for Amazon Titan embeddings)
   - Metric: `cosine`
   - Cloud: `AWS`
   - Region: `us-east-1` (same as your AWS region)
   - Pod Type: `s1.x1` (starter, can upgrade later)
3. Click **Create Index**
4. Wait for index to be ready (~1-2 minutes)

### Get API Key

1. Click on your profile icon (top right)
2. Click **API Keys**
3. Copy your API key (it looks like: `pcsk_...`)
4. **Keep this secret!**

---

## 3. Local Development Setup

### Install Prerequisites

#### macOS

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Install Node.js
brew install node@18

# Install Docker Desktop
brew install --cask docker
```

#### Ubuntu/Debian

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### Windows

1. Install [Python 3.11](https://www.python.org/downloads/)
2. Install [Node.js 18](https://nodejs.org/)
3. Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
4. Install [Git](https://git-scm.com/download/win)

### Clone Repository

```bash
git clone https://github.com/devsohail/resume-screening.git
cd resume-screening
```

### Configure Environment

1. **Copy environment template**
```bash
cp .env.example .env
```

2. **Edit `.env` file** with your credentials:

```bash
# Use your preferred editor
nano .env
# or
code .env
# or
vim .env
```

3. **Fill in these required values**:

```bash
# AWS Credentials (from IAM user CSV)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=wJal...
AWS_REGION=us-east-1

# S3 Bucket (created earlier)
S3_BUCKET_RESUMES=your-bucket-name
S3_BUCKET_MODELS=your-bucket-name
S3_BUCKET_DATA=your-bucket-name

# Pinecone (from Pinecone dashboard)
PINECONE_API_KEY=pcsk_...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=resume-embeddings

# Generate secure keys for production
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Start with Docker Compose

```bash
# Start all services (PostgreSQL, Backend, Frontend, MLflow)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop when done
docker-compose down
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/v1/docs
- **MLflow**: http://localhost:5000

---

## 4. Database Setup

### Option A: Docker (Recommended)

Docker Compose automatically sets up PostgreSQL. No manual setup needed!

### Option B: Local PostgreSQL

If you prefer local PostgreSQL installation:

#### Install PostgreSQL

**macOS**:
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Ubuntu/Debian**:
```bash
sudo apt install postgresql-15
sudo systemctl start postgresql
```

#### Create Database

```bash
# Login to PostgreSQL
sudo -u postgres psql

# Create database and user
CREATE DATABASE resume_screening;
CREATE USER resume_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE resume_screening TO resume_user;
\q
```

#### Update `.env`

```bash
DATABASE_URL=postgresql://resume_user:secure_password@localhost:5432/resume_screening
```

#### Run Migrations

```bash
cd backend
alembic upgrade head
```

---

## 5. AWS Services Configuration

### Configure AWS CLI (Optional but Recommended)

```bash
# Install AWS CLI
# macOS
brew install awscli

# Ubuntu/Debian
sudo apt install awscli

# Configure with your credentials
aws configure
# Enter:
#   AWS Access Key ID: [from IAM CSV]
#   AWS Secret Access Key: [from IAM CSV]
#   Default region: us-east-1
#   Default output format: json
```

### Test AWS Bedrock

```bash
# Test from Python
python3 << EOF
import boto3

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
response = bedrock.invoke_model(
    modelId='amazon.titan-embed-text-v1',
    body='{"inputText": "test"}'
)
print("✅ Bedrock working!")
EOF
```

### Test S3 Access

```bash
# List your buckets
aws s3 ls

# Test upload
echo "test" > test.txt
aws s3 cp test.txt s3://your-bucket-name/test.txt
aws s3 rm s3://your-bucket-name/test.txt
rm test.txt
```

### Test Pinecone

```python
python3 << EOF
from pinecone import Pinecone

pc = Pinecone(api_key='your-api-key')
index = pc.Index('resume-embeddings')
print(index.describe_index_stats())
print("✅ Pinecone working!")
EOF
```

---

## 6. Troubleshooting

### Issue: "AWS credentials not found"

**Cause**: Environment variables not loaded or incorrect

**Solutions**:
1. Check `.env` file exists and has correct values
2. Restart your terminal/IDE to reload environment
3. If using Docker: `docker-compose down && docker-compose up -d`
4. Verify: `echo $AWS_ACCESS_KEY_ID`

### Issue: "Bedrock access denied"

**Cause**: Model access not enabled

**Solutions**:
1. Go to AWS Bedrock console
2. Click **Model access**
3. Enable **Titan Text Embeddings**
4. Wait 1-2 minutes for activation

### Issue: "Pinecone unauthorized"

**Cause**: Invalid API key or index doesn't exist

**Solutions**:
1. Check API key in `.env` matches Pinecone dashboard
2. Verify index name is exactly `resume-embeddings`
3. Check index is in "Ready" state in Pinecone dashboard

### Issue: "S3 bucket not found"

**Cause**: Bucket name incorrect or doesn't exist

**Solutions**:
1. Check bucket name in AWS S3 console
2. Ensure bucket is in the same region as specified in `.env`
3. Verify IAM user has S3 access permissions

### Issue: "Database connection failed"

**Docker**:
```bash
# Restart PostgreSQL container
docker-compose restart postgres

# Check logs
docker-compose logs postgres

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

**Local PostgreSQL**:
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Start if not running
sudo systemctl start postgresql

# Check connection
psql -h localhost -U postgres -d resume_screening
```

### Issue: "Port already in use"

**Find what's using the port**:
```bash
# macOS/Linux
lsof -ti:8000  # Replace 8000 with your port
kill -9 $(lsof -ti:8000)  # Kill the process

# Or change port in docker-compose.yml
```

### Issue: "npm install fails"

```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### Issue: "Python packages not installing"

```bash
# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Try with --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

### Issue: "Docker won't start"

**macOS**: 
- Open Docker Desktop app
- Check for updates
- Restart Docker Desktop

**Linux**:
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

---

## Verification Checklist

Once setup is complete, verify everything works:

- [ ] ✅ AWS credentials configured
- [ ] ✅ Bedrock access enabled and working
- [ ] ✅ S3 bucket created and accessible
- [ ] ✅ Pinecone account created
- [ ] ✅ Pinecone index created (1536 dimensions, cosine metric)
- [ ] ✅ Docker containers running (`docker-compose ps`)
- [ ] ✅ Backend accessible at http://localhost:8000
- [ ] ✅ Frontend accessible at http://localhost:3000
- [ ] ✅ Database connected (check backend logs)
- [ ] ✅ API docs accessible at http://localhost:8000/api/v1/docs

### Quick Test

```bash
# Test health endpoint
curl http://localhost:8000/api/v1/health

# Expected response:
# {"status":"healthy","timestamp":"..."}
```

---

## Next Steps

1. **Create your first job** via the web interface
2. **Upload a test resume** to see the screening in action
3. **Explore the API docs** at http://localhost:8000/api/v1/docs
4. **Check analytics** at http://localhost:3000/analytics
5. **Read the full README** for advanced features

---

## Getting Help

- **Documentation**: Check README.md and other .md files
- **API Issues**: Check http://localhost:8000/api/v1/docs
- **Logs**: `docker-compose logs -f`
- **GitHub Issues**: Report bugs and ask questions

---

**🎉 Congratulations! Your Resume Screening System is now set up!**

