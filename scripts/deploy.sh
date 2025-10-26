#!/bin/bash

# Deployment script for Resume Screening System
set -e

echo "Deploying Resume Screening System..."

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: AWS credentials not configured"
    exit 1
fi

# Build Lambda package
echo "Building Lambda package..."
cd backend
mkdir -p package
pip install -r lambda/requirements_lambda.txt -t package/
cp -r backend package/
cd package && zip -r ../lambda-package.zip .
cd ../..

# Deploy with Terraform
echo "Deploying infrastructure with Terraform..."
cd terraform
terraform init
terraform plan
terraform apply -auto-approve
cd ..

echo "Deployment complete!"

