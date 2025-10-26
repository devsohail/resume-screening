terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "resume-screening-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "ResumeScreening"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"
  
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

# S3 Buckets
module "s3" {
  source = "./modules/s3"
  
  environment      = var.environment
  resumes_bucket   = var.resumes_bucket
  models_bucket    = var.models_bucket
  data_bucket      = var.data_bucket
}

# RDS PostgreSQL
module "rds" {
  source = "./modules/rds"
  
  environment          = var.environment
  vpc_id               = module.vpc.vpc_id
  private_subnet_ids   = module.vpc.private_subnet_ids
  database_name        = var.database_name
  master_username      = var.master_username
  master_password      = var.master_password
}

# Lambda Function
module "lambda" {
  source = "./modules/lambda"
  
  environment       = var.environment
  function_name     = var.lambda_function_name
  models_bucket     = module.s3.models_bucket_name
  vpc_id            = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
}

# SageMaker
module "sagemaker" {
  source = "./modules/sagemaker"
  
  environment    = var.environment
  notebook_name  = var.sagemaker_notebook_name
  endpoint_name  = var.sagemaker_endpoint_name
  models_bucket  = module.s3.models_bucket_name
}

# CloudWatch Monitoring
module "monitoring" {
  source = "./modules/cloudwatch"
  
  environment            = var.environment
  lambda_function_name   = module.lambda.function_name
  sagemaker_endpoint_name = module.sagemaker.endpoint_name
}

# Outputs
output "api_gateway_url" {
  value = module.lambda.api_gateway_url
}

output "rds_endpoint" {
  value = module.rds.endpoint
}

output "sagemaker_endpoint" {
  value = module.sagemaker.endpoint_name
}

