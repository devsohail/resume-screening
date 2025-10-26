variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "resumes_bucket" {
  description = "S3 bucket for resumes"
  type        = string
  default     = "resume-screening-resumes"
}

variable "models_bucket" {
  description = "S3 bucket for ML models"
  type        = string
  default     = "resume-screening-models"
}

variable "data_bucket" {
  description = "S3 bucket for training data"
  type        = string
  default     = "resume-screening-data"
}

variable "database_name" {
  description = "RDS database name"
  type        = string
  default     = "resume_screening"
}

variable "master_username" {
  description = "RDS master username"
  type        = string
  sensitive   = true
}

variable "master_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "lambda_function_name" {
  description = "Lambda function name"
  type        = string
  default     = "resume-screening-inference"
}

variable "sagemaker_notebook_name" {
  description = "SageMaker notebook instance name"
  type        = string
  default     = "resume-screening-notebook"
}

variable "sagemaker_endpoint_name" {
  description = "SageMaker endpoint name"
  type        = string
  default     = "resume-classifier-endpoint"
}

