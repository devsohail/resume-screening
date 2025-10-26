# S3 Buckets for Resume Screening System

resource "aws_s3_bucket" "resumes" {
  bucket = "${var.resumes_bucket}-${var.environment}"
}

resource "aws_s3_bucket_versioning" "resumes" {
  bucket = aws_s3_bucket.resumes.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "resumes" {
  bucket = aws_s3_bucket.resumes.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "models" {
  bucket = "${var.models_bucket}-${var.environment}"
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket" "data" {
  bucket = "${var.data_bucket}-${var.environment}"
}

output "resumes_bucket_name" {
  value = aws_s3_bucket.resumes.bucket
}

output "models_bucket_name" {
  value = aws_s3_bucket.models.bucket
}

output "data_bucket_name" {
  value = aws_s3_bucket.data.bucket
}

