"""
S3 storage handler
Manages file uploads, downloads, and operations for resumes and models
"""

import logging
import io
from typing import Optional, List
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import PyPDF2
import docx

from backend.core.config import settings
from backend.core.exceptions import StorageError, FileProcessingError

logger = logging.getLogger(__name__)


class S3Handler:
    """
    S3 operations handler
    Manages resume uploads, downloads, and file processing
    """
    
    def __init__(self):
        """Initialize S3 client"""
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            
            # Use single bucket with folder structure
            self.bucket = settings.s3_bucket_resumes
            self.resume_folder = "resumes/"
            self.model_folder = "models/"
            self.data_folder = "data/"
            
            # Legacy support
            self.resume_bucket = settings.s3_bucket_resumes
            self.model_bucket = settings.s3_bucket_models
            self.data_bucket = settings.s3_bucket_data
            
            logger.info(f"Initialized S3Handler with bucket: {self.bucket}")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise StorageError(f"S3 initialization failed: {e}")
    
    def upload_file(
        self,
        file_content: bytes,
        bucket: str,
        key: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload file to S3
        
        Args:
            file_content: File content as bytes
            bucket: S3 bucket name
            key: S3 object key
            content_type: MIME content type
            
        Returns:
            S3 path (s3://bucket/key)
        """
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=file_content,
                **extra_args
            )
            
            s3_path = f"s3://{bucket}/{key}"
            logger.info(f"Uploaded file to {s3_path}")
            
            return s3_path
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise StorageError(f"File upload failed: {e}")
    
    def download_file(self, bucket: str, key: str) -> bytes:
        """
        Download file from S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            File content as bytes
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()
            
            logger.debug(f"Downloaded file from s3://{bucket}/{key}")
            return content
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise StorageError(f"File not found: s3://{bucket}/{key}")
            else:
                logger.error(f"Failed to download file: {e}")
                raise StorageError(f"File download failed: {e}")
    
    def upload_resume(
        self,
        resume_id: str,
        file_content: bytes,
        filename: str
    ) -> str:
        """
        Upload resume file to resumes/ folder
        
        Args:
            resume_id: Unique resume identifier
            file_content: File content
            filename: Original filename
            
        Returns:
            S3 path
        """
        # Determine content type from extension
        ext = Path(filename).suffix.lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain'
        }
        content_type = content_types.get(ext, 'application/octet-stream')
        
        # Create S3 key with folder structure: resumes/{resume_id}/{filename}
        key = f"{self.resume_folder}{resume_id}/{filename}"
        
        return self.upload_file(file_content, self.bucket, key, content_type)
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_content: PDF file content
            
        Returns:
            Extracted text
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise FileProcessingError(f"PDF text extraction failed: {e}")
    
    def extract_text_from_docx(self, docx_content: bytes) -> str:
        """
        Extract text from DOCX file
        
        Args:
            docx_content: DOCX file content
            
        Returns:
            Extracted text
        """
        try:
            docx_file = io.BytesIO(docx_content)
            doc = docx.Document(docx_file)
            
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract text from DOCX: {e}")
            raise FileProcessingError(f"DOCX text extraction failed: {e}")
    
    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from resume file (supports PDF, DOCX, TXT)
        
        Args:
            file_content: File content
            filename: Original filename
            
        Returns:
            Extracted text
        """
        ext = Path(filename).suffix.lower()
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_content)
        elif ext == '.docx':
            return self.extract_text_from_docx(file_content)
        elif ext == '.txt':
            return file_content.decode('utf-8')
        else:
            raise FileProcessingError(f"Unsupported file format: {ext}")
    
    def delete_file(self, bucket: str, key: str):
        """
        Delete file from S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
        """
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted file: s3://{bucket}/{key}")
            
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            raise StorageError(f"File deletion failed: {e}")
    
    def list_files(self, bucket: str, prefix: str) -> List[str]:
        """
        List files in S3 bucket with prefix
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix
            
        Returns:
            List of object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            keys = [obj['Key'] for obj in response.get('Contents', [])]
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []


# Singleton instance
_s3_handler_instance = None


def get_s3_handler() -> S3Handler:
    """
    Get singleton instance of S3Handler
    
    Returns:
        S3Handler instance
    """
    global _s3_handler_instance
    
    if _s3_handler_instance is None:
        _s3_handler_instance = S3Handler()
    
    return _s3_handler_instance

