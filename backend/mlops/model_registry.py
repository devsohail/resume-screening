"""
Model registry for version management
Manages model versions in S3 with metadata
"""

import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

from backend.core.config import settings
from backend.core.exceptions import ModelError, StorageError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model registry using S3 for storage
    Tracks model versions, metadata, and performance metrics
    """
    
    def __init__(self):
        """Initialize model registry"""
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
            self.bucket_name = settings.s3_bucket_models
            self.registry_file = "models/registry.json"
            
            logger.info(f"Initialized ModelRegistry with bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ModelRegistry: {e}")
            raise ModelError(f"Model registry initialization failed: {e}")
    
    def _load_registry(self) -> Dict:
        """Load registry metadata from S3"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self.registry_file
            )
            registry_data = json.loads(response['Body'].read().decode('utf-8'))
            return registry_data
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                # Registry doesn't exist yet, create empty one
                return {"models": [], "active_version": None}
            else:
                logger.error(f"Failed to load registry: {e}")
                raise StorageError(f"Failed to load registry: {e}")
    
    def _save_registry(self, registry_data: Dict):
        """Save registry metadata to S3"""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.registry_file,
                Body=json.dumps(registry_data, indent=2),
                ContentType='application/json'
            )
            logger.debug("Saved registry metadata")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise StorageError(f"Failed to save registry: {e}")
    
    def register_model(
        self,
        model_path: str,
        version: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None,
        set_active: bool = False
    ) -> str:
        """
        Register a new model version
        
        Args:
            model_path: Local path to model file
            version: Version string (e.g., "1.0.0")
            metrics: Model performance metrics
            metadata: Additional metadata
            set_active: Whether to set as active version
            
        Returns:
            S3 path to registered model
        """
        try:
            # Upload model to S3
            s3_key = f"models/classifier/v{version}/model.pt"
            
            with open(model_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=f
                )
            
            logger.info(f"Uploaded model to s3://{self.bucket_name}/{s3_key}")
            
            # Load registry
            registry = self._load_registry()
            
            # Add model entry
            model_entry = {
                "version": version,
                "s3_path": f"s3://{self.bucket_name}/{s3_key}",
                "metrics": metrics,
                "metadata": metadata or {},
                "registered_at": datetime.utcnow().isoformat(),
                "is_active": set_active
            }
            
            # Update registry
            registry["models"].append(model_entry)
            
            if set_active:
                # Deactivate other versions
                for model in registry["models"]:
                    if model["version"] != version:
                        model["is_active"] = False
                registry["active_version"] = version
            
            # Save registry
            self._save_registry(registry)
            
            logger.info(f"Registered model version: {version}")
            return model_entry["s3_path"]
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise ModelError(f"Model registration failed: {e}")
    
    def get_model(self, version: Optional[str] = None) -> Dict:
        """
        Get model information
        
        Args:
            version: Model version (None for active version)
            
        Returns:
            Model metadata
        """
        registry = self._load_registry()
        
        if version is None:
            # Get active version
            version = registry.get("active_version")
            if not version:
                raise ModelError("No active model version found")
        
        # Find model
        for model in registry["models"]:
            if model["version"] == version:
                return model
        
        raise ModelError(f"Model version not found: {version}")
    
    def list_models(self) -> List[Dict]:
        """
        List all registered models
        
        Returns:
            List of model metadata
        """
        registry = self._load_registry()
        return registry["models"]
    
    def set_active_version(self, version: str):
        """
        Set active model version
        
        Args:
            version: Version to activate
        """
        registry = self._load_registry()
        
        # Find and activate version
        found = False
        for model in registry["models"]:
            if model["version"] == version:
                model["is_active"] = True
                found = True
            else:
                model["is_active"] = False
        
        if not found:
            raise ModelError(f"Model version not found: {version}")
        
        registry["active_version"] = version
        self._save_registry(registry)
        
        logger.info(f"Set active model version: {version}")
    
    def download_model(self, version: Optional[str] = None, local_path: str = "./model.pt") -> str:
        """
        Download model from S3
        
        Args:
            version: Model version (None for active)
            local_path: Local path to save model
            
        Returns:
            Local path to downloaded model
        """
        model_info = self.get_model(version)
        s3_path = model_info["s3_path"]
        
        # Parse S3 path
        s3_key = s3_path.replace(f"s3://{self.bucket_name}/", "")
        
        # Download
        self.s3_client.download_file(
            Bucket=self.bucket_name,
            Key=s3_key,
            Filename=local_path
        )
        
        logger.info(f"Downloaded model {version} to {local_path}")
        return local_path


# Singleton instance
_registry_instance = None


def get_model_registry() -> ModelRegistry:
    """
    Get singleton instance of ModelRegistry
    
    Returns:
        ModelRegistry instance
    """
    global _registry_instance
    
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    
    return _registry_instance

