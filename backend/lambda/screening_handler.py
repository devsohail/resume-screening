"""
AWS Lambda handler for resume screening
Lightweight inference endpoint for serverless deployment
"""

import json
import logging
import os
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, '/opt/python')  # Lambda layer path

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables for model persistence across invocations
embedder = None
classifier = None
similarity_scorer = None
hybrid_engine = None


def initialize_models():
    """Initialize models on cold start"""
    global embedder, classifier, similarity_scorer, hybrid_engine
    
    try:
        logger.info("Initializing models...")
        
        # Import after path is set
        from backend.ml.embeddings.bedrock_embedder import get_embedder
        from backend.ml.classifier.inference import get_inference_engine
        from backend.ml.similarity.scorer import create_similarity_scorer
        from backend.ml.preprocessing.feature_extractor import get_feature_extractor
        from backend.ml.hybrid_engine import create_hybrid_engine
        
        # Initialize embedder
        embedder = get_embedder()
        logger.info("Embedder initialized")
        
        # Load classifier from S3
        model_path = os.getenv('MODEL_PATH', '/tmp/model.pt')
        if not os.path.exists(model_path):
            # Download model from S3
            import boto3
            s3 = boto3.client('s3')
            bucket = os.getenv('MODEL_BUCKET')
            key = os.getenv('MODEL_KEY')
            s3.download_file(bucket, key, model_path)
            logger.info(f"Downloaded model from s3://{bucket}/{key}")
        
        classifier = get_inference_engine(model_path)
        logger.info("Classifier initialized")
        
        # Initialize similarity scorer
        feature_extractor = get_feature_extractor()
        similarity_scorer = create_similarity_scorer(embedder, feature_extractor)
        logger.info("Similarity scorer initialized")
        
        # Initialize hybrid engine
        hybrid_engine = create_hybrid_engine(similarity_scorer, classifier)
        logger.info("Hybrid engine initialized")
        
        logger.info("All models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        raise


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Lambda handler for screening requests
    
    Args:
        event: API Gateway event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        # Initialize models if needed (cold start)
        global embedder, classifier, similarity_scorer, hybrid_engine
        if hybrid_engine is None:
            initialize_models()
        
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        
        # Extract parameters
        resume_text = body.get('resume_text')
        job_text = body.get('job_text')
        resume_id = body.get('resume_id', 'unknown')
        job_id = body.get('job_id', 'unknown')
        
        if not resume_text or not job_text:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameters: resume_text and job_text'
                })
            }
        
        logger.info(f"Processing screening request for resume={resume_id}, job={job_id}")
        
        # Generate embeddings
        import asyncio
        resume_embedding = asyncio.run(embedder.embed_text(resume_text))
        job_embedding = asyncio.run(embedder.embed_text(job_text))
        
        # Perform screening
        result = asyncio.run(hybrid_engine.screen_candidate(
            resume_id=resume_id,
            job_id=job_id,
            resume_text=resume_text,
            job_text=job_text,
            resume_embedding=resume_embedding,
            job_embedding=job_embedding
        ))
        
        # Convert result to dict
        result_dict = {
            'id': result.id,
            'resume_id': result.resume_id,
            'job_id': result.job_id,
            'decision': result.decision.value if result.decision else None,
            'final_score': result.final_score,
            'explanation': result.explanation,
            'matched_skills': result.matched_skills,
            'missing_skills': result.missing_skills,
            'processing_time_ms': result.processing_time_ms
        }
        
        logger.info(f"Screening completed: {result.decision}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result_dict)
        }
        
    except Exception as e:
        logger.error(f"Lambda handler error: {e}", exc_info=True)
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

