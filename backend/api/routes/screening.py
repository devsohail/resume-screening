"""
Screening API routes
Handles resume screening requests
"""

import logging
from typing import List
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

from backend.core.models import (
    ScreeningRequest, ScreeningResult, BatchScreeningRequest,
    ScreeningResultList, ResumeUpload
)
from backend.core.config import settings
from backend.storage.s3_handler import get_s3_handler
from backend.storage.db_handler import get_db_handler
from backend.storage.vector_store import get_vector_store
from backend.ml.embeddings.bedrock_embedder import get_embedder
from backend.ml.classifier.inference import get_inference_engine
from backend.ml.similarity.scorer import create_similarity_scorer
from backend.ml.preprocessing.feature_extractor import get_feature_extractor
from backend.ml.preprocessing.text_cleaner import get_text_cleaner
from backend.ml.hybrid_engine import create_hybrid_engine
from backend.ml.feedback_generator import generate_improvement_feedback, format_feedback_for_candidate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/screening", tags=["Screening"])

# Initialize components (lazy loading)
s3_handler = None
db_handler = None
vector_store = None
embedder = None
hybrid_engine = None


def get_components():
    """Get or initialize screening components"""
    global s3_handler, db_handler, vector_store, embedder, hybrid_engine
    
    if s3_handler is None:
        s3_handler = get_s3_handler()
    if db_handler is None:
        db_handler = get_db_handler()
    if vector_store is None:
        vector_store = get_vector_store()
    if embedder is None:
        embedder = get_embedder()
    if hybrid_engine is None:
        # Initialize hybrid engine
        feature_extractor = get_feature_extractor()
        similarity_scorer = create_similarity_scorer(embedder, feature_extractor)
        
        # Try to load classifier (optional)
        classifier = None
        try:
            classifier = get_inference_engine()
            logger.info("Classifier loaded successfully")
        except Exception as e:
            logger.warning(f"Classifier not available: {e}. Continuing with similarity-only scoring.")
        
        hybrid_engine = create_hybrid_engine(similarity_scorer, classifier)
    
    return s3_handler, db_handler, vector_store, embedder, hybrid_engine


@router.post("/upload-resume", response_model=dict, status_code=status.HTTP_201_CREATED)
async def upload_resume(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    candidate_name: str = Form(None),
    candidate_email: str = Form(None)
):
    """
    Upload resume and perform screening
    
    Args:
        file: Resume file (PDF, DOCX, TXT)
        job_id: Job ID to screen against
        candidate_name: Candidate name (optional)
        candidate_email: Candidate email (optional)
        
    Returns:
        Screening result
    """
    try:
        s3, db, vector, emb, engine = get_components()
        
        # Validate file size
        file_content = await file.read()
        if len(file_content) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {settings.max_file_size_mb}MB limit"
            )
        
        # Generate resume ID
        resume_id = str(uuid4())
        
        # Upload to S3
        s3_path = s3.upload_resume(resume_id, file_content, file.filename)
        logger.info(f"Uploaded resume to {s3_path}")
        
        # Extract text
        text = s3.extract_text_from_file(file_content, file.filename)
        
        # Clean text
        cleaner = get_text_cleaner()
        cleaned_text = cleaner.clean_text(text, remove_personal_info=False)
        
        # Extract features
        extractor = get_feature_extractor()
        skills = extractor.extract_skills(cleaned_text)
        education = extractor.extract_education(cleaned_text)
        experience_years = cleaner.extract_years_of_experience(text)
        
        # Save resume to database
        resume_data = {
            'id': resume_id,
            'candidate_name': candidate_name,
            'candidate_email': candidate_email,
            'file_path': s3_path,
            'extracted_text': cleaned_text,
            'skills': skills,
            'experience_years': experience_years,
            'education': ', '.join(education)
        }
        db.create_resume(resume_data)
        
        # Get job details
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        # Generate embeddings (optional - skip if AWS not configured)
        resume_embedding = None
        job_embedding = None
        try:
            resume_embedding = await emb.embed_text(cleaned_text)
            job_embedding = await emb.embed_text(job.description)
            
            # Store in vector database (if available)
            if vector and hasattr(vector, 'enabled') and vector.enabled:
                vector.upsert_resume(
                    resume_id=resume_id,
                    embedding=resume_embedding,
                    metadata={
                        'candidate_name': candidate_name,
                        'skills': skills,
                        'experience_years': experience_years
                    }
                )
        except Exception as e:
            logger.warning(f"Embeddings/vector store not available: {e}")
        
        # Perform screening
        screening_result = await engine.screen_candidate(
            resume_id=resume_id,
            job_id=job_id,
            resume_text=cleaned_text,
            job_text=job.description,
            resume_embedding=resume_embedding,
            job_embedding=job_embedding,
            resume_skills=skills,
            required_skills=job.required_skills,
            preferred_skills=job.preferred_skills,
            resume_experience=experience_years,
            min_experience=job.min_experience_years,
            max_experience=job.max_experience_years,
            resume_education=education,
            job_education=[job.education_requirements] if job.education_requirements else []
        )
        
        # Generate improvement feedback
        feedback = generate_improvement_feedback(
            decision=screening_result.decision.value if screening_result.decision else 'reject',
            final_score=screening_result.final_score,
            matched_skills=screening_result.matched_skills or [],
            missing_skills=screening_result.missing_skills or [],
            required_skills=job.required_skills or [],
            preferred_skills=job.preferred_skills or [],
            candidate_experience=experience_years,
            required_experience=job.min_experience_years,
            similarity_scores={
                'skills_match': screening_result.similarity_result.skills_match if screening_result.similarity_result else 0,
                'experience_match': screening_result.similarity_result.experience_match if screening_result.similarity_result else 0,
                'semantic_similarity': screening_result.similarity_result.semantic_similarity if screening_result.similarity_result else 0
            }
        )
        
        feedback_text = format_feedback_for_candidate(feedback)
        
        # Save screening result to database
        result_data = {
            'id': screening_result.id,
            'resume_id': resume_id,
            'job_id': job_id,
            'status': screening_result.status.value,
            'decision': screening_result.decision.value if screening_result.decision else None,
            'final_score': screening_result.final_score,
            'similarity_overall_score': screening_result.similarity_result.overall_score if screening_result.similarity_result else None,
            'similarity_skills_match': screening_result.similarity_result.skills_match if screening_result.similarity_result else None,
            'classifier_probability': screening_result.classifier_result.probability if screening_result.classifier_result else None,
            'explanation': screening_result.explanation,
            'matched_skills': screening_result.matched_skills,
            'missing_skills': screening_result.missing_skills,
            'processing_time_ms': screening_result.processing_time_ms,
            'completed_at': screening_result.completed_at
        }
        db.create_screening_result(result_data)
        
        logger.info(f"Screening completed for {resume_id}: {screening_result.decision}")
        
        return {
            'resume_id': resume_id,
            'screening_result': screening_result,
            'feedback': feedback,
            'feedback_text': feedback_text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and screening failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{job_id}", response_model=ScreeningResultList)
async def get_screening_results(
    job_id: str,
    page: int = 1,
    page_size: int = 20
):
    """
    Get screening results for a job
    
    Args:
        job_id: Job ID
        page: Page number
        page_size: Results per page
        
    Returns:
        List of screening results
    """
    try:
        _, db, _, _, _ = get_components()
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Get results from database
        results = db.list_screening_results(
            job_id=job_id,
            limit=page_size,
            offset=offset
        )
        
        # Count total
        all_results = db.list_screening_results(job_id=job_id, limit=10000)
        total = len(all_results)
        
        # Convert to response models
        result_models = []
        for result in results:
            # Build SimilarityScore if available
            similarity_score = None
            if result.similarity_overall_score:
                from backend.core.models import SimilarityScore
                similarity_score = SimilarityScore(
                    overall_score=result.similarity_overall_score,
                    skills_match=result.similarity_skills_match or 0,
                    experience_match=result.similarity_experience_match or 0,
                    education_match=result.similarity_education_match or 0,
                    semantic_similarity=result.similarity_semantic or 0
                )
            
            # Build ClassifierResult if available
            classifier_result = None
            if result.classifier_probability:
                from backend.core.models import ClassifierResult, ScreeningDecision
                classifier_result = ClassifierResult(
                    probability=result.classifier_probability,
                    prediction=ScreeningDecision(result.decision),
                    confidence=result.classifier_confidence or 0
                )
            
            result_model = ScreeningResult(
                id=result.id,
                resume_id=result.resume_id,
                job_id=result.job_id,
                status=result.status,
                decision=result.decision,
                final_score=result.final_score,
                similarity_result=similarity_score,
                classifier_result=classifier_result,
                explanation=result.explanation,
                matched_skills=result.matched_skills or [],
                missing_skills=result.missing_skills or [],
                created_at=result.created_at,
                completed_at=result.completed_at,
                processing_time_ms=result.processing_time_ms
            )
            result_models.append(result_model)
        
        return ScreeningResultList(
            results=result_models,
            total=total,
            page=page,
            page_size=page_size,
            has_more=(offset + page_size) < total
        )
        
    except Exception as e:
        logger.error(f"Failed to get screening results: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{result_id}", response_model=ScreeningResult)
async def get_screening_result(result_id: str):
    """Get a specific screening result by ID"""
    try:
        _, db, _, _, _ = get_components()
        
        result = db.get_screening_result(result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Screening result not found")
        
        # Convert to response model (simplified)
        return ScreeningResult(
            id=result.id,
            resume_id=result.resume_id,
            job_id=result.job_id,
            status=result.status,
            decision=result.decision,
            final_score=result.final_score,
            explanation=result.explanation,
            matched_skills=result.matched_skills or [],
            missing_skills=result.missing_skills or [],
            created_at=result.created_at,
            completed_at=result.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get screening result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-screen-job/{job_id}")
async def bulk_screen_job(job_id: str):
    """
    Screen all existing resumes against a specific job
    Useful when creating a new job - automatically screen all candidates
    
    Args:
        job_id: Job ID to screen resumes against
        
    Returns:
        Summary of screening results
    """
    try:
        _, db, _, _, engine = get_components()
        
        # Get job
        job = db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get all resumes
        with db.get_session() as session:
            from backend.storage.db_models import Resume
            resumes = session.query(Resume).all()
            # Detach from session
            for resume in resumes:
                session.expunge(resume)
        
        logger.info(f"Bulk screening {len(resumes)} resumes for job: {job.title}")
        
        results = {
            'job_id': job_id,
            'job_title': job.title,
            'total_resumes': len(resumes),
            'screened': 0,
            'shortlisted': 0,
            'reviewed': 0,
            'rejected': 0,
            'skipped': 0,
            'errors': []
        }
        
        emb, _, vector, _, _ = get_components()
        
        for resume in resumes:
            try:
                # Check if already screened
                existing = db.get_session()
                with existing as session:
                    from backend.storage.db_models import ScreeningResult
                    already_screened = session.query(ScreeningResult).filter(
                        ScreeningResult.resume_id == resume.id,
                        ScreeningResult.job_id == job_id
                    ).first()
                    
                    if already_screened:
                        logger.info(f"Skipping {resume.id} - already screened")
                        results['skipped'] += 1
                        continue
                
                # Generate embeddings
                resume_embedding = None
                job_embedding = None
                try:
                    resume_embedding = await emb.embed_text(resume.extracted_text or '')
                    job_embedding = await emb.embed_text(job.description)
                except Exception as e:
                    logger.warning(f"Embeddings failed for {resume.id}: {e}")
                
                # Screen the candidate
                screening_result = await engine.screen_candidate(
                    resume_id=resume.id,
                    job_id=job_id,
                    resume_text=resume.extracted_text or '',
                    job_text=job.description,
                    resume_embedding=resume_embedding,
                    job_embedding=job_embedding,
                    resume_skills=resume.skills or [],
                    required_skills=job.required_skills or [],
                    preferred_skills=job.preferred_skills or [],
                    resume_experience=resume.experience_years,
                    min_experience=job.min_experience_years,
                    max_experience=job.max_experience_years,
                    resume_education=resume.education,
                    job_education=[job.education_requirements] if job.education_requirements else []
                )
                
                # Save result
                result_data = {
                    'id': screening_result.id,
                    'resume_id': resume.id,
                    'job_id': job_id,
                    'status': screening_result.status.value,
                    'decision': screening_result.decision.value if screening_result.decision else None,
                    'final_score': screening_result.final_score,
                    'similarity_overall_score': screening_result.similarity_result.overall_score if screening_result.similarity_result else None,
                    'similarity_skills_match': screening_result.similarity_result.skills_match if screening_result.similarity_result else None,
                    'classifier_probability': screening_result.classifier_result.probability if screening_result.classifier_result else None,
                    'explanation': screening_result.explanation,
                    'matched_skills': screening_result.matched_skills,
                    'missing_skills': screening_result.missing_skills,
                    'processing_time_ms': screening_result.processing_time_ms,
                    'completed_at': screening_result.completed_at
                }
                db.create_screening_result(result_data)
                
                # Update counts
                results['screened'] += 1
                if screening_result.decision:
                    decision_val = screening_result.decision.value if hasattr(screening_result.decision, 'value') else str(screening_result.decision)
                    if decision_val == 'shortlist':
                        results['shortlisted'] += 1
                    elif decision_val == 'review':
                        results['reviewed'] += 1
                    else:
                        results['rejected'] += 1
                    
                    logger.info(
                        f"Screened {resume.candidate_name or resume.id}: "
                        f"{decision_val} ({screening_result.final_score:.1f})"
                    )
                else:
                    results['rejected'] += 1
                    logger.info(
                        f"Screened {resume.candidate_name or resume.id}: "
                        f"no decision ({screening_result.final_score:.1f})"
                    )
                
            except Exception as e:
                logger.error(f"Failed to screen resume {resume.id}: {e}", exc_info=True)
                results['errors'].append({
                    'resume_id': resume.id,
                    'error': str(e)
                })
        
        logger.info(
            f"Bulk screening complete: {results['screened']} screened, "
            f"{results['shortlisted']} shortlisted, {results['rejected']} rejected"
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk screening failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/{result_id}")
async def get_candidate_feedback(result_id: str):
    """
    Get improvement feedback for a specific screening result
    Tells candidate what they need to improve
    
    Args:
        result_id: Screening result ID
        
    Returns:
        Detailed feedback with improvements and recommendations
    """
    try:
        _, db, _, _, _ = get_components()
        
        # Get screening result
        result = db.get_screening_result(result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Screening result not found")
        
        # Get job and resume for context
        job = db.get_job(result.job_id)
        
        with db.get_session() as session:
            from backend.storage.db_models import Resume
            resume = session.query(Resume).filter(Resume.id == result.resume_id).first()
            session.expunge(resume)
        
        # Generate feedback
        feedback = generate_improvement_feedback(
            decision=result.decision.value if result.decision else 'reject',
            final_score=result.final_score or 0,
            matched_skills=result.matched_skills or [],
            missing_skills=result.missing_skills or [],
            required_skills=job.required_skills or [],
            preferred_skills=job.preferred_skills or [],
            candidate_experience=resume.experience_years,
            required_experience=job.min_experience_years,
            similarity_scores={
                'skills_match': result.similarity_skills_match or 0,
                'experience_match': result.similarity_experience_match or 0,
                'semantic_similarity': result.similarity_semantic or 0
            }
        )
        
        feedback_text = format_feedback_for_candidate(feedback)
        
        return {
            'result_id': result_id,
            'job_title': job.title,
            'candidate_name': resume.candidate_name,
            'decision': result.decision.value if result.decision else None,
            'score': result.final_score,
            'feedback': feedback,
            'feedback_text': feedback_text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

