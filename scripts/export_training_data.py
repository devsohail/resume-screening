#!/usr/bin/env python3
"""
Export screening results as training data
Automatically exports resumes with confirmed decisions for training
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.db_handler import get_db_handler
from backend.storage.db_models import ScreeningResult, Resume, Job


def export_training_data(
    output_file: str,
    min_confidence: float = 0.7,
    days_back: int = 30,
    status_filter: str = "completed"
):
    """
    Export screening results as training data
    
    Args:
        output_file: Output CSV file path
        min_confidence: Minimum confidence score to include
        days_back: Only include results from last N days
        status_filter: Only include results with this status
    """
    print(f"📊 Exporting training data...")
    print(f"   Min confidence: {min_confidence}")
    print(f"   Days back: {days_back}")
    print(f"   Status filter: {status_filter}")
    
    db = get_db_handler()
    
    # Get screening results
    with db.get_session() as session:
        query = session.query(ScreeningResult, Resume, Job).join(
            Resume, ScreeningResult.resume_id == Resume.id
        ).join(
            Job, ScreeningResult.job_id == Job.id
        )
        
        # Filter by status
        if status_filter:
            from backend.storage.db_models import ScreeningStatusEnum
            query = query.filter(ScreeningResult.status == status_filter)
        
        # Filter by date
        if days_back:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            query = query.filter(ScreeningResult.created_at >= cutoff_date)
        
        # Filter by decision (must have a decision)
        query = query.filter(ScreeningResult.decision.isnot(None))
        
        results = query.all()
        
        # Detach from session
        for result, resume, job in results:
            session.expunge(result)
            session.expunge(resume)
            session.expunge(job)
    
    print(f"   Found {len(results)} screening results")
    
    # Convert to training format
    training_data = []
    for result, resume, job in results:
        # Convert decision to label
        label = 1 if result.decision.value in ['shortlist', 'review'] else 0
        
        # Only include if confidence is high enough
        confidence = result.final_score / 100 if result.final_score else 0.5
        if confidence < min_confidence and label == 1:
            continue  # Skip low-confidence positives
        
        training_data.append({
            'resume_id': resume.id,
            'job_id': job.id,
            'resume_text': resume.extracted_text or '',
            'job_description': job.description,
            'label': label,
            'candidate_name': resume.candidate_name or '',
            'skills': ','.join(resume.skills) if resume.skills else '',
            'experience_years': resume.experience_years or 0,
            'final_score': result.final_score or 0,
            'decision': result.decision.value,
            'created_at': result.created_at.isoformat()
        })
    
    # Create DataFrame
    df = pd.DataFrame(training_data)
    
    print(f"\n✅ Exported {len(df)} training samples:")
    print(f"   Positive (SHORTLIST/REVIEW): {(df['label'] == 1).sum()}")
    print(f"   Negative (REJECT): {(df['label'] == 0).sum()}")
    
    # Save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n📁 Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export training data from screening results')
    parser.add_argument('--output', default='training_data/auto_exported.csv', help='Output CSV file')
    parser.add_argument('--min-confidence', type=float, default=0.7, help='Minimum confidence')
    parser.add_argument('--days', type=int, default=30, help='Days to look back')
    parser.add_argument('--status', default='completed', help='Status filter')
    
    args = parser.parse_args()
    
    df = export_training_data(
        args.output,
        args.min_confidence,
        args.days,
        args.status
    )
    
    # Check if we have enough data to train
    if len(df) >= 100:
        print(f"\n🎯 You have {len(df)} samples - enough to train a model!")
        print(f"   Run: python scripts/auto_train.py --data {args.output}")
    else:
        print(f"\n⏳ Need more data: {len(df)}/100 samples")
        print(f"   Upload {100 - len(df)} more resumes to enable training")

