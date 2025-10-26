#!/usr/bin/env python3
"""
Scheduled training script
Run this via cron or AWS EventBridge to automatically check for new data and train
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.storage.db_handler import get_db_handler
from backend.storage.db_models import ScreeningResult
from backend.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


async def check_and_train(
    min_samples: int = 100,
    force: bool = False,
    dry_run: bool = False
):
    """
    Check if enough new data is available and trigger training
    
    Args:
        min_samples: Minimum reviewed samples needed to trigger training
        force: Force training even without enough samples
        dry_run: Check status only, don't train
    """
    logger.info("=" * 70)
    logger.info("🤖 SCHEDULED TRAINING CHECK")
    logger.info(f"   Time: {datetime.now()}")
    logger.info("=" * 70)
    
    # Check database for reviewed samples
    db = get_db_handler()
    
    with db.get_session() as session:
        # Count total reviewed
        total_reviewed = session.query(ScreeningResult).filter(
            ScreeningResult.human_reviewed == True
        ).count()
        
        # Count needing retraining (corrections)
        needs_retraining = session.query(ScreeningResult).filter(
            ScreeningResult.needs_retraining == True
        ).count()
        
        # Count positives and negatives
        positives = session.query(ScreeningResult).filter(
            ScreeningResult.human_reviewed == True,
            ScreeningResult.human_decision == 'shortlist'
        ).count()
        
        negatives = session.query(ScreeningResult).filter(
            ScreeningResult.human_reviewed == True,
            ScreeningResult.human_decision == 'reject'
        ).count()
    
    logger.info(f"\n📊 Training Data Status:")
    logger.info(f"   Total reviewed: {total_reviewed}")
    logger.info(f"   Positive samples: {positives}")
    logger.info(f"   Negative samples: {negatives}")
    logger.info(f"   Needs retraining: {needs_retraining}")
    logger.info(f"   Threshold: {min_samples}")
    
    # Check if enough data
    ready = total_reviewed >= min_samples
    
    if not ready and not force:
        logger.info(f"\n⏳ NOT READY TO TRAIN")
        logger.info(f"   Need {min_samples - total_reviewed} more reviewed samples")
        logger.info(f"   Current: {total_reviewed}/{min_samples}")
        return False
    
    if dry_run:
        logger.info(f"\n✅ READY TO TRAIN (dry run - not training)")
        logger.info(f"   Would train with {total_reviewed} samples")
        return True
    
    # Trigger training
    logger.info(f"\n🚀 TRIGGERING TRAINING...")
    logger.info(f"   Samples: {total_reviewed}")
    logger.info(f"   Ratio: {positives}:{negatives}")
    
    try:
        import subprocess
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Step 1: Export data
        logger.info("\n📤 Step 1: Exporting training data...")
        export_cmd = [
            sys.executable,
            "scripts/export_training_data.py",
            "--output", f"training_data/scheduled_{timestamp}.csv",
            "--min-confidence", "0.5",  # Lower threshold for human-reviewed data
            "--status", "completed"
        ]
        
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Export failed: {result.stderr}")
            return False
        
        logger.info(result.stdout)
        
        # Step 2: Train model
        logger.info("\n🏋️  Step 2: Training model...")
        train_cmd = [
            sys.executable,
            "scripts/auto_train.py",
            "--data", f"training_data/scheduled_{timestamp}.csv"
        ]
        
        if force:
            train_cmd.append("--force")
        
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return False
        
        logger.info(result.stdout)
        
        # Success!
        logger.info("\n" + "=" * 70)
        logger.info("✅ SCHEDULED TRAINING COMPLETE!")
        logger.info(f"   Trained with: {total_reviewed} samples")
        logger.info(f"   Timestamp: {timestamp}")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Training pipeline failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scheduled model training')
    parser.add_argument('--min-samples', type=int, default=100, help='Minimum samples to train')
    parser.add_argument('--force', action='store_true', help='Force training')
    parser.add_argument('--dry-run', action='store_true', help='Check status only')
    
    args = parser.parse_args()
    
    success = asyncio.run(check_and_train(
        args.min_samples,
        args.force,
        args.dry_run
    ))
    
    sys.exit(0 if success else 1)

