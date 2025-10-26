"""
SageMaker training script
Trains the resume classifier on SageMaker
"""

import argparse
import logging
import os
import json
import numpy as np
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args):
    """
    Main training function
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting training...")
    logger.info(f"Arguments: {args}")
    
    # Import models
    from backend.ml.classifier.model import ResumeClassifier
    from backend.ml.classifier.trainer import ClassifierTrainer, ResumeDataset
    
    # Load training data from S3
    train_data_path = Path(args.train_data)
    logger.info(f"Loading training data from {train_data_path}")
    
    train_data = np.load(train_data_path / 'train_data.npz')
    resume_embeddings = train_data['resume_embeddings']
    job_embeddings = train_data['job_embeddings']
    labels = train_data['labels']
    
    logger.info(f"Loaded {len(labels)} training samples")
    
    # Create dataset
    train_dataset = ResumeDataset(resume_embeddings, job_embeddings, labels)
    
    # Validation split
    val_split = args.val_split
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    logger.info(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create model
    model = ResumeClassifier(
        embedding_dim=args.embedding_dim,
        hidden_dims=[512, 256, 128],
        dropout_rate=args.dropout
    )
    
    logger.info(f"Model parameters: {model.get_model_size()}")
    
    # Create trainer
    trainer = ClassifierTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Train
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        checkpoint_dir="/opt/ml/checkpoints"
    )
    
    # Save final model
    model_path = Path(args.model_dir) / "model.pt"
    model.save_checkpoint(
        str(model_path),
        trainer.optimizer,
        epoch=args.epochs,
        metrics=history
    )
    
    logger.info(f"Saved model to {model_path}")
    
    # Save training history
    history_path = Path(args.output_dir) / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument('--train-data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    # Model parameters
    parser.add_argument('--embedding-dim', type=int, default=1536)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=5)
    
    args = parser.parse_args()
    
    train(args)

