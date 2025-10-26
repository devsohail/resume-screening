"""
Training pipeline for the binary classifier
Handles data loading, training loop, validation, and checkpointing
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from backend.ml.classifier.model import ResumeClassifier
from backend.core.config import settings

logger = logging.getLogger(__name__)


class ResumeDataset(Dataset):
    """PyTorch dataset for resume-job pairs"""
    
    def __init__(
        self,
        resume_embeddings: np.ndarray,
        job_embeddings: np.ndarray,
        labels: np.ndarray
    ):
        """
        Initialize dataset
        
        Args:
            resume_embeddings: Array of resume embeddings (n_samples, embedding_dim)
            job_embeddings: Array of job embeddings (n_samples, embedding_dim)
            labels: Binary labels (n_samples,) - 1 for shortlist, 0 for reject
        """
        self.resume_embeddings = torch.FloatTensor(resume_embeddings)
        self.job_embeddings = torch.FloatTensor(job_embeddings)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        
        assert len(resume_embeddings) == len(job_embeddings) == len(labels), \
            "All inputs must have the same length"
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.resume_embeddings[idx],
            self.job_embeddings[idx],
            self.labels[idx]
        )


class ClassifierTrainer:
    """
    Trainer for ResumeClassifier
    Handles training loop, validation, and model checkpointing
    """
    
    def __init__(
        self,
        model: ResumeClassifier,
        device: str = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer
        
        Args:
            model: ResumeClassifier model
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function (Binary Cross Entropy)
        self.criterion = nn.BCELoss()
        
        # Optimizer (Adam with weight decay)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for resume_emb, job_emb, labels in tqdm(train_loader, desc="Training"):
            # Move to device
            resume_emb = resume_emb.to(self.device)
            job_emb = job_emb.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(resume_emb, job_emb)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for resume_emb, job_emb, labels in tqdm(val_loader, desc="Validating"):
                # Move to device
                resume_emb = resume_emb.to(self.device)
                job_emb = job_emb.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(resume_emb, job_emb)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc = 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        return metrics
    
    def train(
        self,
        train_dataset: ResumeDataset,
        val_dataset: Optional[ResumeDataset] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        checkpoint_dir: str = "./checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Complete training loop with validation and early stopping
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Epochs to wait before early stopping
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if self.device == 'cuda' else False
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['loss']
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_metrics['accuracy'])
                
                logger.info(
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1_score']:.4f}, "
                    f"Val AUC: {val_metrics['auc']:.4f}"
                )
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    import os
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.model.save_checkpoint(
                        f"{checkpoint_dir}/best_model.pt",
                        self.optimizer,
                        epoch,
                        val_metrics
                    )
                    logger.info("Saved best model checkpoint")
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        logger.info("Training completed!")
        return self.history


def train_classifier_from_data(
    resume_embeddings: np.ndarray,
    job_embeddings: np.ndarray,
    labels: np.ndarray,
    val_split: float = 0.2,
    **training_kwargs
) -> Tuple[ResumeClassifier, Dict]:
    """
    Convenience function to train classifier from data
    
    Args:
        resume_embeddings: Resume embeddings
        job_embeddings: Job embeddings
        labels: Binary labels
        val_split: Validation split ratio
        **training_kwargs: Additional arguments for training
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Create dataset
    full_dataset = ResumeDataset(resume_embeddings, job_embeddings, labels)
    
    # Split into train and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create model
    model = ResumeClassifier()
    
    # Create trainer
    trainer = ClassifierTrainer(model)
    
    # Train
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **training_kwargs
    )
    
    return model, history

