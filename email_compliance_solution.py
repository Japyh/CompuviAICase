"""
Email Compliance Classification System
Complete implementation for the case study
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EvalPrediction
)
from transformers import logging as transformers_logging

# PDF processing
import PyPDF2
import pdfplumber
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_warning()
warnings.filterwarnings('ignore')

# Configuration
@dataclass
class Config:
    """Configuration for the email compliance classification system"""
    # Model configuration
    model_name: str = "distilbert-base-uncased"  # Smaller, faster model
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Data configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    model_dir: Path = Path("models")
    
    # Labels
    labels: List[str] = None
    
    def __post_init__(self):
        self.labels = [
            "customer_sharing",
            "exclusive_contracts",
            "bid_rigging",
            "market_allocation",
            "abuse_of_dominance",
            "price_fixing",
            "other_competition_violation",
            "clean"
        ]
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)


class TextPreprocessor:
    """Handles text cleaning and normalization"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove email signatures (common patterns)
        signature_patterns = [
            r'Best regards.*?$',
            r'Sincerely.*?$',
            r'Thanks.*?$',
            r'Sent from.*?$',
            r'--\s*\n.*?$'
        ]
        
        for pattern in signature_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        # Remove email quotes (lines starting with >)
        text = re.sub(r'^>.*?$', '', text, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def extract_email_content(subject: str, body: str) -> str:
        """Combine subject and body into single text"""
        subject = subject or ""
        body = body or ""
        return f"Subject: {subject}\n\n{body}".strip()


class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                logger.error(f"Failed to extract text from {pdf_path}: {e2}")
                return ""
        
        return text
    
    @staticmethod
    def infer_label_from_filename(filename: str) -> str:
        """Infer label from PDF filename"""
        filename_lower = filename.lower()
        
        # Check for keywords in filename
        label_keywords = {
            "price_fixing": ["price", "fixing", "pricing"],
            "bid_rigging": ["bid", "rigging", "tender"],
            "market_allocation": ["market", "allocation", "territory"],
            "customer_sharing": ["customer", "sharing", "client"],
            "exclusive_contracts": ["exclusive", "contract"],
            "abuse_of_dominance": ["abuse", "dominance", "monopoly"],
            "other_competition_violation": ["competition", "violation", "antitrust"],
            "clean": ["clean", "compliant", "normal"]
        }
        
        for label, keywords in label_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                return label
        
        # Default to clean if no violations detected
        return "clean"
    
    @classmethod
    def process_pdfs(cls, pdf_dir: Path, output_path: Path) -> List[Dict]:
        """Process all PDFs in directory and save as JSONL"""
        samples = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            text = cls.extract_text_from_pdf(pdf_path)
            if text:
                cleaned_text = TextPreprocessor.clean_text(text)
                label = cls.infer_label_from_filename(pdf_path.stem)
                
                samples.append({
                    "text": cleaned_text[:5000],  # Limit text length
                    "label": label,
                    "source": pdf_path.name
                })
        
        # Save to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(samples)} PDF samples to {output_path}")
        return samples


class ComplianceDataset(Dataset):
    """PyTorch dataset for compliance classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataProcessor:
    """Handles data loading and splitting"""
    
    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(config.labels)
    
    def load_jsonl(self, path: Path) -> pd.DataFrame:
        """Load JSONL file into DataFrame"""
        data = []
        # Open with utf-8 and replace errors to be robust to mixed encodings
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    # skip malformed lines
                    continue
        return pd.DataFrame(data)
    
    def prepare_data(self, jsonl_path: Path, pdf_samples_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split data into train/val/test sets"""
        
        # Load main JSONL data
        df = self.load_jsonl(jsonl_path)
        
        # Load PDF samples if available
        if pdf_samples_path and pdf_samples_path.exists():
            pdf_df = self.load_jsonl(pdf_samples_path)
            df = pd.concat([df, pdf_df], ignore_index=True)
        
        # Clean texts
        df['text'] = df['text'].apply(TextPreprocessor.clean_text)
        
        # Normalize labels: map any unseen label to 'other_competition_violation'
        df['label'] = df['label'].fillna('clean')
        valid_labels = set(self.config.labels)
        df['label'] = df['label'].apply(lambda x: x if x in valid_labels else 'other_competition_violation')

        # Encode labels
        df['label_encoded'] = self.label_encoder.transform(df['label'])
        
        # Stratified split
        X = df['text'].values
        y = df['label_encoded'].values
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_ratio,
            random_state=self.config.random_seed, stratify=y
        )
        
        # Second split: train vs val
        val_ratio_adjusted = self.config.val_ratio / (1 - self.config.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adjusted,
            random_state=self.config.random_seed, stratify=y_temp
        )
        
        # Create DataFrames
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        val_df = pd.DataFrame({'text': X_val, 'label': y_val})
        test_df = pd.DataFrame({'text': X_test, 'label': y_test})
        
        # Save splits
        self._save_splits(train_df, val_df, test_df)
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save data splits to JSONL files"""
        for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
            output_path = self.config.data_dir / f'{name}.jsonl'
            df.to_json(output_path, orient='records', lines=True)
            logger.info(f"Saved {name} split to {output_path}")
        
        # Save label map
        label_map = {i: label for i, label in enumerate(self.config.labels)}
        with open(self.config.data_dir / 'label_map.json', 'w') as f:
            json.dump(label_map, f, indent=2)


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=len(config.labels),
            ignore_mismatched_sizes=True
        )
    
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute metrics for evaluation"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Train the model"""
        
        # Create datasets
        train_dataset = ComplianceDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            self.config.max_length
        )
        
        val_dataset = ComplianceDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer,
            self.config.max_length
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.model_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=str(self.config.output_dir / 'logs'),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        self.save_model()
        
        logger.info("Training completed!")
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model on test set"""
        test_dataset = ComplianceDataset(
            test_df['text'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer,
            self.config.max_length
        )
        
        # Get predictions
        predictions = []
        true_labels = []
        confidence_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for item in tqdm(test_dataset, desc="Evaluating"):
                inputs = {k: v.unsqueeze(0).to(self.device) for k, v in item.items() if k != 'labels'}
                outputs = self.model(**inputs)
                
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                
                predictions.append(pred.cpu().numpy()[0])
                true_labels.append(item['labels'].numpy())
                confidence_scores.append(probs.cpu().numpy()[0])
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        macro_f1 = precision_recall_fscore_support(
            true_labels, predictions, average='macro'
        )[2]
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Generate report
        report = classification_report(
            true_labels, predictions,
            target_names=self.config.labels,
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_metrics': {
                label: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': int(support[i])
                }
                for i, label in enumerate(self.config.labels)
            },
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels,
            'confidence_scores': confidence_scores
        }
        
        # Save results
        self._save_evaluation_results(results)
        
        return results
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results and visualizations"""
        
        # Save metrics
        metrics_path = self.config.output_dir / 'evaluation_metrics.json'
        metrics_to_save = {
            'accuracy': float(results['accuracy']),
            'macro_f1': float(results['macro_f1']),
            'per_class_metrics': results['per_class_metrics']
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.config.labels,
            yticklabels=self.config.labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        # Save classification report
        report_path = self.config.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(classification_report(
                results['true_labels'],
                results['predictions'],
                target_names=self.config.labels
            ))
        
        logger.info(f"Evaluation results saved to {self.config.output_dir}")
    
    def save_model(self):
        """Save model and tokenizer"""
        model_path = self.config.model_dir / 'final_model'
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load saved model"""
        model_path = self.config.model_dir / 'final_model'
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()


class ComplianceClassifier:
    """Production-ready interface for email compliance classification"""
    
    def __init__(self, model_path: Path, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Decision threshold (adjust based on cost of false negatives)
        self.violation_threshold = 0.3  # Lower threshold for violation detection
    
    def predict(self, subject: str, body: str) -> Dict[str, Any]:
        """
        Classify email and return label with confidence score
        
        Args:
            subject: Email subject line
            body: Email body text
            
        Returns:
            Dictionary with 'label', 'confidence', and 'all_scores'
        """
        # Combine and clean text
        text = TextPreprocessor.extract_email_content(subject, body)
        text = TextPreprocessor.clean_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        # Get predicted class and confidence
        confidence, predicted_idx = torch.max(probs, dim=-1)
        predicted_label = self.config.labels[predicted_idx.item()]
        confidence_score = confidence.item()
        
        # Apply decision rule for violations
        # If any violation class has probability > threshold, flag it
        violation_labels = [l for l in self.config.labels if l != 'clean']
        violation_indices = [self.config.labels.index(l) for l in violation_labels]
        
        max_violation_prob = max([probs[0, idx].item() for idx in violation_indices])
        
        if max_violation_prob > self.violation_threshold and predicted_label == 'clean':
            # Override clean prediction if violation probability is high
            violation_idx = violation_indices[np.argmax([probs[0, idx].item() for idx in violation_indices])]
            predicted_label = self.config.labels[violation_idx]
            confidence_score = probs[0, violation_idx].item()
        
        # Prepare all scores
        all_scores = {
            label: float(probs[0, i].item())
            for i, label in enumerate(self.config.labels)
        }
        
        return {
            'label': predicted_label,
            'confidence': float(confidence_score),
            'all_scores': all_scores,
            'flagged': predicted_label != 'clean'
        }


def main():
    """Main execution function"""
    
    # Initialize configuration
    config = Config()
    
    # Step 1: Process PDFs (if PDF directory exists)
    pdf_dir = Path("pdfs")  # Adjust path as needed
    if pdf_dir.exists():
        pdf_samples_path = config.data_dir / "pdf_samples.jsonl"
        PDFProcessor.process_pdfs(pdf_dir, pdf_samples_path)
    else:
        pdf_samples_path = None
        logger.warning("PDF directory not found. Skipping PDF processing.")
    
    # Step 2: Load and prepare data
    jsonl_path = Path("email_data.jsonl")  # Adjust path as needed
    if not jsonl_path.exists():
        logger.error(f"JSONL file not found at {jsonl_path}")
        return
    
    processor = DataProcessor(config)
    train_df, val_df, test_df = processor.prepare_data(jsonl_path, pdf_samples_path)
    
    # Step 3: Train model
    trainer = ModelTrainer(config)
    trainer.train(train_df, val_df)
    
    # Step 4: Evaluate model
    results = trainer.evaluate(test_df)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1-Score: {results['macro_f1']:.4f}")
    print("\nPer-Class Metrics:")
    for label, metrics in results['per_class_metrics'].items():
        print(f"\n{label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Support: {metrics['support']}")
    
    # Step 5: Test the production interface
    print("\n" + "="*50)
    print("TESTING PRODUCTION INTERFACE")
    print("="*50)
    
    classifier = ComplianceClassifier(config.model_dir / 'final_model', config)
    
    # Test examples
    test_examples = [
        {
            "subject": "Meeting about pricing strategy",
            "body": "Let's discuss how we can coordinate our prices with competitors to maximize profits."
        },
        {
            "subject": "Quarterly Report",
            "body": "Please find attached our Q3 financial results and market analysis."
        }
    ]
    
    for i, example in enumerate(test_examples, 1):
        result = classifier.predict(example["subject"], example["body"])
        print(f"\nExample {i}:")
        print(f"  Subject: {example['subject']}")
        print(f"  Predicted Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Flagged: {result['flagged']}")
    
    print("\n" + "="*50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"All deliverables saved to:")
    print(f"  - Model: {config.model_dir}")
    print(f"  - Data: {config.data_dir}")
    print(f"  - Results: {config.output_dir}")


if __name__ == "__main__":
    main()
