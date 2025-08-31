#!/usr/bin/env python3
"""
Email Compliance Classifier - Backend Interface
Production-ready script for email classification

Usage:
    python backend_classifier.py --subject "Meeting tomorrow" --body "Let's discuss pricing"

Or import and use in your code:
    from backend_classifier import EmailClassifier
    classifier = EmailClassifier()
    result = classifier.classify(subject="...", body="...")
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmailClassifier:
    """
    Production-ready email compliance classifier.

    This classifier identifies potential competition law violations in emails.
    It returns a predicted label, a confidence score, and an associated risk level.
    """

    # Competition violation categories
    LABELS = [
        "customer_sharing",
        "exclusive_contracts",
        "bid_rigging",
        "market_allocation",
        "abuse_of_dominance",
        "price_fixing",
        "other_competition_violation",
        "clean"
    ]

    # Risk levels for each category
    RISK_LEVELS = {
        "price_fixing": "HIGH",
        "bid_rigging": "HIGH",
        "market_allocation": "HIGH",
        "customer_sharing": "MEDIUM",
        "exclusive_contracts": "MEDIUM",
        "abuse_of_dominance": "MEDIUM",
        "other_competition_violation": "LOW",
        "clean": "NONE"
    }

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initializes the classifier.

        Args:
            model_path: Path to the trained model directory.
                        If None, uses the default path './models/final_model'.
        """
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "final_model"

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please ensure the model is trained and saved in the correct directory."
            )

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and tokenizer
        print(f"Loading model from {self.model_path}...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Decision threshold for flagging violations.
        # A lower threshold makes the model more sensitive to potential violations,
        # reducing the risk of costly false negatives.
        self.violation_threshold = 0.3

        print(f"Model loaded successfully. Using device: {self.device}", file=sys.stderr)

    def _clean_text(self, text: str) -> str:
        """
        Cleans and normalizes raw text.

        Args:
            text: The raw text to clean.

        Returns:
            The cleaned text.
        """
        if not text:
            return ""

        # Remove email signatures (common patterns)
        signature_patterns = [
            r'Best regards.*?$', r'Sincerely.*?$', r'Thanks.*?$',
            r'Sent from.*?$', r'--\s*\n.*?$'
        ]
        for pattern in signature_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

        # Remove email quotes (lines starting with >)
        text = re.sub(r'^>.*?$', '', text, flags=re.MULTILINE)

        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def classify(self, subject: str, body: str) -> Dict[str, Any]:
        """
        Classifies an email based on its subject and body.

        Args:
            subject: The email subject line.
            body: The email body text.

        Returns:
            A dictionary containing the predicted label, confidence score, risk level,
            and a boolean indicating if the email was flagged.
        """
        # Combine subject and body and clean the resulting text
        full_text = f"Subject: {subject or ''}\n\n{body or ''}"
        cleaned_text = self._clean_text(full_text)

        # Tokenize the text for the model
        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # Get the primary prediction
        predicted_index = np.argmax(probabilities)
        predicted_label = self.LABELS[predicted_index]
        confidence = probabilities[predicted_index]

        # Apply the decision rule: override 'clean' if any violation is above the threshold
        if predicted_label == 'clean':
            violation_labels = [l for l in self.LABELS if l != 'clean']
            violation_indices = [self.LABELS.index(l) for l in violation_labels]
            violation_probabilities = probabilities[violation_indices]

            if np.any(violation_probabilities > self.violation_threshold):
                # If a violation is detected, find the most likely one
                highest_violation_index = violation_indices[np.argmax(violation_probabilities)]
                predicted_label = self.LABELS[highest_violation_index]
                confidence = probabilities[highest_violation_index]

        risk_level = self.RISK_LEVELS.get(predicted_label, "UNKNOWN")

        return {
            'label': predicted_label,
            'confidence': float(confidence),
            'risk_level': risk_level,
            'flagged': predicted_label != 'clean'
        }


def main():
    """
    Main function to run the script from the command line.
    """
    parser = argparse.ArgumentParser(description="Email Compliance Classifier")
    # Make subject/body optional so the script can accept stdin JSON or prompt interactively
    parser.add_argument("--subject", type=str, required=False, help="Email subject line")
    parser.add_argument("--body", type=str, required=False, help="Email body text")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model directory")
    args = parser.parse_args()

    # If subject/body missing, try to read JSON from stdin (useful for piping)
    subject = args.subject
    body = args.body

    if (subject is None) or (body is None):
        try:
            if not sys.stdin.isatty():
                # Read whole stdin and parse JSON {"subject":..., "body":...}
                raw = sys.stdin.read()
                if raw.strip():
                    payload = json.loads(raw)
                    subject = subject or payload.get("subject")
                    body = body or payload.get("body")
        except Exception:
            # fall through to interactive prompt
            pass

    # If still missing, prompt interactively (keeps backward compatibility)
    if (subject is None) or (body is None):
        try:
            if subject is None:
                subject = input("Enter email subject: ")
            if body is None:
                print("Enter email body (end with Ctrl-D on a new line):")
                # Read multiline body from stdin until EOF
                body_lines = []
                try:
                    while True:
                        line = input()
                        body_lines.append(line)
                except EOFError:
                    pass
                body = "\n".join(body_lines)
        except Exception:
            # If interactive input fails, provide usage and exit
            parser.print_usage(sys.stderr)
            sys.exit(2)

    try:
        model_path = Path(args.model_path) if args.model_path else None
        classifier = EmailClassifier(model_path=model_path)
        result = classifier.classify(subject=subject, body=body)

        # Print result as a JSON string to stdout
        print(json.dumps(result, indent=2))

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
