<<<<<<< HEAD
# CompuviAICase
Compuvi AI Case
=======
# Email Compliance Classification — Case Study
---

## Table of Contents
- [Objective](#objective)
- [Inputs](#inputs)
- [Expected Data Format](#expected-data-format)
- [Tasks](#tasks)
- [Evaluation & Metrics](#evaluation--metrics)
- [Deliverables](#deliverables)
- [Quickstart (Example Commands)](#quickstart-example-commands)
- [Suggested Project Structure](#suggested-project-structure)
- [Engineering Standards (Code Quality)](#engineering-standards-code-quality)
- [Legal / Permissions (Summary)](#legal--permissions-summary)

---

## Objective
From the provided JSONL and PDF documents, the aim is to build an end-to-end NLP workflow:
normalize texts, fine-tune or create your own model for classification, evaluate with sound
metrics, and provide a simple script interface ready for the back-end. We will also enforce rigorous
code-quality standards (clean code practices, appropriate design patterns, and static type
checking) along with consistent linting and basic tests to keep the implementation maintainable
and production-ready.

## Inputs
- **One JSONL sample file** (email-like texts).
- **100–200 PDFs**.
- Example tags: `customer_sharing`, `exclusive_contracts`, `bid_rigging`, `market_allocation`, `abuse_of_dominance`, `price_fixing`, `other_competition_violation`, `clean`.

## Expected Data Format
For each record:
- `text`: cleaned plain text
- `label`: **sub_tag** in given dataset.

> Note: Samples extracted from PDFs will be stored as **JSONL** using the same **text/label** schema.

## Tasks

### 1) Preprocess & Normalize
- Basic cleaning: signature/quote trimming (optional).
- Create stratified Train/Val/Test splits.

### 2) PDF Ingestion
- Extract text from PDFs and label them with the appropriate tag(s).
- Save PDF-derived samples as **JSONL** using the same **text/label** schema.

### 3) Training
- Fine-tune a suitable model or build your own architecture. If you select a larger model, training may take longer; using a smaller model is acceptable.
- Primary metric: **macro-F1**. Also report per-class **precision/recall**.
- Produce a **confusion matrix**.
- Since **false negatives** are costlier than **false positives**, define a simple **thresholding/decision** rule
# Email Compliance Classification — Case Study
---

## Table of Contents
- [Objective](#objective)
- [Inputs](#inputs)
- [Expected Data Format](#expected-data-format)
- [Tasks](#tasks)
- [Evaluation & Metrics](#evaluation--metrics)
- [Deliverables](#deliverables)
- [Quickstart (Example Commands)](#quickstart-example-commands)
- [Suggested Project Structure](#suggested-project-structure)
- [Engineering Standards (Code Quality)](#engineering-standards-code-quality)
- [Legal / Permissions (Summary)](#legal--permissions-summary)

---

## Objective
From the provided JSONL and PDF documents, the aim is to build an end-to-end NLP workflow:
normalize texts, fine-tune or create your own model for classification, evaluate with sound
metrics, and provide a simple script interface ready for the back-end. We will also enforce rigorous
code-quality standards (clean code practices, appropriate design patterns, and static type
checking) along with consistent linting and basic tests to keep the implementation maintainable
and production-ready.

## Inputs
- **One JSONL sample file** (email-like texts).
- **100–200 PDFs**.
- Example tags: `customer_sharing`, `exclusive_contracts`, `bid_rigging`, `market_allocation`, `abuse_of_dominance`, `pric
e_fixing`, `other_competition_violation`, `clean`.                                                                        
## Expected Data Format
For each record:
- `text`: cleaned plain text
- `label`: **sub_tag** in given dataset.

> Note: Samples extracted from PDFs will be stored as **JSONL** using the same **text/label** schema.

## Tasks

### 1) Preprocess & Normalize
- Basic cleaning: signature/quote trimming (optional).
- Create stratified Train/Val/Test splits.

### 2) PDF Ingestion
- Extract text from PDFs and label them with the appropriate tag(s).
- Save PDF-derived samples as **JSONL** using the same **text/label** schema.

### 3) Training
- Fine-tune a suitable model or build your own architecture. If you select a larger model, training may take longer; using
 a smaller model is acceptable.                                                                                           - Primary metric: **macro-F1**. Also report per-class **precision/recall**.
- Produce a **confusion matrix**.
- Since **false negatives** are costlier than **false positives**, define a simple **thresholding/decision** rule
and justify it.

### 4) Evaluation
- Test-set report (macro-F1, accuracy, per-class metrics).
- Confusion matrix figure.
- Brief analysis of class imbalance and potential **data leakage** risks.

### 5) Backend-Ready Simple Interface
- Provide a **Python script** that accepts `subject`/`body` and returns **label** and **confidence score**.

## Deliverables
- Final trained model package, including weights, configuration, and tokenizer files.
- Normalized **JSONL** files (train/val/test) and a **label map**.
- Training outputs: core metrics, **confusion matrix** image, a brief **model card**.
- Summary of hyperparameters and split sizes used.
- Submit the backend Python script you developed, which accepts an email subject and body as
input and returns the predicted label along with its confidence score.
- Preprocessed dataset (Data extracted from PDF’s should also be delivered separately).

## Legal / Permissions
© 2025 Compuvi Inc. All Rights Reserved. The datasets provided (the “Datasets”) and
any models trained, fine-tuned, or derived from them (“Derivative Models”) are the exclusive
property of Compuvi Inc. Access is granted solely for internal evaluation in connection with the
Compuvi case study. You may not copy, share, publish, sell, sublicense, or make the Datasets or
Derivative Models available to any third party, nor use them for training, benchmarking, or any
purpose beyond this case study, and you may not remove notices, attempt re-identification, or
otherwise misuse the content.
You must implement safeguards to prevent unauthorized access and promptly delete all copies
and Derivative Models upon request. Any unauthorized use constitutes infringement and/or
breach and may result in civil and/or criminal liability, including injunctive relief, damages,
attorneys’ fees, and other remedies to the fullest extent permitted by law. By accessing the
Datasets, you agree to these terms. Permissions: legal@compuvi.com.
