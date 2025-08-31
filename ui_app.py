import streamlit as st
import torch
import numpy as np
from pathlib import Path
import json
import tempfile

from backend_classifier import EmailClassifier
from email_compliance_solution import PDFProcessor, TextPreprocessor

st.set_page_config(page_title="Email Compliance Classifier with PDF Support", layout="wide")

st.title("📧 Email Compliance Classifier — Enhanced Demo")
st.write("Classify email text or PDF documents for potential competition law violations.")

# Sidebar for configuration
st.sidebar.title("⚙️ Configuration")
model_path_input = st.sidebar.text_input("Model path (leave empty to use default)", value="")

# Cache the classifier to avoid reloading
@st.cache_resource
def load_classifier(path: str | None):
    return EmailClassifier(model_path=Path(path) if path else None)

# Main interface tabs
tab1, tab2, tab3 = st.tabs(["📧 Email Text", "📄 PDF Upload", "📊 Batch Results"])

with tab1:
    st.header("Email Text Classification")
    
    subject = st.text_input("Subject", value="Meeting about pricing with Competitor X")
    body = st.text_area("Body", value="Hi team, let's align our pricing strategy with Competitor X. We should discuss discounts and territories.")
    
    if st.button("Classify Email", type="primary"):
        try:
            with st.spinner("Loading model and running inference..."):
                classifier = load_classifier(model_path_input if model_path_input.strip() else None)
                
                # Classify email
                result = classifier.classify(subject=subject, body=body)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", result['label'])
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.3f}")
                with col3:
                    st.metric("Risk Level", result['risk_level'])
                
                # Show detailed results
                if result['flagged']:
                    st.error(f"🚨 VIOLATION DETECTED: {result['label']} (Risk: {result['risk_level']})")
                else:
                    st.success("✅ No violations detected")
                
                # Show full probability distribution
                st.subheader("📊 Classification Probabilities")
                
                # Get full probabilities
                full_text = f"Subject: {subject or ''}\n\n{body or ''}"
                cleaned = classifier._clean_text(full_text)
                inputs = classifier.tokenizer(
                    cleaned, truncation=True, padding=True, max_length=512, return_tensors='pt'
                ).to(classifier.device)
                
                with torch.no_grad():
                    outputs = classifier.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                
                prob_dict = {classifier.LABELS[i]: float(probs[i]) for i in range(len(classifier.LABELS))}
                st.bar_chart(prob_dict)
                
                st.json(result)
                
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.header("PDF Document Classification")
    st.write("Upload a PDF document to extract text and classify it for compliance violations.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Classify PDF", type="primary"):
            try:
                with st.spinner("Extracting text and running classification..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)
                    
                    # Load classifier
                    classifier = load_classifier(model_path_input if model_path_input.strip() else None)
                    
                    # Extract text from PDF
                    text = PDFProcessor.extract_text_from_pdf(tmp_path)
                    
                    if not text or len(text.strip()) < 10:
                        st.error("❌ Could not extract meaningful text from the PDF")
                    else:
                        # Clean and truncate text
                        cleaned_text = TextPreprocessor.clean_text(text)
                        truncated_text = cleaned_text[:2000]  # Limit for classification
                        
                        # Classify (treat as email body with empty subject)
                        result = classifier.classify(subject="", body=truncated_text)
                        
                        # Display file info
                        st.info(f"📄 **File:** {uploaded_file.name}")
                        st.info(f"📏 **Text length:** {len(text):,} characters")
                        st.info(f"🧹 **Cleaned length:** {len(cleaned_text):,} characters")
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Prediction", result['label'])
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.3f}")
                        with col3:
                            st.metric("Risk Level", result['risk_level'])
                        
                        if result['flagged']:
                            st.error(f"🚨 VIOLATION DETECTED: {result['label']} (Risk: {result['risk_level']})")
                        else:
                            st.success("✅ No violations detected")
                        
                        # Show text preview
                        with st.expander("📖 Extracted Text Preview"):
                            st.text_area("First 1000 characters:", truncated_text[:1000], height=200)
                        
                        # Show probabilities
                        st.subheader("📊 Classification Probabilities")
                        inputs = classifier.tokenizer(
                            truncated_text, truncation=True, padding=True, max_length=512, return_tensors='pt'
                        ).to(classifier.device)
                        
                        with torch.no_grad():
                            outputs = classifier.model(**inputs)
                            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                        
                        prob_dict = {classifier.LABELS[i]: float(probs[i]) for i in range(len(classifier.LABELS))}
                        st.bar_chart(prob_dict)
                        
                        st.json(result)
                        
                        # Clean up temp file
                        tmp_path.unlink()
                        
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

with tab3:
    st.header("Batch Processing Results")
    st.write("View results from previous batch PDF processing.")
    
    # Check if results file exists
    results_file = Path("pdf_classification_results.json")
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
            
            st.success(f"📊 Found results for {len(batch_results)} PDFs")
            
            # Summary statistics
            successful = [r for r in batch_results if r["status"] == "success"]
            flagged_count = sum(1 for r in successful if r["classification"]["flagged"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", len(batch_results))
            with col2:
                st.metric("Successful", len(successful))
            with col3:
                st.metric("Flagged", flagged_count)
            
            # Classification distribution
            if successful:
                classifications = [r["classification"]["label"] for r in successful]
                from collections import Counter
                label_counts = Counter(classifications)
                
                st.subheader("🏷️ Classification Distribution")
                st.bar_chart(dict(label_counts))
                
                # Detailed results table
                st.subheader("📋 Detailed Results")
                
                # Create a DataFrame for better display
                import pandas as pd
                df_data = []
                for result in successful:
                    df_data.append({
                        "File": result["file"][:50] + "..." if len(result["file"]) > 50 else result["file"],
                        "Classification": result["classification"]["label"],
                        "Confidence": f"{result['classification']['confidence']:.3f}",
                        "Risk Level": result["classification"]["risk_level"],
                        "Flagged": "🚨" if result["classification"]["flagged"] else "✅",
                        "Text Length": f"{result['text_length']:,}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # Download option
                if st.button("📥 Download Full Results"):
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(batch_results, indent=2),
                        file_name="pdf_classification_results.json",
                        mime="application/json"
                    )
        
        except Exception as e:
            st.error(f"Error loading batch results: {e}")
    else:
        st.info("📝 No batch processing results found. Run `python test_pdf_classifier.py` to generate batch results.")
        
        if st.button("🚀 Run Batch Processing Now"):
            try:
                with st.spinner("Running batch processing on sample PDFs..."):
                    import subprocess
                    result = subprocess.run(["python", "test_pdf_classifier.py"], 
                                          capture_output=True, text=True, cwd=".")
                    
                    if result.returncode == 0:
                        st.success("✅ Batch processing completed successfully!")
                        st.rerun()  # Refresh the page to show new results
                    else:
                        st.error(f"❌ Batch processing failed: {result.stderr}")
                        
            except Exception as e:
                st.error(f"Error running batch processing: {e}")

# Footer
st.markdown("---")
st.markdown("🔍 **Email Compliance Classifier** | Detects potential competition law violations using AI")
