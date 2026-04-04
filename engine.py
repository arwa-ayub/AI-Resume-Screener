import fitz  # PyMuPDF
import re
import torch
from sentence_transformers import SentenceTransformer, util

class ResumeEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the Transformer model."""
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load AI model: {e}")

    def extract_text(self, pdf_file):
        """Extract text from PDF with basic error handling."""
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            if not text.strip():
                return None
            return text
        except Exception:
            return None

    def segment_text(self, text):
        """
        Simple heuristic to split resume into sections.
        In a real prod app, we'd use a dedicated NER model.
        """
        sections = {
            "skills": "",
            "experience": "",
            "education": ""
        }
        
        # Basic regex to find headers
        lines = text.split('\n')
        current_section = "skills" # Default
        
        for line in lines:
            l = line.lower().strip()
            if "experience" in l or "work" in l or "history" in l:
                current_section = "experience"
            elif "education" in l or "academic" in l:
                current_section = "education"
            elif "skills" in l or "technologies" in l:
                current_section = "skills"
            
            sections[current_section] += line + " "
            
        return sections

    def calculate_match(self, job_desc, resume_text):
        """
        Performs multi-category semantic matching.
        """
        # Segment the resume
        segments = self.segment_text(resume_text)
        
        # Encode Job Description once
        job_embedding = self.model.encode(job_desc, convert_to_tensor=True)
        
        # Calculate scores for each segment
        scores = {}
        for category, content in segments.items():
            if len(content.strip()) < 10:
                scores[category] = 0.0
                continue
                
            content_embedding = self.model.encode(content, convert_to_tensor=True)
            similarity = util.cos_sim(job_embedding, content_embedding).item()
            scores[category] = round(max(0, similarity) * 100, 2)
            
        # Overall weighted score (Skills 50%, Experience 30%, Education 20%)
        overall = (scores['skills'] * 0.5) + (scores['experience'] * 0.3) + (scores['education'] * 0.2)
        
        return {
            "overall": round(overall, 2),
            "breakdown": scores,
            "segments": segments
        }

    def get_semantic_insights(self, job_desc, resume_text):
        """Finds conceptually similar phrases, not just exact words."""
        # This is a placeholder for a more complex cross-attention mechanism
        # For now, we improve over regex by looking for common technical keywords
        keywords = ["python", "java", "machine learning", "ai", "sql", "git", "aws", "docker", "react"]
        found = [k.upper() for k in keywords if k in job_desc.lower() and k in resume_text.lower()]
        return found