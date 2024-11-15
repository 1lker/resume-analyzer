import openai
from pdf2image import convert_from_path
import pytesseract
import os
from dotenv import load_dotenv
import json
import logging
from typing import Dict, List, Optional, Union, Any
import re
from datetime import datetime
from pathlib import Path
import unicodedata
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cv_parser.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CVAnalysis:
    contact_info: Dict[str, Any]
    experiences: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    skills: Dict[str, List[str]]
    achievements: Dict[str, Any]
    metrics: Dict[str, Any]

class EnhancedCVParserV3:
    def __init__(self, api_key: Optional[str] = None, language: str = 'mixed'):
        """Initialize Enhanced CV Parser V3 with improved configuration."""
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.language = language

    def _normalize_text(self, text: str) -> str:
        """Normalize text with enhanced character handling."""
        # Convert to NFKC form for better compatibility
        text = unicodedata.normalize('NFKC', text)
        
        # Replace problematic characters
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '\u200b': '',  # Zero width space
            '\xa0': ' ',   # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Enhance image quality for better OCR results."""
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        img_cv = cv2.adaptiveThreshold(
            img_cv, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal
        img_cv = cv2.medianBlur(img_cv, 3)
        
        return img_cv

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with enhanced preprocessing and OCR."""
        try:
            temp_dir = Path("temp_images")
            temp_dir.mkdir(exist_ok=True)
            
            # Log the PDF processing
            logger.info(f"Processing PDF: {file_path}")
            
            images = convert_from_path(file_path, dpi=300)
            logger.info(f"Converted PDF to {len(images)} images")
            
            extracted_text = ""
            
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}")
                
                # Enhance image for better OCR
                img_cv = self._preprocess_image(image)
                
                # Configure Tesseract
                lang_param = 'tur+eng' if self.language == 'mixed' else \
                        'tur' if self.language == 'tr' else 'eng'
                
                custom_config = f'--oem 3 --psm 6 -l {lang_param}'
                
                # Extract text
                page_text = pytesseract.image_to_string(
                    img_cv,
                    config=custom_config,
                    lang=lang_param
                )
                
                # Log the extracted text length
                logger.info(f"Page {i+1} extracted text length: {len(page_text)}")
                
                # Normalize and clean text
                page_text = self._normalize_text(page_text)
                extracted_text += page_text + "\n"
            
            # Clean up
            for file in temp_dir.glob("*.png"):
                file.unlink()
            temp_dir.rmdir()
            
            # Log final text length
            logger.info(f"Total extracted text length: {len(extracted_text)}")
            
            # If text is too short, might indicate an extraction problem
            if len(extracted_text.strip()) < 100:
                logger.warning("Extracted text is suspiciously short!")
                
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _analyze_with_gpt(self, text: str) -> Dict:
        """Analyze CV using GPT with optimized prompting."""
        try:
            logger.info(f"Extracted text length: {len(text)}")
            
            system_prompt = """Expert CV Analyzer. Provide complete JSON analysis with these requirements:

    1. MUST analyze ALL experiences listed in CV, not just the most recent
    2. Required fields (never empty/null):
    - contact_info: full_name, email, phone, location
    - professional_summary: title, seniority_level, years_of_experience
    - experiences[]: ALL positions with company, title, dates, responsibilities
    - education: highest degree details
    - skills: technical (min 5), soft (min 3), languages
    - metrics: all scoring fields (0-1 scale)
    - analysis_summary: strengths, improvements, suggestions

    Output Structure:
    {
        "contact_info": {
            "full_name": str,
            "email": str,
            "phone": str,
            "location": str,
            "linkedin": str,
            "website": str,
            "other_profiles": [str]
        },
        "professional_summary": {
            "title": str,
            "seniority_level": str,
            "industry_focus": [str],
            "career_highlights": [str],
            "years_of_experience": float
        },
        "experiences": [
            {
                "company": {
                    "name": str,
                    "industry": str,
                    "location": str
                },
                "position": str,
                "start_date": str,
                "end_date": str,
                "duration": float,
                "is_current": bool,
                "responsibilities": [str],
                "achievements": [
                    {
                        "description": str,
                        "impact": str,
                        "metrics": {"value": num, "unit": str, "change_type": str}
                    }
                ],
                "technologies": [str],
                "keywords": [str]
            }
        ],
        "education": [
            {
                "institution": str,
                "degree": str,
                "field": str,
                "start_date": str,
                "end_date": str,
                "gpa": float,
                "honors": [str]
            }
        ],
        "skills": {
            "technical": [
                {
                    "category": str,
                    "skills": [str],
                    "proficiency": str
                }
            ],
            "soft": [str],
            "languages": [{"language": str, "proficiency": str}]
        },
        "metrics": {
            "profile_strength": {
                "score": float,
                "factors": {
                    "experience_depth": float,
                    "education_quality": float,
                    "skills_relevance": float,
                    "achievements_impact": float,
                    "presentation_quality": float
                }
            }
        },
        "analysis_summary": {
            "strengths": [str],
            "improvement_areas": [str],
            "unique_selling_points": [str],
            "career_level_assessment": str,
            "industry_fit": [str],
            "next_career_suggestions": [str]
        }
    }

    Key Rules:
    - Parse ALL work experiences, not just recent ones
    - Use industry context for missing info
    - Dates as YYYY-MM
    - Scores 0-1 scale
    - Required arrays min 1 item"""

            logger.info("Attempting GPT API call...")
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze ALL experiences and details in this CV:\n\n{text}"}
                ],
                max_tokens=3000,
                temperature=0.1
            )

            logger.info("GPT API call successful")
            analysis = json.loads(response.choices[0].message.content)
            
            # Validate minimum requirements
            if len(analysis.get('experiences', [])) < 1:
                logger.warning("No experiences found in analysis")
                raise ValueError("Analysis must include all work experiences")
                
            self._validate_basic_requirements(analysis)
            
            logger.info(f"Analyzed {len(analysis.get('experiences', []))} experiences")
            return analysis

        except Exception as e:
            logger.error(f"Error in GPT analysis: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Full error details: ", exc_info=True)
            return self._get_default_analysis()

    def _validate_basic_requirements(self, analysis: Dict) -> None:
        """Validate minimum required fields are present."""
        if not analysis.get('experiences'):
            raise ValueError("No experiences found")
            
        required = {
            'contact_info': ['full_name', 'email', 'phone', 'location'],
            'professional_summary': ['title', 'seniority_level', 'years_of_experience'],
            'experiences': ['company', 'position', 'start_date', 'end_date', 'responsibilities'],
            'skills': ['technical', 'soft', 'languages'],
            'analysis_summary': ['strengths', 'improvement_areas', 'next_career_suggestions']
        }
        
        for section, fields in required.items():
            if section not in analysis:
                raise ValueError(f"Missing required section: {section}")
                
            if section == 'experiences':
                for exp in analysis[section]:
                    for field in fields:
                        if field not in exp and field not in exp.get('company', {}):
                            raise ValueError(f"Missing required field in experience: {field}")
            else:
                for field in fields:
                    if field not in analysis[section]:
                        raise ValueError(f"Missing required field: {section}.{field}")

    def _get_default_analysis(self) -> Dict:
        """Return a minimal default analysis structure."""
        return {
            "contact_info": {
                "full_name": "Not specified",
                "email": "Not specified",
                "phone": "Not specified",
                "location": "Not specified"
            },
            "professional_summary": {
                "title": "Not specified",
                "seniority_level": "Unknown",
                "years_of_experience": 0.0
            },
            "experiences": [],
            "education": [],
            "skills": {
                "technical": [],
                "soft": [],
                "languages": []
            },
            "metrics": {
                "profile_strength": {"score": 0.0}
            },
            "analysis_summary": {
                "strengths": ["Not analyzed"],
                "improvement_areas": ["Not analyzed"],
                "next_career_suggestions": ["Not analyzed"]
            }
        }

    def parse_cv(self, file_path: str) -> Dict:
        """Parse CV with enhanced analysis and structured output."""
        try:
            # Extract and normalize text
            text = self._extract_text_from_pdf(file_path)
            text = self._normalize_text(text)
            
            # Get analysis from GPT
            analysis = self._analyze_with_gpt(text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing CV: {str(e)}")
            raise

    def analyze_file(self, file_path: str) -> Dict:
        """Analyze CV file and return structured results."""
        try:
            analysis = self.parse_cv(file_path)
            
            # Add metadata
            result = {
                "metadata": {
                    "filename": Path(file_path).name,
                    "processed_date": datetime.now().isoformat(),
                    "parser_version": "4.0.0",
                    "language": self.language,
                    "quality_score": analysis.get("metrics", {}).get("profile_strength", {}).get("score", 0.5),
                    "analysis_quality": {
                        "text_extraction_success": True,
                        "content_completeness": analysis.get("metrics", {}).get("profile_strength", {}).get("factors", {}).get("presentation_quality", 0.0),
                        "processing_duration": "duration_in_seconds",  # Eklenebilir
                        "confidence_score": analysis.get("metrics", {}).get("profile_strength", {}).get("score", 0.5)
                    }
                },
                "analysis": analysis
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing file: {str(e)}")
            raise

def main():
    try:
        parser = EnhancedCVParserV3(language='mixed')
        result = parser.analyze_file("./funda-hanim.pdf")
        print(json.dumps(result, indent=2))

        # Save to file
        with open('cv_analysis-ilker.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    # main()
    pass
