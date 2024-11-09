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
        """Analyze CV using GPT with enhanced prompting."""
        try:
            system_prompt = """You are an expert CV analyzer. Analyze the provided CV text and extract information in the following JSON format:

            {
                "contact_info": {
                    "email": "string",
                    "phone": "string",
                    "linkedin": "string",
                    "location": "string",
                    "website": "string"
                },
                "experiences": [
                    {
                        "company": "string",
                        "position": "string",
                        "start_date": "string",
                        "end_date": "string",
                        "duration": float,
                        "responsibilities": ["string"],
                        "achievements": ["string"],
                        "technologies": ["string"]
                    }
                ],
                "education": [
                    {
                        "institution": "string",
                        "degree": "string",
                        "field": "string",
                        "start_date": "string",
                        "end_date": "string",
                        "gpa": float,
                        "achievements": ["string"]
                    }
                ],
                "skills": {
                    "technical": ["string"],
                    "soft": ["string"],
                    "domain": ["string"]
                },
                "achievements": {
                    "quantifiable": ["string"],
                    "awards": ["string"],
                    "certifications": ["string"]
                },
                "metrics": {
                    "total_experience": float,
                    "num_companies": int,
                    "num_roles": int,
                    "quality_score": float,
                    "skill_diversity": float,
                    "education_level": string
                }
            }

            Guidelines:
            1. Extract all dates in a consistent format
            2. Convert durations to years (float)
            3. Calculate quality_score (0-1) based on completeness and detail
            4. Include all quantifiable achievements with numbers
            5. Handle both English and Turkish text
            6. Preserve all technical terms exactly as written
            7. Calculate total_experience accurately
            8. Set quality_score based on CV completeness and detail level

            Be thorough and precise in the analysis.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this CV:\n\n{text}"}
                ],
                max_tokens=3000,
                temperature=0.2
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error in GPT analysis: {str(e)}")
            return {
                "contact_info": {},
                "experiences": [],
                "education": [],
                "skills": {"technical": [], "soft": [], "domain": []},
                "achievements": {"quantifiable": [], "awards": [], "certifications": []},
                "metrics": {
                    "total_experience": 0.0,
                    "num_companies": 0,
                    "num_roles": 0,
                    "quality_score": 0.0,
                    "skill_diversity": 0.0,
                    "education_level": "unknown"
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
                    "parser_version": "3.0.0",
                    "language": self.language,
                    "quality_score": analysis["metrics"]["quality_score"]
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
        result = parser.analyze_file("profile.pdf")
        print(json.dumps(result, indent=2))

        # Save to file
        with open('cv_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()