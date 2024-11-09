import openai
from pdf2image import convert_from_path
import pytesseract
import os
from dotenv import load_dotenv
import json
import logging
from typing import Dict, List, Optional, Union
import re
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import Counter
from rake_nltk import Rake
import nltk
from textblob import TextBlob
import unicodedata
from PIL import Image
import cv2
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cv_parser.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EnhancedCVParser:
    def __init__(self, api_key: Optional[str] = None, language: str = 'mixed'):
        """
        Initialize the Enhanced CV Parser with configuration.
        
        Args:
            api_key (str, optional): OpenAI API key
            language (str): Default language for parsing ('en', 'tr', 'mixed')
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.rake = Rake()
        self.language = language
        
        # Enhanced regex patterns with Turkish character support
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'),
            'date': self._get_date_pattern(),
            'linkedin': re.compile(r'linkedin\.com/in/[\w\-ğüşıöçĞÜŞİÖÇ]+'),
            'github': re.compile(r'github\.com/[\w\-ğüşıöçĞÜŞİÖÇ]+'),
            'education': self._get_education_pattern(),
            'urls': re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*')
        }

    def _get_date_pattern(self) -> re.Pattern:
        """Get date pattern based on language setting."""
        tr_months = r'(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)'
        en_months = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        
        if self.language == 'tr':
            months = tr_months
        elif self.language == 'en':
            months = en_months
        else:
            months = f'(?:{tr_months}|{en_months})'
            
        return re.compile(
            rf'\b{months}\s+\d{{4}}\s*(?:-|–|to)\s*'
            r'(?:Present|Current|Now|Günümüz|Devam Ediyor|'
            rf'{months}\s+\d{{4}})\b',
            re.IGNORECASE
        )

    def _get_education_pattern(self) -> re.Pattern:
        """Get education pattern based on language setting."""
        tr_degrees = r'(?:Lisans|Yüksek Lisans|Doktora|Önlisans|Lise)'
        en_degrees = r'(?:Bachelor|Master|PhD|BSc|MSc|MBA|MD|BS|MS|BA|MA)'
        
        if self.language == 'tr':
            degrees = tr_degrees
        elif self.language == 'en':
            degrees = en_degrees
        else:
            degrees = f'(?:{tr_degrees}|{en_degrees})'
            
        return re.compile(rf'\b{degrees}\b', re.IGNORECASE)

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR results."""
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply image preprocessing
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_cv = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Noise removal
        img_cv = cv2.medianBlur(img_cv, 3)
        
        # Convert back to PIL Image
        return Image.fromarray(img_cv)

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Enhanced text extraction with better handling and preprocessing."""
        try:
            temp_dir = Path("temp_images")
            temp_dir.mkdir(exist_ok=True)
            
            images = convert_from_path(file_path, dpi=300)
            extracted_text = ""
            
            for i, image in enumerate(images):
                # Enhance image quality
                image = self._preprocess_image(image)
                temp_image_path = temp_dir / f"page_{i}.png"
                image.save(temp_image_path, quality=95)
                
                # Configure Tesseract with language support
                lang_param = 'tur+eng' if self.language == 'mixed' else \
                           'tur' if self.language == 'tr' else 'eng'
                custom_config = f'--oem 3 --psm 6 -l {lang_param}'
                
                page_text = pytesseract.image_to_string(
                    image, 
                    config=custom_config,
                    lang=lang_param
                )
                
                # Normalize Unicode characters
                page_text = unicodedata.normalize('NFKC', page_text)
                extracted_text += page_text + "\n"
            
            # Clean up
            for file in temp_dir.glob("*.png"):
                file.unlink()
            temp_dir.rmdir()
            
            # Advanced text preprocessing
            extracted_text = self._clean_text(extracted_text)
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """Advanced text cleaning with special character handling."""
        # Replace common problematic characters
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
        
        # Remove extra whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        return text.strip()

    def analyze_cv_with_gpt(self, text: str) -> Dict:
        """Enhanced CV analysis using GPT with multilingual support."""
        system_prompt = """
        You are a professional CV analyzer. Please analyze the CV in any language 
        (particularly Turkish or English) and provide a detailed analysis in the same 
        language as the CV. Focus on maintaining proper character encoding for 
        special characters and diacritical marks.
        """
        
        user_prompt = f"""
        Please analyze this CV and provide a detailed JSON response with the following 
        sections (use the same language as the CV):
        
        1. Professional_Profile
        2. Skills_Analysis
        3. Experience_Timeline
        4. Education_And_Certifications
        5. Projects_Portfolio
        6. Career_Highlights
        7. Skills_Gap_Analysis
        
        CV Content:
        {text}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2500,
                temperature=0.2
            )
            
            # Parse and validate JSON response
            result = json.loads(response.choices[0].message.content)
            return self._validate_json_encoding(result)
            
        except Exception as e:
            logger.error(f"Error in GPT analysis: {str(e)}")
            return {}

    def _validate_json_encoding(self, data: Union[Dict, List, str]) -> Union[Dict, List, str]:
        """Recursively validate and fix JSON encoding issues."""
        if isinstance(data, dict):
            return {k: self._validate_json_encoding(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._validate_json_encoding(item) for item in data]
        elif isinstance(data, str):
            return unicodedata.normalize('NFKC', data)
        return data

    def save_results(self, result: Dict, output_file: str) -> None:
        """Save results with proper encoding."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)




if __name__ == "__main__":
    cv_parser = EnhancedCVParser()
    cv_text = cv_parser.extract_text_from_pdf('../profile.pdf')
    #cv_text = cv_parser.extract_text_from_pdf('../../resume-recent-ilker-yoru.pdf')
    # Parse CV text using GPT
    result = cv_parser.analyze_cv_with_gpt(cv_text)
    cv_parser.save_results(result, 'cv_analysis-ahmet.json')
    logger.info("CV analysis completed successfully")
