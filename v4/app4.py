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
            logger.info(f"Extracted text length: {len(text)}")
            
            system_prompt = """You are an expert CV and Resume Analyzer with deep experience in HR, recruitment, and career development. 
            Your task is to perform a comprehensive analysis of the provided CV/Resume with these specific objectives:

            OUTPUT FORMAT:
            Provide a detailed JSON response with the following structure. Be extremely precise with the JSON format:

            {
                "contact_info": {
                    "full_name": string,
                    "email": string,
                    "phone": string,
                    "location": string,
                    "linkedin": string,
                    "website": string,
                    "other_profiles": [string]
                },
                "professional_summary": {
                    "title": string,
                    "seniority_level": string,  // Junior/Mid/Senior/Lead/Executive
                    "industry_focus": [string],
                    "career_highlights": [string],
                    "years_of_experience": float
                },
                "experiences": [
                    {
                        "company": {
                            "name": string,
                            "industry": string,
                            "location": string
                        },
                        "position": string,
                        "start_date": string,
                        "end_date": string,
                        "duration": float,  // in years
                        "is_current": boolean,
                        "responsibilities": [string],
                        "achievements": [
                            {
                                "description": string,
                                "impact": string,
                                "metrics": {
                                    "value": number,
                                    "unit": string,
                                    "change_type": string  // increase/decrease/absolute
                                }
                            }
                        ],
                        "technologies": [string],
                        "keywords": [string]
                    }
                ],
                "education": [
                    {
                        "institution": string,
                        "degree": string,
                        "field": string,
                        "start_date": string,
                        "end_date": string,
                        "gpa": float,
                        "honors": [string],
                        "relevant_coursework": [string],
                        "projects": [string]
                    }
                ],
                "skills": {
                    "technical": [
                        {
                            "category": string,
                            "skills": [string],
                            "proficiency": string  // Expert/Advanced/Intermediate/Beginner
                        }
                    ],
                    "soft": [string],
                    "languages": [
                        {
                            "language": string,
                            "proficiency": string
                        }
                    ],
                    "certifications": [
                        {
                            "name": string,
                            "issuer": string,
                            "date": string,
                            "expires": string
                        }
                    ]
                },
                "projects": [
                    {
                        "name": string,
                        "description": string,
                        "technologies": [string],
                        "role": string,
                        "impact": string,
                        "url": string
                    }
                ],
                "achievements": {
                    "awards": [
                        {
                            "title": string,
                            "issuer": string,
                            "date": string,
                            "description": string
                        }
                    ],
                    "publications": [
                        {
                            "title": string,
                            "publisher": string,
                            "date": string,
                            "url": string
                        }
                    ],
                    "presentations": [string],
                    "patents": [string]
                },
                "metrics": {
                    "profile_strength": {
                        "score": float,  // 0-1
                        "factors": {
                            "experience_depth": float,
                            "education_quality": float,
                            "skills_relevance": float,
                            "achievements_impact": float,
                            "presentation_quality": float
                        }
                    },
                    "career_progression": {
                        "total_experience": float,
                        "companies_worked": int,
                        "average_tenure": float,
                        "promotion_frequency": float,
                        "industry_changes": int
                    },
                    "skill_analysis": {
                        "technical_diversity": float,
                        "leadership_indicators": float,
                        "communication_skills": float,
                        "domain_expertise": float
                    },
                    "impact_metrics": {
                        "quantified_achievements": int,
                        "team_size_managed": int,
                        "budget_managed": string,
                        "project_scale": string
                    }
                },
                "analysis_summary": {
                    "strengths": [string],
                    "improvement_areas": [string],
                    "unique_selling_points": [string],
                    "career_level_assessment": string,
                    "industry_fit": [string],
                    "next_career_suggestions": [string]
                }
            }

            ANALYSIS GUIDELINES:

            1. Profile Analysis:
            - Identify career level and progression pattern
            - Assess industry specialization and expertise depth
            - Evaluate leadership and management experience
            - Analyze career transitions and growth

            2. Experience Evaluation:
            - Look for quantifiable achievements (numbers, percentages, scales)
            - Identify project scales and team sizes
            - Assess technological proficiency and adaptation
            - Evaluate problem-solving capabilities and innovation

            3. Skills Assessment:
            - Categorize technical skills by domain
            - Identify skill proficiency levels
            - Assess skill relevance to career goals
            - Evaluate soft skills through responsibility descriptions

            4. Impact Measurement:
            - Quantify business impact where possible
            - Assess scope of influence
            - Evaluate leadership and team management
            - Measure project and program success indicators

            5. Quality Scoring:
            - Assess CV completeness (0-1)
            - Evaluate detail quality (0-1)
            - Measure achievement quantification (0-1)
            - Rate presentation and clarity (0-1)

            6. Career Insights:
            - Identify career trajectory
            - Assess industry alignment
            - Evaluate growth potential
            - Suggest development areas

            SPECIAL INSTRUCTIONS:

            1. Language Processing:
            - Handle both English and Turkish text
            - Maintain technical terms in original form
            - Preserve proper nouns and company names
            - Convert date formats consistently

            2. Metric Calculation:
            - Calculate experience durations precisely
            - Convert all monetary values to standardized format
            - Quantify team sizes and project scales
            - Assess impact percentages and growth metrics

            3. Data Validation:
            - Ensure date consistency
            - Verify logical progression
            - Check for gaps in employment
            - Validate achievement metrics

            4. Critical Analysis:
            - Identify unique selling points
            - Assess market positioning
            - Evaluate competitive advantages
            - Suggest improvement areas

            Remember to:
            - Be thorough and precise
            - Maintain the exact JSON structure
            - Include all relevant metrics and scores
            - Provide actionable insights
            - Consider industry context
            - Account for career level expectations

            If certain information is not available, use null or appropriate empty values rather than omitting fields.
            """

            logger.info("Attempting GPT API call...")
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this CV thoroughly:\n\n{text}"}
                ],
                max_tokens=4000,
                temperature=0.2
            )

            logger.info("GPT API call successful")
            analysis = json.loads(response.choices[0].message.content)
            logger.info("Analysis keys: " + str(analysis.keys()))
            
            return analysis

        except Exception as e:
            logger.error(f"Error in GPT analysis: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Full error details: ", exc_info=True)
            
            # Return a detailed default structure
            return {
                "contact_info": {
                    "full_name": "",
                    "email": "",
                    "phone": "",
                    "location": "",
                    "linkedin": "",
                    "website": "",
                    "other_profiles": []
                },
                "professional_summary": {
                    "title": "",
                    "seniority_level": "Unknown",
                    "industry_focus": [],
                    "career_highlights": [],
                    "years_of_experience": 0.0
                },
                "experiences": [],
                "education": [],
                "skills": {
                    "technical": [],
                    "soft": [],
                    "languages": [],
                    "certifications": []
                },
                "projects": [],
                "achievements": {
                    "awards": [],
                    "publications": [],
                    "presentations": [],
                    "patents": []
                },
                "metrics": {
                    "profile_strength": {
                        "score": 0.0,
                        "factors": {
                            "experience_depth": 0.0,
                            "education_quality": 0.0,
                            "skills_relevance": 0.0,
                            "achievements_impact": 0.0,
                            "presentation_quality": 0.0
                        }
                    },
                    "career_progression": {
                        "total_experience": 0.0,
                        "companies_worked": 0,
                        "average_tenure": 0.0,
                        "promotion_frequency": 0.0,
                        "industry_changes": 0
                    },
                    "skill_analysis": {
                        "technical_diversity": 0.0,
                        "leadership_indicators": 0.0,
                        "communication_skills": 0.0,
                        "domain_expertise": 0.0
                    },
                    "impact_metrics": {
                        "quantified_achievements": 0,
                        "team_size_managed": 0,
                        "budget_managed": "",
                        "project_scale": ""
                    }
                },
                "analysis_summary": {
                    "strengths": [],
                    "improvement_areas": [],
                    "unique_selling_points": [],
                    "career_level_assessment": "Unknown",
                    "industry_fit": [],
                    "next_career_suggestions": []
                }

            }
        
        except Exception as e:
            logger.error(f"Error in GPT analysis: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Full error details: ", exc_info=True)
            
            return {
                "contact_info": {
                    "full_name": "",
                    "email": "",
                    "phone": "",
                    "location": "",
                    "linkedin": "",
                    "website": "",
                    "other_profiles": []
                },
                "professional_summary": {
                    "title": "",
                    "seniority_level": "Unknown",
                    "industry_focus": [],
                    "career_highlights": [],
                    "years_of_experience": 0.0
                },
                "experiences": [],
                "education": [],
                "skills": {
                    "technical": [],
                    "soft": [],
                    "languages": []
                },
                "projects": [],
                "achievements": {
                    "awards": [],
                    "publications": [],
                    "presentations": [],
                    "patents": []
                },
                "metrics": {
                    "profile_strength": {
                        "score": 0.5,  # Default middle score
                        "factors": {
                            "experience_depth": 0.0,
                            "education_quality": 0.0,
                            "skills_relevance": 0.0,
                            "achievements_impact": 0.0,
                            "presentation_quality": 0.0
                        }
                    },
                    "career_progression": {
                        "total_experience": 0.0,
                        "companies_worked": 0,
                        "average_tenure": 0.0,
                        "promotion_frequency": 0.0,
                        "industry_changes": 0
                    },
                    "skill_analysis": {
                        "technical_diversity": 0.0,
                        "leadership_indicators": 0.0,
                        "communication_skills": 0.0,
                        "domain_expertise": 0.0
                    }
                },
                "analysis_summary": {
                    "strengths": [],
                    "improvement_areas": [],
                    "unique_selling_points": [],
                    "career_level_assessment": "",
                    "industry_fit": [],
                    "next_career_suggestions": []
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
        result = parser.analyze_file("../../resume-recent-ilker-yoru.pdf")
        print(json.dumps(result, indent=2))

        # Save to file
        with open('cv_analysis-ilker.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    # main()
    pass