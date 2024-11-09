# Resume Analyzer v3

This project is a resume analyzer that extracts and analyzes information from PDF resumes using OCR and GPT-4.

## Features

- **OCR Processing**: Converts PDF pages to images and extracts text using Tesseract OCR.
- **Text Normalization**: Cleans and normalizes extracted text for better analysis.
- **GPT-4 Analysis**: Uses OpenAI's GPT-4 to analyze the extracted text and structure it into a detailed JSON format.
- **Logging**: Comprehensive logging for debugging and monitoring.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/resume-analyzer.git
    cd resume-analyzer
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    ```sh
    cp .env.example .env
    # Edit .env to include your OpenAI API key
    ```

### Dependencies

Add the following dependencies to your `requirements.txt` file:
```
openai
pdf2image
pytesseract
python-dotenv
opencv-python-headless
Pillow
numpy
```

## Usage

1. Place the PDF resume you want to analyze in the project directory.

2. Run the main script:
    ```sh
    python main.py
    ```

3. The analysis result will be saved to `cv_analysis.json`.

## Code Overview

### `EnhancedCVParserV3`

- **Initialization**: Loads environment variables and sets up the OpenAI client.
- **Text Normalization**: `_normalize_text` method to clean and normalize text.
- **Image Preprocessing**: `_preprocess_image` method to enhance image quality for OCR.
- **Text Extraction**: `_extract_text_from_pdf` method to convert PDF to images and extract text.
- **GPT Analysis**: `_analyze_with_gpt` method to analyze the text using GPT-4.
- **CV Parsing**: `parse_cv` method to extract and analyze text from a PDF.
- **File Analysis**: `analyze_file` method to analyze a CV file and return structured results.

### `main`

- Initializes the parser and analyzes a sample PDF file.
- Saves the analysis result to a JSON file.

## Logging

Logs are saved to `cv_parser.log` and also printed to the console.

## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact [ilker.07yoru@gmail.com](mailto:ilker.07yoru@gmail.com).
