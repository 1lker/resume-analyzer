# Resume Analyzer v2

This project is an enhanced CV parser and analyzer that leverages OCR and GPT-3.5 for extracting and analyzing information from resumes. It supports multilingual processing, particularly for English and Turkish.

## Features

- **OCR Processing**: Converts PDF resumes to text using Tesseract OCR with enhanced image preprocessing.
- **Multilingual Support**: Handles both English and Turkish resumes.
- **Regex Patterns**: Extracts key information such as emails, phone numbers, dates, LinkedIn, GitHub profiles, and education details.
- **GPT-3.5 Analysis**: Analyzes the extracted text to provide detailed insights into the CV content.
- **Logging**: Logs the processing steps and errors with UTF-8 encoding.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/1lker/resume-analyzer.git
    cd resume-analyzer/v2
    ```
    2. Install the required dependencies:
        ```bash
        pip install -r requirements.txt
        ```

        Ensure your `requirements.txt` includes the following dependencies:
        ```
        openai
        pdf2image
        pytesseract
        python-dotenv
        json
        logging
        typing
        re
        datetime
        pathlib
        concurrent.futures
        numpy
        collections
        rake-nltk
        nltk
        textblob
        unicodedata
        pillow
        opencv-python-headless
        ```

3. Download the required NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your OpenAI API key:
        ```
        OPENAI_API_KEY=your_openai_api_key
        ```

## Usage

1. Run the script to analyze a CV:
    ```bash
    python cv_parser.py
    ```

2. The script will process the PDF, extract text, analyze it using GPT-3.5, and save the results in a JSON file.

## Example

```python
if __name__ == "__main__":
    cv_parser = EnhancedCVParser()
    cv_text = cv_parser.extract_text_from_pdf('../profile.pdf')
    result = cv_parser.analyze_cv_with_gpt(cv_text)
    cv_parser.save_results(result, 'cv_analysis.json')
    logger.info("CV analysis completed successfully")
```

## Logging

Logs are saved in `cv_parser.log` with detailed information about each processing step.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact [ilker.07yoru@gmail.com](mailto:ilker.07yoru@gmail.com).
