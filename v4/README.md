# Resume Analyzer v4

This project is a comprehensive CV and Resume Analyzer using OpenAI's GPT-4 and various image processing libraries to extract and analyze information from PDF resumes.

## Features

- **PDF to Image Conversion**: Converts PDF pages to images for OCR processing.
- **OCR Processing**: Uses Tesseract to extract text from images.
- **Text Normalization**: Cleans and normalizes extracted text for better analysis.
- **GPT-4 Analysis**: Analyzes the extracted text using OpenAI's GPT-4 to provide a detailed JSON response.
- **Logging**: Comprehensive logging for debugging and monitoring.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/resume-analyzer.git
    cd resume-analyzer/v4
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    ```bash
    cp .env.example .env
    # Add your OpenAI API key to the .env file
    ```

### requirements.txt
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
unicodedata
opencv-python-headless
numpy
Pillow
dataclasses
```

## Usage

1. Run the main script:
    ```bash
    python main.py
    ```

2. The script will process the specified PDF file and output the analysis as a JSON file.

## Configuration

- **API Key**: Ensure your OpenAI API key is set in the `.env` file.
- **Language**: The parser can handle both English and Turkish text. Set the language parameter when initializing the parser.

## Example

```python
from enhanced_cv_parser_v3 import EnhancedCVParserV3

parser = EnhancedCVParserV3(language='mixed')
result = parser.analyze_file("/path/to/your/resume.pdf")
print(json.dumps(result, indent=2))
```

## Logging

Logs are saved to `cv_parser.log` and also printed to the console. Adjust the logging configuration as needed.

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact [ilker.07yoru@gmail.com](mailto:ilker.07yoru@gmail.com).


