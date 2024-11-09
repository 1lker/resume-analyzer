# Resume Analyzer

This project extracts text from a PDF resume and converts it into a structured JSON format using OpenAI's GPT-3.5-turbo model.

## Prerequisites

- Python 3.7+
- OpenAI API key
- Poppler
- Tesseract OCR
- `dotenv` package

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/1lker/resume-analyzer.git
    cd resume-analyzer
    ```

2. Install the required packages:
    ```sh
    pip install openai pdf2image pytesseract python-dotenv
    ```

3. Install Poppler:
    - macOS:
        ```sh
        brew install poppler
        ```
    - Ubuntu:
        ```sh
        sudo apt-get install poppler-utils
        ```

4. Install Tesseract OCR:
    - macOS:
        ```sh
        brew install tesseract
        ```
    - Ubuntu:
        ```sh
        sudo apt-get install tesseract-ocr
        ```

5. Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. Place the PDF resume you want to analyze in the project directory.

2. Update the `file_path` variable in the script to point to your PDF file.

3. Run the script:
    ```sh
    python script_name.py
    ```

4. The extracted JSON data will be saved to `ilker_cv_data.json`.

## Example

```python
import openai
from pdf2image import convert_from_path
import pytesseract
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set. Please set it in the environment or in a .env file.")

client = openai.OpenAI(api_key=openai_api_key)

# Function to extract text from PDF file
def extract_text_from_pdf(file_path):
    images = convert_from_path(file_path, poppler_path='/opt/homebrew/bin')  # Adjust poppler path if needed
    extracted_text = ""
    for image in images:
        extracted_text += pytesseract.image_to_string(image) + "\n"
    return extracted_text

# Function to get detailed JSON from CV text
def get_cv_json(cv_text):
    prompt = f"""
    I am sending you a CV in plain text below. Please analyze and organize the details into a structured JSON format.
    Include the following fields: 'Summary', 'Personal_Info', 'Education', 'Experience','Experience Duration', 'Skills', 'Projects', 
    'Certifications', 'Languages', 'Strengths', 'Weaknesses', 'Volunteer_Activities', 'Teaching_Assistances', 
    'Social_Media_Links', 'Achievements', 'Certifications_Details', 'Project_Impacts', 'Volunteer_Impacts', 
    and 'Weakness'. 
    
    Here is the CV content:
    {cv_text}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.2
    )

    json_response = response.choices[0].message.content
    return json_response

# Example usage
file_path = "../../resume-recent-ilker-yoru.pdf"
cv_text = extract_text_from_pdf(file_path)
json_data = get_cv_json(cv_text)
print(json_data)

# save the json data to a file
with open("ilker_cv_data.json", "w") as file:
    file.write(json_data)
```

## License

This project is licensed under the MIT License.