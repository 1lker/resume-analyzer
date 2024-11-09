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
#file_path = "../profile.pdf"
cv_text = extract_text_from_pdf(file_path)
json_data = get_cv_json(cv_text)
print(json_data)

# save the json data to a file
with open("ilker_cv_data.json", "w") as file:
    file.write(json_data)

