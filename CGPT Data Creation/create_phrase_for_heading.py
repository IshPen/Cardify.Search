'''[
    {"search_term": "licensing good", "database_phrase": "Without licensing, AI erodes trust in information, causes newspaper closure, and leads to mass job loss of journalists—that kills democracy."},
    {"search_term": "AI journalism impact", "database_phrase": "AI erodes search traffic for journalism"},
    // Add more pairs
]'''
import openai
import os
import dotenv

dotenv.load_dotenv()
openai.api_key = dotenv.get_key("OPENAI_API_KEY")

from docx import Document

# Load the .docx file
file_path = dotenv.get_key("DOCX_PATH_FILE")
doc = Document(file_path)

# Dictionary to store headings by their type
headings = {}

# Define a function to determine the type of heading
def get_heading_level(paragraph):
    if paragraph.style.name.startswith('Heading 3'):
        return paragraph.style.name
    return None

# Iterate through paragraphs
for para in doc.paragraphs:
    heading_level = get_heading_level(para)
    if heading_level:
        if heading_level not in headings:
            headings[heading_level] = []
        headings[heading_level].append(para.text)

# Print out the headings by type
for heading_type, texts in headings.items():
    print(texts)
    for text in texts:
        print(f'  - {text}')
headings = texts

data = []
for i in headings:
    system_prompt = r"For this tagline of a debate card, write three to five words that sum it up. For instance, for the tagline 'Data confirms uncertainty is driving investment to other countries', the output could be 'uncertainty drives away investment'."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": i}
    ]
    # print(messages)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # You can use other models like "gpt-3.5-turbo"
        messages=messages
    )
    input_string = response["choices"][0]["message"]["content"]
    print({"search_term": input_string, "database_phrase": i})
    data.append({"search_term": input_string, "database_phrase": i})