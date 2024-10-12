from docx import Document
import dotenv

dotenv.load_dotenv()
# Load the .docx file
file_path = dotenv.get_key("DOCX_FILE_PATH")
doc = Document(file_path)

# Dictionary to store headings by their type
headings = {}

# Define a function to determine the type of heading
def get_heading_level(paragraph):
    if paragraph.style.name.startswith('Heading 4'):
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


