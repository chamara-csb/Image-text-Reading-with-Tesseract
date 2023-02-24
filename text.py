import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\\tesseract.exe"

# Open the PDF file
with open('Sunday Times (Sri Lanka) - Hitad.pdf', 'rb') as f:
    pdf = f.read()

# Convert the PDF to a PIL Image object
image = Image.open(pdf)

# Use Tesseract to extract the text from the image
text = pytesseract.image_to_string(image)

print(text)