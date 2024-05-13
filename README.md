# PDF Scanner

This application utilizes OCR capabilities and integrates with OpenAI's API.

## Prerequisites

Before running the application, ensure you have the following installed:
- **Python**: Required to run the Flask server. Download and install from [Python.org](https://www.python.org/downloads/).
- **Tesseract OCR**: Necessary for OCR functionalities. Installation guidelines can be found on [Tesseract's GitHub](https://github.com/tesseract-ocr/tesseract).
- **Poppler**: Used for handling PDF files. Installation instructions are available on [Poppler's website](https://poppler.freedesktop.org/).

## Installation

Follow these steps to get the application running:

1. **Install Dependencies**:
   ``` 
   pip install -r requirements.txt
   ```

2. Configure OpenAI API key
  - Obtain an OpenAI API key from OpenAI.
  - Inside the `keys` folder in the repository root, save the API key inside an `openai.txt` file


## Running the Application
```
python server.py
```

## Usage
```
http://localhost:8081
```
