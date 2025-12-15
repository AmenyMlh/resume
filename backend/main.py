from fastapi import FastAPI, UploadFile
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Fonction pour extraire le texte d'un PDF avec pdfplumber
def extract_text_from_pdf(content: bytes):
    file = io.BytesIO(content)
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Fonction pour extraire le texte d'un fichier Word
def extract_text_from_docx(content: bytes):
    doc = docx.Document(io.BytesIO(content))
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

# Fonction de nettoyage avancée
def clean_text_advanced(text: str):
    text = ' '.join(text.split())  # Supprimer les espaces inutiles
    text = text.replace("\n", " ").replace("\r", "")  # Retirer les sauts de ligne
    text = text.lower()  # Mettre en minuscule
    return text

@app.post("/summarize/")
async def summarize_file(file: UploadFile):
    content = await file.read()

    # Identifier le type de fichier et extraire le texte
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(content)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(content)
    else:
        text = content.decode("utf-8")

    # Nettoyer le texte
    text = clean_text_advanced(text)

    # Limiter le texte à la longueur max du modèle si nécessaire
    max_input_length = 1024
    if len(text) > max_input_length:
        text = text[:max_input_length]

    # Résumer le texte avec des paramètres ajustés
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False, top_k=50)
    
    return {"summary": summary[0]["summary_text"]}

