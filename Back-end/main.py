from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch
import PyPDF2
import io
import re

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the question-answering model using Hugging Face Transformers Library and the corresponding tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load a text generation model and tokenizer for questions
text_gen_model_name = "gpt2"
text_gen_model = GPT2LMHeadModel.from_pretrained(text_gen_model_name)  # Change to QA model
text_gen_tokenizer = GPT2Tokenizer.from_pretrained(text_gen_model_name)

#  Pydantic Model that specifies the expected request body structures for the endpoint
class QARequest(BaseModel):
    context: str
    question: str

class MedicalQuestionRequest(BaseModel):
    question: str


@app.post('/answer')
async def answer_question(data: QARequest):
    # Use the model to answer the question
    inputs = tokenizer(data.question, data.context, return_tensors="pt")
    answer = model(**inputs)

    # Extract the answer text from the model's response
    answer_start = torch.argmax(answer['start_logits'])
    answer_end = torch.argmax(answer['end_logits']) + 1
    answer_text = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])

    # Return the context, question, and the extracted answer
    return {
        "context": data.context,
        "question": data.question,
        "answer": answer_text
    }

@app.post('/ask-medical-question')
async def ask_medical_question(data: MedicalQuestionRequest):
    # Generate an answer from the medical question
    input_text = data.question
    input_ids = text_gen_tokenizer.encode(input_text, return_tensors="pt", max_length=50, truncation=True)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    generated_text = text_gen_model.generate(input_ids, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, pad_token_id=text_gen_tokenizer.eos_token_id, attention_mask=attention_mask)
    answer = text_gen_tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return {"question": data.question, "answer": answer}

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf.pages:
        text += page.extract_text()
    return text

@app.post('/document')
#  Endpoint expects file upload and a question sent as form data
async def upload_document(file: UploadFile = File(...), question: str = Form(...)):
    # Open the uploaded PDF file and extract text
    file_content = io.BytesIO(file.file.read())
    text = extract_text_from_pdf(file_content)
    
    # Tokenize input text,pass it through question answering model and select answer span by finding tokens with highest start and end logits. Finally decode it to human-readable text
    # Tokenize the extracted text and question
    inputs = tokenizer(question, text, return_tensors="pt", padding=True, truncation=True)

    # Perform inference to extract answer start and end logits
    start_logits, end_logits = model(**inputs).start_logits, model(**inputs).end_logits

    # Find the answer span by selecting the tokens with the highest logits
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1

    # Extract and return the answer text
    answer_text = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])

    return {"question": question, "answer": answer_text}


# Define the clean_text function to remove bullet points and special characters
def clean_text(text):
    # Remove bullet points and special characters
    cleaned_text = re.sub(r'[•o]', '', text)

    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
