import streamlit as st
import helper
from sklearn.ensemble import RandomForestClassifier
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import faiss
import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv('D:/Python/Quora/Quora/quora-question-pairs/Data/Dataset/train/dataset.csv')

# Load models
@st.cache_resource
def load_models():
    model_cls = AutoModelForSequenceClassification.from_pretrained("../phobert-finetuned")
    tokenizer = AutoTokenizer.from_pretrained("../phobert-finetuned")
    model_embed = AutoModel.from_pretrained("../phobert-embed")
    index = faiss.read_index("../faiss_index.bin")
    return model_cls, tokenizer, model_embed, index

model_cls, tokenizer, model_embed, index = load_models()

# Hàm dự đoán tương đồng
def predict_similarity(question1, question2):
    inputs = tokenizer(question1, question2, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model_cls(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()

# Hàm encode để tìm vector FAISS
def encode(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model_embed(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


# Hàm tìm câu trả lời gần nhất
def find_answer_faiss(input_question, df):
    input_vector = normalize(encode(input_question)).astype('float32').reshape(1, -1)
    D, I = index.search(input_vector, 1)  # Top-1
    best_match = df.iloc[I[0][0]]
    return best_match['answer'] if D[0][0] > 0.6 else "Không tìm thấy câu trả lời phù hợp."

def find_answer_faiss_all(input_question, df, threshold=0.6, top_k=5):
    # Mã hóa và chuẩn hóa câu hỏi đầu vào
    input_vector = normalize(encode(input_question)).astype('float32').reshape(1, -1)

    # Tìm top_k câu hỏi gần nhất theo cosine similarity
    D, I = index.search(input_vector, top_k)

    # Tìm câu có độ tương đồng cao nhất trong top_k và > threshold
    best_idx = None
    best_score = -1

    for score, idx in zip(D[0], I[0]):
        if score > threshold and score > best_score:
            best_score = score
            best_idx = idx

    # Trả lại kết quả nếu tìm thấy
    if best_idx is not None:
        return df.iloc[best_idx]['answer']
    else:
        return "Không tìm thấy câu trả lời phù hợp."


# Giao diện Streamlit
st.title("Quora Question Pairs")

question1 = st.text_input("Nhập Câu hỏi 1")
question2 = st.text_input("Nhập Câu hỏi 2")

if st.button("🔍 Kiểm tra"):
    q1 = helper.preprocess(question1)
    q2 = helper.preprocess(question2)
    score = predict_similarity(q1, q2)

    if score > 0.6:
        answer = find_answer_faiss(q1, df)
        st.success(f"✅ Duplicate (Score: {round(score * 100)}%)")
        st.markdown(f"**Gợi ý trả lời:** {answer}")
    else:
        st.warning(f"❌ Not Duplicate (Score: {round(score * 100)}%)")
