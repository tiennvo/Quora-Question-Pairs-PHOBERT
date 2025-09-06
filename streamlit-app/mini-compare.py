import threading
import time
import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import pandas as pd
import os
from gensim.models.keyedvectors import KeyedVectors
from PIL import Image, ImageTk
import re
import jellyfish
from typing import List, Set
import openpyxl

stop_event = threading.Event()
# Kiểu mô hình mặc định
default_model = "PhoBERT"
model = None
tokenizer = None
word2vec_model = None
doc2vec_model = None
test1_path = None
test2_path = None
# độ dài mã thông báo
MAX_LENGTH = 512
local_model_dir = "./models"
matrix_file_path1 = None
matrix_file_path2 = None

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_or_download_phobert(progress, step):
    phobert_model_path = os.path.join(local_model_dir, "models--vinai--phobert-base")
    if not os.path.exists(phobert_model_path):
        print("Downloading PhoBERT model...")
        AutoModel.from_pretrained("vinai/phobert-base", cache_dir=local_model_dir)
        AutoTokenizer.from_pretrained("vinai/phobert-base", cache_dir=local_model_dir)
        progress.set(progress.get() + step)
    else:
        print("PhoBERT model loaded from local storage.")
        progress.set(progress.get() + step)
    return AutoModel.from_pretrained(
        "vinai/phobert-base", cache_dir=local_model_dir
    ), AutoTokenizer.from_pretrained("vinai/phobert-base", cache_dir=local_model_dir)


def load_or_download_word2vec(progress, step):
    word2vec_model_path = os.path.join(local_model_dir, "word2vec-google-news-300.gz")
    if not os.path.exists(word2vec_model_path):
        print("Downloading Word2Vec model...")
        model = api.load("word2vec-google-news-300")
        model.save(word2vec_model_path)
        progress.set(progress.get() + step)
    else:
        print("Word2Vec model loaded from local storage.")
        progress.set(progress.get() + step)
    return KeyedVectors.load(word2vec_model_path)


def load_or_download_doc2vec(progress, step):
    doc2vec_model_path = os.path.join(local_model_dir, "doc2vec.model")
    if not os.path.exists(doc2vec_model_path):
        print("Training and saving Doc2Vec model...")
        documents = [
            TaggedDocument(doc, [i]) for i, doc in enumerate(api.load("text8"))
        ]
        model = Doc2Vec(
            documents, vector_size=50, window=5, min_count=2, workers=8, epochs=50, dm=0
        )
        model.save(doc2vec_model_path)
        progress.set(progress.get() + step)
    else:
        print("Doc2Vec model loaded from local storage.")
        progress.set(progress.get() + step)
    return Doc2Vec.load(doc2vec_model_path)


def preload_models(startup_label, progress):
    global model, tokenizer, word2vec_model, doc2vec_model

    def update_startup_label(text):
        root.after(0, lambda: startup_label.configure(text=text))

    try:
        if not os.path.exists(local_model_dir):
            os.makedirs(local_model_dir)
            print(f"Directory created: {local_model_dir}")
        update_startup_label("Is starting, please wait...")
        ensure_dir_exists(local_model_dir)
        step = 1 / 3  # For three models, progress bar range is [0, 1]
        model, tokenizer = load_or_download_phobert(progress, step)
        word2vec_model = load_or_download_word2vec(progress, step)
        doc2vec_model = load_or_download_doc2vec(progress, step)
        update_startup_label("Starting completed!")
        time.sleep(1)
        root.after(0, startup_window.destroy)  # Đóng startup window
        root.after(0, root.deiconify)  # Hiển thị root window, sửa lỗi gọi hàm
    except Exception as e:
        root.after(0, update_startup_label, f"Error loading models: {str(e)}")


def normalize_similarity_matrix(matrix):
    return (matrix + 1) / 2

# chuẩn hoá question
def get_embedding(sentence):
    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


def sentence_similarity(sent1, sent2):
    embed1 = get_embedding(sent1)
    embed2 = get_embedding(sent2)
    cosine_similarity = np.dot(embed1, embed2) / (
        np.linalg.norm(embed1) * np.linalg.norm(embed2)
    )
    return cosine_similarity


def calculate_similarity_phobert(questions):
    try:
        normalized_similarity_matrix = sentence_similarity(questions[0], questions[1])
        # if normalized_similarity_matrix >1:
        #     normalized_similarity_matrix = 1.0
        return normalized_similarity_matrix
    except Exception as e:

        raise RuntimeError(f"Error calculate similarity Phobert: {str(e)}")

def calculate_similarity_word2vec(questions):
    try:
        embeddings = []
        for q in questions:
            words = q.split()
            word_vecs = [word2vec_model[w] for w in words if w in word2vec_model]
            if word_vecs:
                embeddings.append(np.mean(word_vecs, axis=0))
            else:
                embeddings.append(np.zeros(word2vec_model.vector_size))
        similarity_matrix = cosine_similarity(embeddings)
        normalized_similarity_matrix = normalize_similarity_matrix(similarity_matrix)
        return normalized_similarity_matrix
    except Exception as e:
        raise RuntimeError(f"Error calculate similarity Word2vec: {str(e)}")


def calculate_similarity_doc2vec(questions):
    try:
        embeddings = [doc2vec_model.infer_vector(q.split()) for q in questions]
        similarity_matrix = cosine_similarity(embeddings)
        normalized_similarity_matrix = normalize_similarity_matrix(similarity_matrix)
        return normalized_similarity_matrix
    except Exception as e:
        raise RuntimeError(f"Error calculate similarity Doc2vec: {str(e)}")

def calculate_similarity(questions, model_type):
    questions1 = questions.copy()
    questions2 = questions.copy()
    questions3 = questions.copy()

    if model_type == "PhoBERT":
        return calculate_similarity_phobert(questions1)
    elif model_type == "word2vec":
        return calculate_similarity_word2vec(questions2)
    elif model_type == "doc2vec":
        return calculate_similarity_doc2vec(questions3)



# ---------------------------------------------
# Tab 3: So sánh dựa trên từ khóa
# ---------------------------------------------

# Khai báo biến toàn cục
data_file_path1 = ""
data_file_path2 = ""
keywords_file_path = ""
matrix_tab3 = []

# Hàm đọc từ khóa từ file Excel
def load_keywords_from_excel(
    file_path: str, sheet_name: str, column_name: str
) -> Set[str]:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    keywords = set(df[column_name].dropna().str.strip().str.lower())
    return keywords

# Hàm chuẩn bị văn bản
def preprocess(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

# Hàm kiểm tra từ khóa dựa trên ngưỡng tương đồng Jaro-Winkler
def is_keyword_present_with_threshold(
    keyword: str, words: List[str], threshold: float = 0.6
) -> int:
    for word in words:
        similarity = jellyfish.jaro_winkler_similarity(keyword, word)
        if similarity >= threshold:
            return 1
    return 0

# Hàm tạo véc-tơ đặc trưng cho câu hỏi
def create_feature_vector(
    words: List[str], keywords: Set[str], threshold: float = 0.6
) -> List[int]:
    feature_vector = [
        is_keyword_present_with_threshold(keyword, words, threshold)
        for keyword in keywords
    ]
    return feature_vector

# Hàm tính độ tương đồng Jaccard
def jaccard_similarity(vec1: List[int], vec2: List[int]) -> float:
    intersection = sum(1 for v1, v2 in zip(vec1, vec2) if v1 == 1 and v2 == 1)
    union = sum(1 for v1, v2 in zip(vec1, vec2) if v1 == 1 or v2 == 1)
    return intersection / union if union != 0 else 0