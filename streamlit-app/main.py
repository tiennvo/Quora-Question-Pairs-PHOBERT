from tkinter import *
import helper
from sklearn.ensemble import RandomForestClassifier
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModel
import faiss
import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv('D:/Python/Quora/Quora/quora-question-pairs/Data/Dataset/train/dataset.csv')
# Tải lại mô hình và tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./phobert-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./phobert-finetuned")
model_embed = AutoModel.from_pretrained("./phobert-embed")
index = faiss.read_index("faiss_index.bin")

# Dự đoán độ tương đồng
def predict_similarity(question1, question2):
    inputs = tokenizer(question1, question2, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()  # Xác suất thuộc lớp tương đồng (1)

def isDuplicate(question1, question2):
    q1 = helper.preprocess(question1)
    q2 = helper.preprocess(question2)
    result = predict_similarity(q1, q2)
    return result

def Find():
    question1 = q1.get()
    question2 = q2.get()

    score = isDuplicate(question1, question2)
    if score > 0.6:
        result_label.config(text= "Duplicate (Score: " + str(int(round(score * 100))) + "%)\n" + "Answer: " + find_answer_faiss(helper.preprocess(question1), df))
    else:
        result_label.config(text= "Not Duplicate (Score: " + str(int(round(score * 100))) + "%)")

# Dự đoán độ tương đồng
def predict_similarity(question1, question2):
    inputs = tokenizer(question1, question2, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()

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

root=Tk()
root.title('Quora')
root.minsize(height=300, width=850)

Label(root, text='Question 1').grid(row=2, column=0)
q1=Entry(root, width=80)
q1.grid(row=2, column=1)

Label(root, text='Question 2').grid(row=3, column=0)
q2=Entry(root, width=80)
q2.grid(row=3, column=1)

button=Frame(root)
Button(button, text="Find", command=Find).pack(side=LEFT)
button.grid(row=4, column=0)

result_label = Label(root, text="")
result_label.grid(row=5, column=0)

root.mainloop()