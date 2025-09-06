import threading
import time
import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
from gensim.models.keyedvectors import KeyedVectors
from PIL import Image, ImageTk
import re
import jellyfish
from typing import List, Set
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AdamW
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Tải lại mô hình và tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./phobert-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./phobert-finetuned")

# Dự đoán độ tương đồng
def predict_similarity(question1, question2):
    inputs = tokenizer(question1, question2, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[0][1].item()  # Xác suất thuộc lớp tương đồng (1)

