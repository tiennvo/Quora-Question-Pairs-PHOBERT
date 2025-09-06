# Quora Question Pairs - PhoBERT

🚀 **Comparing Question Similarity in Vietnamese Forums using PhoBERT & FAISS**

[![Paper DOI](https://img.shields.io/badge/DOI-10.34238%2Ftnu--jst.12516-blue)](https://doi.org/10.34238/tnu-jst.12516)
[![GitHub stars](https://img.shields.io/github/stars/tiennvo/Quora-Question-Pairs-PHOBERT?style=social)](https://github.com/tiennvo/Quora-Question-Pairs-PHOBERT/stargazers)

---

## 📖 Giới thiệu

Dự án này được phát triển dựa trên nghiên cứu khoa học đăng tại **TNU Journal of Science and Technology**:  
👉 [Comparing Question Similarity in Forums (2025)](https://jst.tnu.edu.vn/jst/article/view/12516)

Mục tiêu:  
- Phát triển hệ thống **so sánh độ tương đồng câu hỏi** trên diễn đàn trực tuyến.  
- Ứng dụng **PhoBERT** (mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt) kết hợp **FAISS** để tăng tốc tìm kiếm tương đồng.  
- Giúp **tự động gợi ý câu trả lời** phù hợp, tối ưu trải nghiệm người dùng trên nền tảng Q&A, diễn đàn, và hệ thống hỗ trợ khách hàng.  

---

## 🛠️ Kiến trúc hệ thống

1. **Thu thập & Xử lý dữ liệu**  
   - 31,201 cặp câu hỏi tiếng Việt.  
   - Nguồn từ diễn đàn, confession sinh viên, fanpage + dịch từ tập *Quora Question Pairs*.  
   - Gán nhãn thủ công (1: tương đồng, 0: không tương đồng).  

2. **Huấn luyện mô hình PhoBERT**  
   - Fine-tuned trên tập dữ liệu trên.  
   - Accuracy đạt **82.98%**, vượt trội hơn TF-IDF + Cosine Similarity (75.62%).  

3. **Tìm kiếm nhanh với FAISS**  
   - Các câu hỏi được mã hóa thành vector 768 chiều bằng PhoBERT.  
   - FAISS index (IndexFlatIP / IndexIVFFlat) cho phép tìm kiếm thời gian thực trong cơ sở dữ liệu lớn.  

📊 **So sánh hiệu năng** (Bảng 1 trong bài báo):  
| Phương pháp                  | Accuracy | Precision | Recall | F1-score |
|-------------------------------|----------|-----------|--------|----------|
| TF-IDF + Cosine Similarity    | 75.62%   | 74.31%    | 76.85% | 75.56%   |
| **PhoBERT (fine-tuned)**      | **82.98%** | **82.70%** | **84.62%** | **83.65%** |

---

## 📂 Cấu trúc repo

```
├── data/                  # Dữ liệu huấn luyện và kiểm thử
├── models/                # Lưu mô hình PhoBERT fine-tuned
├── phobert-embed/         # Vector embeddings
├── results/               # Kết quả huấn luyện và đánh giá
├── streamlit-app/         # Ứng dụng demo với Streamlit
├── train.py               # Script huấn luyện
├── inference.py           # Script dự đoán
└── README.md
```

---

## 🚀 Cài đặt & Chạy thử

### 1. Cài đặt môi trường
```bash
git clone https://github.com/tiennvo/Quora-Question-Pairs-PHOBERT.git
cd Quora-Question-Pairs-PHOBERT

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình
```bash
python train.py --epochs 5 --batch_size 32
```

### 3. Chạy inference (dự đoán độ tương đồng)
```bash
python inference.py --q1 "Trường nào có ngành CNTT?" --q2 "Đại học nào đào tạo ngành Công nghệ thông tin?"
```

### 4. Ứng dụng demo với Streamlit
```bash
streamlit run streamlit-app/app.py
```

---

## 📑 Bài báo khoa học

- **Tiêu đề**: Comparing Question Similarity in Forums  
- **Tác giả**: Võ Trần Tiến, Lương Trần Ngọc Khiết, Nguyễn Phương Nam, Huỳnh Thị Tường Vi, Nguyễn Huỳnh Phúc Khang, Phan Thị Nam Anh, Lương Trần Hy Hiến  
- **Tạp chí**: *TNU Journal of Science and Technology*, Vol. 230(07), pp. 198–207, 2025.  
- **DOI**: [10.34238/tnu-jst.12516](https://doi.org/10.34238/tnu-jst.12516)  

---

## 🌟 Ứng dụng thực tế

- Nền tảng hỏi đáp trực tuyến (forums, Quora tiếng Việt).  
- Chatbot hỗ trợ khách hàng.  
- Ngân hàng đề thi (phát hiện câu hỏi trùng lặp).  
- Phát hiện đạo văn, phân tích ngữ nghĩa.  

---

## 🤝 Đóng góp

Mọi đóng góp, báo lỗi hoặc đề xuất cải tiến đều được chào đón!  
Hãy tạo **issue** hoặc **pull request** để cùng phát triển dự án.

---

## 📜 Giấy phép

MIT License © 2025 [Tienn Vo](https://github.com/tiennvo)
