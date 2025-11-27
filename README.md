# 🎬 Hybrid Explainable Movie Recommendation Framework (HEMRF)

A **Hybrid Explainable AI (XAI) Movie Recommendation System** that combines **Neural Collaborative Filtering (NCF)**, **Content-Based Filtering**, **Knowledge Graphs**, and **SHAP/LIME-based explanations** to provide **accurate, transparent, and user-friendly movie recommendations**.

Developed by **Humaima Anwar**

---

## 🚀 Features

- 🔥 **Hybrid Recommendation Engine**
  - Neural Collaborative Filtering (NCF)
  - Content-Based Filtering
  - Collaborative Filtering
  - Popularity Baseline

- 🧠 **Explainable AI (XAI) Module**
  - Multi-level explanations (simple + detailed)
  - SHAP/LIME for feature contribution
  - Knowledge Graph-based reasoning
  - User-friendly “why” statements

- 🌐 **Multi-source Dataset Integration**
  - MovieLens 100K
  - IMDb metadata
  - TMDB (posters, keywords, popularity)

- 🗃 **Knowledge Graph Construction**
  - Actor → Movie → Genre → Director relationships
  - Relationship weighting  
  - Supports contextual & semantic explanations

---
## 📁 Repository Structure

├── data/
│ ├── movielens/
│ ├── imdb/
│ ├── tmdb/
│
├── src/
│ ├── data_preprocessing/
│ ├── feature_engineering/
│ ├── ncf/
│ ├── content_based/
│ ├── collaborative_filtering/
│ ├── explanation_engine/
│ ├── knowledge_graph/
│ └── evaluation/
│
├── notebooks/
│ ├── MovieLens100K_Analysis.ipynb
│ ├── IMDB & TMDB Analysis.ipynb
│ ├── Multi-dataset Analysis.ipynb
│
├── results/
│ ├── plots/
│ ├── shap_outputs/
│ ├── evaluation_scores/
│
└── README.md

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Deep Learning | PyTorch / TensorFlow |
| Graph Reasoning | PyTorch Geometric / DGL |
| Explainability | SHAP, LIME |
| Data Wrangling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Web Demo (optional) | Flask |

---

## 📚 Datasets

### **MovieLens 100K**
- 100k user–movie ratings  
- 943 users, 1,682 movies  
- Genre + timestamp metadata  

### **IMDb**
- `title.basics` (year, runtime, genres)
- `title.ratings` (avg rating, vote count)

### **TMDB**
- Posters  
- Movie keywords  
- Cast + crew  
- Popularity scores  

---

## 🧩 Methodology Overview

### **1. Data Preprocessing**
- Cleaned MovieLens, IMDb, TMDB  
- Merged using shared movie IDs  
- Handled sparsity & cold-start issues  

### **2. Feature Engineering**
- TF-IDF vectorization (plot summaries)  
- Genre encoding  
- Actor/director graph nodes  
- Knowledge Graph construction  

### **3. Model Training**
#### ✔ Baseline Models
- Popularity  
- Content-Based  
- Item-Based CF  

#### ✔ Hybrid Model (NCF)
- 50-dimensional embeddings  
- MLP layers: `[64 → 32]`  
- Trained for 10 epochs  

### **4. Explanation Engine**
- Layer 1: One-line explanation  
- Layer 2: Detailed SHAP reasoning  
- KG-based semantic path explanations  
- Fidelity & stability validation  

---

## 📈 Results Summary

### 🔹 Baseline Models
| Model | Precision@K | Recall@K | HitRate@K |
|-------|-------------|----------|-----------|
| Popularity | 0.040 | 0.0094 | 0.36 |
| Item-CF | 0.032 | 0.0063 | 0.26 |
| Content-Based | **0.102** | **0.0314** | **0.66** |


### 🔹 Hybrid NCF Model (Best)
| Metric | Score |
|--------|--------|
| Precision@K | **0.1233** |
| Recall@K | **0.0329** |
| HitRate@K | **0.6667** |
| NDCG@K | **0.1290** |
| Diversity | **Highest** |

![CF Contributions](https://github.com/user-attachments/assets/9650b2ef-5d91-47c5-aec1-9c2adc1fd829)

---

## 🎯 Sample Recommendation Output

### **NCF Recommendations for User 224**

1. In the Name of the Father (1993)
2. Spy Hard (1996)
3. My Left Foot (1989)


### **SHAP Explanation**
| Feature | Contribution | User Preference |
|---------|--------------|-----------------|
| Drama | 0.255 | 3.03 |
| Action | 0.192 | 3.48 |
| Romance | 0.187 | 3.08 |

![SHAP Contributions](https://github.com/user-attachments/assets/a93a4bf1-879a-4992-8c95-e728245e0687)

---

## 🧪 How to Run the Project

### 1️⃣ Clone the repo
```bash
git clone https://github.com/yourusername/HEMRF.git
cd HEMRF
```

## 📌 Future Work

- Multi-modal recommendation (images, trailers, audio)
- Reinforcement learning for adaptive suggestions
- Larger cross-platform datasets
- LLM-guided explanation refinement

## 👩‍💻 Authors

Humaima Anwar
📧 humaimaanwar123@gmail.com

## 📜 License
Released under the MIT License.


---

If you want, I can also:

✅ Generate a **requirements.txt**  
✅ Create full **folder structure with empty Python files**  
✅ Add **badges (Build, License, Dataset Links, Stars)**  
✅ Create a **banner/logo for your GitHub repo**  

Just tell me!
