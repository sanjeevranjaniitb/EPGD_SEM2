# Learning to Label – From Clustering to Classification  

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)](https://scikit-learn.org/)  
[![NLTK](https://img.shields.io/badge/NLTK-Text--Processing-green)](https://www.nltk.org/)  
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)  

> **Semi-supervised learning pipeline** for **text** and **image data**:  
> from *clustering unlabeled data* ➝ *pseudo-labeling* ➝ *classification*.

---

## 📌 Overview  

This project demonstrates how to build **end-to-end pipelines** for clustering and classifying **unlabeled text and image datasets**.  

We experimented with:  
- **Text Documents** → Word embeddings, clustering, and classification.  
- **Handwritten Images (MNIST)** → Convolution-based feature extraction, clustering, and semi-supervised classification.  

---


## 📝 Text Processing  

<details>
<summary>Click to expand</summary>

### Problem  
Unlabeled text documents (5 topics) clustered and classified.  

### Steps  
1. **Preprocessing** – tokenization, stopword removal, lemmatization.  
2. **EDA** – word frequency, token count distribution.  
3. **Vectorization** – Word2Vec (100-d vectors).  
4. **Dimensionality Reduction** – PCA reduced to 5 components.  
5. **Clustering** – Compared KMeans, DBSCAN, Agglomerative.  
   - Best: **KMeans** (silhouette score: `0.2275`).  
6. **Classification** – Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting.  
   - Best: **Random Forest**.  

</details>

---

## 🖼 Image Processing  

<details>
<summary>Click to expand</summary>

### Problem  
Unlabeled MNIST digit images clustered and classified.  

### Approaches  
- **Flattening + PCA + Clustering**  
  - Reduced 784 → 87 features.  
  - Silhouette Score: `0.0725`.  

- **2D Convolution + PCA + Clustering**  
  - Filters: sobel_x, sobel_y, laplacian, sharpen, gaussian_3x3.  
  - Silhouette Score: `0.3373`.  

- **2D Convolution + MaxPooling + PCA + Classification**  
  - Features reduced to 980 → 130 PCA comps.  
  - Logistic Regression performed best.  
  - Silhouette Score: `0.0792`.  

</details>

---

## 🚀 Results & Learnings  

✅ Word embeddings (Word2Vec) > TF-IDF for text clustering  
✅ KMeans + Word2Vec + Random Forest → best pipeline for text  
✅ Convolution + Pooling + PCA + Logistic Regression → best pipeline for images  
✅ Semi-supervised approach effective for large unlabeled datasets  

---

## ⚡ Challenges  

- Evaluating clustering without true labels  
- Computational cost of embeddings + clustering  
- Image data harder to cluster due to pixel-level noise  

---

## 📂 Repository Structure  

```bash
📦 Learning-to-Label
 ┣ 📜 README.md
 ┣ 📓 EPGD_SEM1_TEXT_PROCESSING.ipynb
 ┣ 📓 EPGD_SEM1_IMAGE_PROCESSING.ipynb
 ┣ 📂 data/
 ┃ ┗ raw/ , processed/
 ┣ 📂 results/
 ┃ ┗ plots , metrics , models
 ┗ 📂 docs/
    ┗ workflow.png

🔧 Tech Stack

Languages: Python 3.8+
Libraries: NumPy, Pandas, scikit-learn, NLTK, Matplotlib, Seaborn, Gensim (Word2Vec)
Datasets: Custom text corpus, MNIST handwritten digits

👨‍💻 Authors

Sanjeev Ranjan
Shreyas Shanbhag
Kunal Sant
