# E-Commerce Product Text Classification using NLP

## Project Overview

This project focuses on **multi-class text classification** using Natural Language Processing (NLP) techniques on a real-world e-commerce dataset. The goal is to automatically predict the product category based on its textual description.

In large-scale e-commerce platforms, manually categorizing thousands of products is inefficient, inconsistent, and error-prone. This project demonstrates how machine learning and NLP can be used to automate product categorization, improving catalog management, search relevance, and user experience.

The notebook explores and compares multiple text representation techniques such as:

* Bag of Words (BoW)
* TF-IDF
* Word Embeddings (Word2Vec / GloVe)

These embeddings are then used with machine learning models to evaluate classification performance.

---

## 📂 Repository Structure

```
├── Text Classification.ipynb   # Main Google Colab Notebook
├── README.md                   # Project Documentation
```

---

## 📊 Dataset

* **Source:** Zenodo E-commerce Dataset
* **Size:** ~50,000 product records
* **Columns:**

  * `Category` – Product category label
  * `Description` – Product text description

The dataset contains noisy, real-world product descriptions across multiple categories such as:

* Household items
* Office supplies
* Home decor
* Gifts and more

---

## Project Workflow

### 1 - Data Loading & Exploration

* Loaded dataset directly from online source
* Data inspection and validation
* Handling missing values and text cleaning

### 2️ - Text Preprocessing

* Lowercasing text
* Removing punctuation & noise
* Tokenization
* Stopword removal (if applied)

### 3️ - Feature Engineering (NLP Techniques)

The project compares multiple vectorization approaches:

* Bag of Words (Unigrams & Bigrams)
* TF-IDF Vectorization
* Word Embeddings (GloVe / Word2Vec)

### 4️ - Model Training

Machine Learning models are trained on different text embeddings to evaluate performance on multi-class classification.

### 5️ - Model Evaluation

* Accuracy comparison across techniques
* Performance benchmarking of embedding methods
* Insight into best-performing representation for text classification

---

## Key Features

* End-to-end NLP classification pipeline
* Multiple embedding technique comparison
* Real-world noisy dataset handling
* Scalable approach for automated product tagging
* Reproducible Google Colab workflow

---

## Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* Gensim (Word Embeddings)
* Matplotlib / Seaborn (Visualization)
* Google Colab (Development Environment)

---

## How to Run the Project

### Option 1: Open in Google Colab (Recommended)

1. Upload the notebook to Google Colab OR
2. Use the Colab link after uploading to GitHub:

```
https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Text%20Classification.ipynb
```

3. Run all cells sequentially.

### Option 2: Run Locally

```bash
pip install numpy==1.24.4 scipy==1.10.1 gensim scikit-learn pandas matplotlib seaborn
jupyter notebook
```

Then open:

```
Text Classification.ipynb
```

---

## Results & Insights

* TF-IDF and word embeddings significantly improve classification performance compared to basic Bag-of-Words.
* Handling noisy e-commerce text is crucial for real-world NLP applications.
* Embedding-based models capture semantic meaning better than frequency-based approaches.

---

## Real-World Applications

* Automated product categorization in e-commerce
* Content tagging systems
* Search and recommendation engines
* Document classification systems

---

## ⭐ Future Improvements

* Add Deep Learning models (LSTM / BERT)
* Hyperparameter tuning
* SHAP for model interpretability
* Deployment as an API for real-time classification
