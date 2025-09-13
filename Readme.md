# 📄 Automated Resume Screening App

## 🎯 Project Objective

Build an **AI-powered Resume Screening App** using **NLP + Machine Learning + Streamlit** to help recruiters automatically classify resumes, extract skills, and compute candidate fit scores.

[Preview App](https://automated-resume-screening-ml.streamlit.app/#automated-resume-screening)

---

## 📂 Dataset

* Source: [Kaggle - Resume Dataset](https://www.kaggle.com/datasets) (Resume Dataset, \~1000 resumes, 25 job categories)
* Columns: `Category` (target job role), `Resume` (raw text)

---

## ⚙️ Applied Concepts & Steps

1. **Text Preprocessing**: lowercasing, regex cleaning, stopword removal, lemmatization
2. **Feature Engineering**: TF-IDF vectorization (unigrams + bigrams)
3. **Model Training**: Logistic Regression, Naive Bayes (final model = Logistic Regression, \~99% accuracy)
4. **Fit Score Formula**: `70% prediction probability + 30% skill matching`
5. **Deployment**: Streamlit app with file upload & prediction

---

## 🔍 Key Insights

* Class imbalance observed (Java Dev = 84 vs Advocate = 20 resumes)
* Resumes contained noisy data (emails, URLs, headers → cleaned via preprocessing)
* Logistic Regression handled imbalanced classes better than Naive Bayes
* Accurate classification even for small classes like Advocate

---

## 💡 Skills Learnt

NLP · Text Preprocessing · TF-IDF · Logistic Regression · Naive Bayes · EDA · Regex · NLTK · scikit-learn · Streamlit

---

## 🛠️ Project Structure & Run

```
## 🛠️ Project Structure

resume_screening/
├── nltk_data/                    # NLTK stopwords, tokenizers
├── src/
│   ├── resumes_dataset.csv       # Original dataset
│   ├── resumes_df.pkl            # Pickled preprocessed dataset
│   ├── file_utils.py             # Extract text from PDF/DOCX
│   ├── skills_db.py              # Skills database per job category
│   ├── app_helpers.py            # Helper functions for skills, education, fit score
│
├── eda.ipynb                     # Exploratory Data Analysis
├── preprocess_utils.ipynb        # Preprocessing pipeline
├── train_model.py                # ML model training script
├── app.py                        # Streamlit app
├── Readme.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

Run locally:

```bash
pip install -r src/requirements.txt
streamlit run app.py
```

---

## 🔮 Future Scope

* Deploy on Streamlit Cloud / Hugging Face
* Use advanced NLP models (BERT, DistilBERT)
* Handle class imbalance with SMOTE/weighted loss
* Enhance fit score with real HR metrics

---

## ✨ Author

Developed by **[Mariam Khan](https://www.linkedin.com/in/mariam-khan0424)** 🎯
