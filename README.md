# 💬 Indonesian Sentiment Analysis — App Reviews

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-154f5b)
![Sastrawi](https://img.shields.io/badge/Sastrawi-Indonesian%20NLP-2e7d32)
![License](https://img.shields.io/badge/License-MIT-green.svg)

End-to-end **Indonesian-language** sentiment classification on real app reviews — from data collection to a working inference demo.

## Pipeline
1. **Data scraping** — collect reviews directly from the Google Play Store (`google-play-scraper`).
2. **Preprocessing** — Indonesian text cleaning with **Sastrawi** stemming and **NLTK** stopword removal.
3. **Labeling & EDA** — sentiment labeling and word-cloud / distribution exploration.
4. **Imbalance handling** — `SMOTE` / `RandomOverSampler` to balance sentiment classes.
5. **Modeling** — **TF-IDF** features feeding deep models (Embedding / LSTM / Conv1D in Keras).
6. **Inference** — predict sentiment on unseen text.

## Tech Stack
`TensorFlow` · `Keras` · `NLTK` · `Sastrawi` · `scikit-learn` · `imbalanced-learn` · `WordCloud`

## Notebooks
- `data_scraping.ipynb` — review collection
- `sentiment_analysis.ipynb` — preprocessing, modeling, and inference

## Run
```bash
pip install -r requirements.txt
jupyter notebook sentiment_analysis.ipynb
```

## 👤 Author

**Aldo Maretra Putra** — Astronomy student & ML practitioner
📧 aldomaretraputra7@gmail.com · 🤗 [aldomrtr](https://huggingface.co/aldomrtr) · 🐙 [AllsHub](https://github.com/AllsHub)
