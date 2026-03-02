# Depression-Detection-from-Tweets-Using-Machine-Learning-Models
A comparative machine learning study using NLP, POS-aware lemmatization, and TF-IDF to detect depressive indicators in Twitter data with 89% accuracy.
🧠 Twitter Depression Detection Analysis
📌 Project Overview
This project implements a binary classification pipeline to identify signs of depression in social media text. Using a dataset of tweets spanning Dec 2019 to Dec 2020, we developed a system that distinguishes between Depressed (0) and Non-Depressed (1) states with high precision.

The core innovation of this project is the Linguistic Neutrality Filter, which programmatically identifies and removes high-frequency words that appear equally in both classes to reduce model noise.

🗂 Project Workflow
Data Cleaning: Handled null values, removed ~10,000 duplicates, and filtered out non-informative single-word tweets.

Advanced Preprocessing: * Regex cleaning (URLs, Mentions, RT tags).

POS-aware Lemmatization using NLTK WordNetLemmatizer to preserve semantic context.

Custom stopword filtering based on class-frequency ratios.

Exploratory Data Analysis (EDA):

Statistical comparison of word/character counts (KDE Plots).

TF-IDF analysis of Unigrams, Bigrams, and Trigrams.

Visualizing unique emotional lexicons via WordClouds.

Feature Engineering: * Implementation of a "Neutral Zone" (ratios 0.35–0.65) to isolate 1,656 non-discriminative words.

TF-IDF Vectorization with ngram_range=(1, 2) and 10,000 feature limit.

Benchmarking: Comparative evaluation of five machine learning architectures.
