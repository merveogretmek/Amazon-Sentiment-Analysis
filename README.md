# Sentiment Analysis on Amazon Customer Reviews

This repository contains the code for my Bachelor's Thesis titled **"Machine Learning: A powerful tool for enhancing customer experience through sentiment analysis"**. It explores how sentiment analysis-applying various machine learning techniques-can extract and quantify consumer feedback from product reviews, helping businesses make data-driven decisions to improve customer experience.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [How to Run](#how-to-run)
- [Results](#results)

## Overview

This thesis investigates how companies can leverage sentiment analysis to interpret large volumes of customer reviews from e-commerce platforms (in this case, Amazon). By automating the classification of text into *positive, negative*, and *neutral* categories, businesses can more rapidly identify trends, strengths, and areas for improvement in their products.

- Utilizes Amazon Customer US Reviews Dataset (2015–2020)
- Employs a supervised machine learning approach (Logistic Regression, Decision Tree, KNN, etc.)
- Also includes a lexicon and rule-based approach using the VADER sentiment library
- Demonstrates comparisons of model performance (accuracy, precision, recall, F1-score) for two-class (positive/negative) and three-class (positive/negative/neutral) sentiment analysis

## Repository Structure

```bash
.
├── analysis_across_category.py
├── lexicon_analysis.py
├── sentiment_models.py
└── README.md
```

`analysis_across_category.py`
- Demonstrates category-based sentiment analysis using Logistic Regression.
- Trains a model on one category (e.g., Electronics) and tests on another (e.g., Office Products), showcasing the impact of category bias.

`lexicon_analysis.py`
- Illustrates a lexicon and rule-based approach using the VADER Sentiment Analyzer.
- Computes sentiment scores and compares them to star ratings to evaluate model accuracy.

`sentiment_models.py`
- Bundles multiple experiment setups, each using a different ML algorithm:
  - Decision Tree
  - K-Nearest Neighbors
  - Logistic Regression
  - One-vs-One SVM (LinearSVC)
  - Gaussian Naive Bayes
  - Multilayer Perceptron
  - Quadratic Discriminant Analysis
  - Linear SVC
- Provides confusion matrices and classification reports for performance evaluation.

## Getting Started

### Requirements
- Python 3.7+
- Python libraries:
  - `pandas`
  - `scikit-learn`
  - `numpy`
  - `vaderSentiment`
 
## How to Run

Below are the steps for running each script.

### 1. Lexicon-Based Analysis

```bash
python lexicon_analysis.py
```

- Loads the last 10,000 Amazon reviews from `amazon_dataframe.csv`.
- Removes the entries with a star rating of 3 (ambiguous) and define the rest as positive (star rating > 3) or negative (< 3).
- Applies the VADER sentiment analyzer to each review, then prints a confusion matrix and classification report.

### 2. Category-Level Analysis

```bash
python analysis_across_category.py
```

- Loads the dataset.
- Splits it into training and testing subsets.
- Removes reviews with star_rating == 3.
- Trains a Logistic Regression model on the vectorized text data.
- Evaluates the model using a confusion matrix and classification report.
- Demonstrates how classification accuracy can vary across product categories.

### 3. Multiple ML Models

```bash
python sentiment_models.py
```

- Loads the Amazon reviews from `amazon_dataframe.csv`.
- Runs eight different experiments, each applying a distinct machine-learning approach for sentiment classification:
  1. Decision Tree
  2. K-Nearest Neighbors
  3. Logistic Regression
  4. One-vs-One SVM (LinearSVC)
  5. Gaussian Naive Bayes
  6. MLP Classifier
  7. Quadratic Discriminant Analysis
  8. Linear SVC
- Each experiment outputs its confusion matrix and classification report to compare precision, recall, F1-score, and accuracy across algorithms.

## Results

Key findings from this project:
1. Logistic Regression generally outperformed other models (in two-class sentiment analysis) regarding accuracy (around 93%).
2. Lexicon-based (VADER) analysis can be very effective for two-class analysis (≈90% accuracy) but tends to drop in performance for three-class scenarios.
3. Category Bias is significant. Training on multi-category data generally yields more robust models across different product types.
4. Adding a neutral class (three-class analysis) reduces overall accuracy, highlighting the complexity of identifying mixed or borderline sentiment.


**Thank you for visiting this repository!**
For any questions or feedback, please open an issue or reach out via email.


