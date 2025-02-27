# Sentiment Analysis on Amazon Customer Reviews

This repository contains the code for my Bachelor's Thesis titled **"Machine Learning: A powerful tool for enhancing customer experience through sentiment analysis"**. It explores how sentiment analysis-applying various machine learning techniques-can extract and quantify consumer feedback from product reviews, helping businesses make data-driven decisions to improve customer experience.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [How to Run](#how-to-run)

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
