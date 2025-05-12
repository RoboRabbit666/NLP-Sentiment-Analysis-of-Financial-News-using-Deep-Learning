# NLP Sentiment Analysis of Financial News using Deep Learning

![License](https://img.shields.io/github/license/RoboRabbit666/NLP-Sentiment-Analysis-of-Financial-News-using-Deep-Learning)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)
![BERT](https://img.shields.io/badge/BERT-transformer-yellow.svg)

## Overview
This project implements multiple deep learning models to perform sentiment analysis on financial news headlines, classifying them as positive, negative, or neutral. The models include simple feedforward neural networks (FNN), convolutional neural networks (CNN), long short-term memory networks (LSTM), and pre-trained BERT transformers.

## Table of Contents
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [References](#references)

## Dataset
The dataset consists of financial news headlines labeled with sentiment classes (positive, negative, neutral). The data distribution is:
- Positive: X%
- Negative: Y%
- Neutral: Z%

## Models Implemented
1. **Baseline Models**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - SVM

2. **Deep Learning Models**
   - Feedforward Neural Network
   - Convolutional Neural Network (CNN)
   - Long Short-Term Memory (LSTM) with attention mechanism
   - BERT Transformer

## Results
The models achieved the following accuracy scores:
| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| BERT | 87.56% | 0.87 | 0.94 |
| LSTM | 82.34% | 0.83 | 0.91 |
| CNN | 80.12% | 0.80 | 0.89 |
| FNN | 78.45% | 0.79 | 0.88 |
| LogReg | 76.23% | 0.76 | 0.85 |
| RandomForest | 75.67% | 0.75 | 0.83 |
| DecisionTree | 70.12% | 0.69 | 0.78 |
| Null Model | 45.23% | 0.45 | N/A |

![Model Comparison](assets/model_comparison.png)

## Installation
```bash
# Clone this repository
git clone https://github.com/RoboRabbit666/NLP-Sentiment-Analysis-of-Financial-News-using-Deep-Learning.git

# Install dependencies
pip install -r requirements.txt