# -*- coding: utf-8 -*-
"""
Financial Sentiment Analysis using Multiple Deep Learning Models
This script implements sentiment analysis on financial news headlines
using various machine learning and deep learning models.
"""

# =============================================================================
# IMPORT SECTION - All libraries consolidated
# =============================================================================
# General libraries
import os
import re
import json
import string
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import cycle
from tqdm.auto import tqdm

# Data processing and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                            ConfusionMatrixDisplay, f1_score,
                            roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Deep learning libraries - Traditional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Deep learning libraries - Keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Deep learning libraries - Transformers
from transformers import (BertTokenizer, BertForSequenceClassification,
                         AdamW, get_linear_schedule_with_warmup)

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
class Config:
    """Configuration class to store all parameters"""
    # Data parameters
    data_path = 'all-data.csv'  # Update this path
    test_size = 0.2
    random_state = 42
    
    # Model parameters
    # FNN parameters
    fnn_epochs = 200
    fnn_batch_size = 128
    fnn_learning_rate = 0.001
    
    # CNN parameters
    cnn_epochs = 200
    cnn_batch_size = 128
    cnn_learning_rate = 0.001
    cnn_embed_dim = 128
    
    # LSTM parameters
    lstm_epochs = 200
    lstm_batch_size = 128
    lstm_learning_rate = 0.001
    lstm_embed_dim = 128
    lstm_hidden_dim = 64
    lstm_layers = 2
    lstm_dropout = 0.4
    
    # BERT parameters
    bert_seed_val = 42
    bert_epochs = 10
    bert_batch_size = 6
    bert_seq_length = 256
    bert_learning_rate = 2e-5
    bert_epsilon = 1e-8
    bert_pretrained_model = 'bert-base-uncased'
    bert_add_special_tokens = True
    bert_return_attention_mask = True
    bert_pad_to_max_length = True
    bert_do_lower_case = False
    bert_return_tensors = 'pt'
    
    # Visualization parameters
    sentiment_colors = {'positive': 'green', 'negative': 'red', 'neutral': 'grey'}
    sentiment_order = ['positive', 'negative', 'neutral']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def download_nltk_resources():
    """Download necessary NLTK resources"""
    nltk.download('stopwords')
    nltk.download('punkt')

def remove_stopwords(text):
    """Remove stopwords from text"""
    stop_words = set(stopwords.words('english'))
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

def clean_text(text):
    """Clean text by removing special characters, URLs, etc."""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = ' '.join(text.split())
    return text

def stemming(sentence):
    """Apply stemming to sentence"""
    stemmer = SnowballStemmer("english")
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def preprocess_text(df):
    """Apply all text preprocessing steps to dataframe"""
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].apply(lambda x: remove_stopwords(x))
    df['text'] = df['text'].apply(lambda x: clean_text(x))
    df['text'] = df['text'].apply(stemming)
    return df

def flat_accuracy(preds, labels):
    """Calculate accuracy for BERT model"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def load_data(config):
    """Load data from CSV file"""
    df = pd.read_csv(config.data_path, names=["Sentiment", "text"],
                    encoding="utf-8", encoding_errors="replace")
    print(f"Loaded data with shape: {df.shape}")
    return df

def clean_data(df):
    """Clean the data by removing duplicates and nulls"""
    print("Initial data info:")
    print(df.info())
    print(f"Null values: {df.isnull().sum()}")
    print(f"Duplicate values: {df.duplicated().sum()}")
    
    # Remove duplicates
    df.drop_duplicates(keep='first', inplace=True)
    
    print("\nCleaned data info:")
    print(df.info())
    print(f"Sentiment distribution: {df['Sentiment'].value_counts()}")
    
    return df

def visualize_data(df, config):
    """Create visualizations of the data"""
    # Sentiment distribution visualization
    sentiment_count = df['Sentiment'].value_counts().reindex(config.sentiment_order)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bar chart
    sns.barplot(x=sentiment_count.index, y=sentiment_count.values, 
                palette=config.sentiment_colors, ax=ax[0])
    ax[0].set_title('Sentiment Count', fontsize=16)
    ax[0].set_xlabel('Sentiment', fontsize=12)
    ax[0].set_ylabel('Count', fontsize=12)
    for i, count in enumerate(sentiment_count.values):
        ax[0].text(i, count, count, ha='center', va='bottom', fontsize=10)
    
    # Pie chart
    ax[1].pie(sentiment_count.values, labels=sentiment_count.index, 
              colors=[config.sentiment_colors[i] for i in config.sentiment_order], 
              autopct='%1.1f%%', textprops={'fontsize': 10})
    ax[1].set_title('Sentiment Proportions', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Word cloud visualization
    all_words = ' '.join(word for word in df['text'])
    wordcloud = WordCloud(width=800, height=500, random_state=21, 
                         max_font_size=110, background_color='skyblue').generate(all_words)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
    plt.show()

def prepare_data(df, config):
    """Split data into train and test sets"""
    X = df['text']
    y = df['Sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state)
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Vectorize text data for traditional ML models
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_vectorized, X_test_vectorized, vectorizer


# =============================================================================
# BASELINE MODELS
# =============================================================================
def train_evaluate_null_model(df):
    """Train and evaluate a null model (baseline)"""
    print("\n===== Null Model =====")
    # Null model accuracy
    score_null_model = df['Sentiment'].value_counts(normalize=True).max()
    score_null_model = round(score_null_model*100, 2)
    print(f'Null model accuracy: {score_null_model}%')
    
    # Null model F1 score
    true_labels = df['Sentiment'].values
    null_predictions = np.full(true_labels.shape, df['Sentiment'].value_counts().idxmax())
    f1_null_model_report = f1_score(true_labels, null_predictions, average='weighted')
    print(f'F1-score of the null model: {round(f1_null_model_report*100, 2)}%')
    
    return score_null_model, f1_null_model_report

def train_evaluate_logistic_regression(X_train_vectorized, X_test_vectorized, y_train, y_test):
    """Train and evaluate a logistic regression model"""
    print("\n===== Logistic Regression Model =====")
    
    # Train model
    LR_model = LogisticRegression(n_jobs=-1)
    LR_model.fit(X_train_vectorized, y_train)
    
    # Predictions
    LR_model_predictions = LR_model.predict(X_test_vectorized)
    
    # Accuracy
    LR_model_score = accuracy_score(y_test, LR_model_predictions)
    LR_model_score = round(LR_model_score*100, 2)
    print(f'Logistic Regression model accuracy: {LR_model_score}%')
    
    # Classification report
    report = classification_report(y_test, LR_model_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop('support', axis=1)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title('Classification Report of Logistic Regression Model')
    plt.savefig('lr_classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # F1 score
    f1_LR_model_report = df_report.loc['weighted avg', 'f1-score']
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, LR_model_predictions)
    plt.title('Confusion Matrix of Logistic Regression Model')
    plt.savefig('lr_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC curve
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
    n_classes = y_test_binarized.shape[1]
    probs = LR_model.predict_proba(X_test_vectorized)
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Define class names
    class_names = {0: "negative", 1: "neutral", 2: "positive"}
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Logistic Regression Model')
    plt.legend(loc="lower right")
    plt.savefig('lr_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC AUC score
    roc_auc_LR_model = roc_auc_score(y_test_binarized, probs, average='weighted', multi_class='ovr')
    
    return LR_model_score, f1_LR_model_report, roc_auc_LR_model, LR_model

def train_evaluate_decision_tree(X_train_vectorized, X_test_vectorized, y_train, y_test):
    """Train and evaluate a decision tree model"""
    print("\n===== Decision Tree Model =====")
    
    # Train model
    DT_model = DecisionTreeClassifier(random_state=42)
    DT_model.fit(X_train_vectorized, y_train)
    
    # Predictions
    DT_model_predictions = DT_model.predict(X_test_vectorized)
    
    # Accuracy
    DT_model_score = DT_model.score(X_test_vectorized, y_test)
    DT_model_score = round(DT_model_score*100, 2)
    print(f'Decision Tree model accuracy: {DT_model_score}%')
    
    # Classification report
    report = classification_report(y_test, DT_model_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop('support', axis=1)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title('Classification Report of Decision Tree Model')
    plt.savefig('dt_classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # F1 score
    f1_DT_model_report = df_report.loc['weighted avg', 'f1-score']
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, DT_model_predictions)
    plt.title('Confusion Matrix of Decision Tree Model')
    plt.savefig('dt_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC curve (similar to Logistic Regression)
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
    n_classes = y_test_binarized.shape[1]
    probs = DT_model.predict_proba(X_test_vectorized)
    
    # Compute ROC curve and area
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    class_names = {0: "negative", 1: "neutral", 2: "positive"}
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Decision Tree Model')
    plt.legend(loc="lower right")
    plt.savefig('dt_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC AUC score
    roc_auc_DT_model = roc_auc_score(y_test_binarized, probs, average='weighted', multi_class='ovr')
    
    return DT_model_score, f1_DT_model_report, roc_auc_DT_model, DT_model

def train_evaluate_random_forest(X_train_vectorized, X_test_vectorized, y_train, y_test):
    """Train and evaluate a random forest model"""
    print("\n===== Random Forest Model =====")
    
    # Train model
    RF_model = RandomForestClassifier(random_state=0)
    RF_model.fit(X_train_vectorized, y_train)
    
    # Predictions
    RF_model_predictions = RF_model.predict(X_test_vectorized)
    
    # Accuracy
    RF_model_score = RF_model.score(X_test_vectorized, y_test)
    RF_model_score = round(RF_model_score*100, 2)
    print(f'Random Forest model accuracy: {RF_model_score}%')
    
    # Classification report
    report = classification_report(y_test, RF_model_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop('support', axis=1)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title('Classification Report of Random Forest Model')
    plt.savefig('rf_classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # F1 score
    f1_RF_model_report = df_report.loc['weighted avg', 'f1-score']
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, RF_model_predictions)
    plt.title('Confusion Matrix of Random Forest Model')
    plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return RF_model_score, f1_RF_model_report, RF_model

def train_evaluate_svm(X_train_vectorized, X_test_vectorized, y_train, y_test):
    """Train and evaluate a Support Vector Machine model"""
    print("\n===== Support Vector Machine Model =====")
    
    # Train model
    SVM_model = SVC()
    SVM_model.fit(X_train_vectorized, y_train)
    
    # Predictions
    SVM_model_predictions = SVM_model.predict(X_test_vectorized)
    
    # Accuracy
    SVM_model_score = SVM_model.score(X_test_vectorized, y_test)
    SVM_model_score = round(SVM_model_score*100, 2)
    print(f'Support Vector Machine model accuracy: {SVM_model_score}%')
    
    # Classification report
    report = classification_report(y_test, SVM_model_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop('support', axis=1)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title('Classification Report of Support Vector Machine Model')
    plt.savefig('svm_classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, SVM_model_predictions)
    plt.title('Confusion Matrix of Support Vector Machine Model')
    plt.savefig('svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return SVM_model_score, SVM_model


# =============================================================================
# DEEP LEARNING MODELS
# =============================================================================
# Define Feedforward Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_evaluate_fnn(X_train_vectorized, X_test_vectorized, y_train, y_test, config):
    """Train and evaluate a Feedforward Neural Network"""
    print("\n===== Feedforward Neural Network Model =====")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train_vectorized.toarray(), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_vectorized.toarray(), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.int64).to(device)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.int64).to(device)
    
    # Create DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=config.fnn_batch_size, shuffle=True)
    
    # Initialize model
    model = NeuralNetwork(input_dim=X_train_vectorized.shape[1])
    model.to(device)
    print(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.fnn_learning_rate)
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    
    # Training loop
    for epoch in range(config.fnn_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Record metrics
        train_losses.append(running_loss/len(train_loader))
        train_accuracies.append(100 * correct / total)
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{config.fnn_epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%')
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.fnn_epochs+1), train_losses, 'r-')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.fnn_epochs+1), train_accuracies, 'b-')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('fnn_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, NN_model_predictions = torch.max(outputs, 1)
    
    # Accuracy
    NN_model_score = accuracy_score(y_test_encoded, NN_model_predictions.cpu().numpy())
    NN_model_score = round(NN_model_score*100, 2)
    print(f'FNN model accuracy: {NN_model_score}%')
    
    # Generate probabilities for ROC curve
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    
    # ROC curve
    y_test_binarized = label_binarize(y_test_encoded, classes=np.unique(y_train_encoded))
    n_classes = y_test_binarized.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    class_names = {0: "negative", 1: "neutral", 2: "positive"}
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Feedforward Neural Network Model')
    plt.legend(loc="lower right")
    plt.savefig('fnn_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC AUC score
    roc_auc_NN_model = roc_auc_score(y_test_binarized, probs, average='weighted', multi_class='ovr')
    
    # Convert predictions back to original labels
    decoded_predictions = label_encoder.inverse_transform(NN_model_predictions.cpu().numpy())
    decoded_y_test = label_encoder.inverse_transform(y_test_encoded)
    
    # Classification report with decoded labels
    report = classification_report(decoded_y_test, decoded_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop('support', axis=1)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title('Classification Report of Feedforward Neural Network Model')
    plt.savefig('fnn_classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # F1 score
    f1_NN_model_report = df_report.loc['weighted avg', 'f1-score']
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(decoded_y_test, decoded_predictions)
    plt.title('Confusion Matrix of Feedforward Neural Network Model')
    plt.savefig('fnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return NN_model_score, f1_NN_model_report, roc_auc_NN_model, model

# Text dataset class for CNN and LSTM
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)

# Define CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, max_length):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.fc = nn.Linear(64 * (max_length - 2 * 3 + 2), num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_evaluate_cnn(X_train, X_test, y_train, y_test, config):
    """Train and evaluate a Convolutional Neural Network"""
    print("\n===== Convolutional Neural Network Model =====")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Tokenize and pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    
    max_length = max(max(len(x) for x in X_train_sequences), max(len(x) for x in X_test_sequences))
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')
    
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding
    
    # Create DataLoader
    train_data = TextDataset(X_train_padded, y_train_encoded)
    train_loader = DataLoader(train_data, batch_size=config.cnn_batch_size, shuffle=True)
    
    # Initialize model
    model = TextCNN(vocab_size, config.cnn_embed_dim, 3, max_length)
    model.to(device)
    print(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.cnn_learning_rate)
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    
    # Training loop
    for epoch in range(config.cnn_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Record metrics
        train_losses.append(running_loss/len(train_loader))
        train_accuracies.append(100 * correct / total)
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{config.cnn_epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%')
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.cnn_epochs+1), train_losses, 'r-')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.cnn_epochs+1), train_accuracies, 'b-')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('cnn_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test_padded, dtype=torch.long).to(device))
        _, CNN_model_predictions = torch.max(outputs, 1)
    
    # Accuracy
    CNN_model_score = accuracy_score(y_test_encoded, CNN_model_predictions.cpu().numpy())
    CNN_model_score = round(CNN_model_score*100, 2)
    print(f'CNN model accuracy: {CNN_model_score}%')
    
    # Generate probabilities for ROC curve
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    
    # ROC curve
    y_test_binarized = label_binarize(y_test_encoded, classes=np.unique(y_train_encoded))
    n_classes = y_test_binarized.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    class_names = {0: "negative", 1: "neutral", 2: "positive"}
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Convolutional Neural Network Model')
    plt.legend(loc="lower right")
    plt.savefig('cnn_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC AUC score
    roc_auc_CNN_model = roc_auc_score(y_test_binarized, probs, average='weighted', multi_class='ovr')
    
    # Convert predictions back to original labels
    decoded_predictions = label_encoder.inverse_transform(CNN_model_predictions.cpu().numpy())
    decoded_y_test = label_encoder.inverse_transform(y_test_encoded)
    
    # Classification report with decoded labels
    report = classification_report(decoded_y_test, decoded_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop('support', axis=1)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title('Classification Report of Convolutional Neural Network Model')
    plt.savefig('cnn_classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # F1 score
    f1_CNN_model_report = df_report.loc['weighted avg', 'f1-score']
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(decoded_y_test, decoded_predictions)
    plt.title('Confusion Matrix of Convolutional Neural Network Model')
    plt.savefig('cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return CNN_model_score, f1_CNN_model_report, roc_auc_CNN_model, model, tokenizer, max_length

# Self-attention module for LSTM
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    
    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.attention = SelfAttention(hidden_dim*2 if bidirectional else hidden_dim)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        attn_output = self.attention(lstm_output)
        out = self.fc(self.dropout(attn_output))
        return out

def train_evaluate_lstm(X_train, X_test, y_train, y_test, config):
    """Train and evaluate an LSTM model"""
    print("\n===== LSTM Model =====")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Tokenize and pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    
    max_length = max(max(len(x) for x in X_train_sequences), max(len(x) for x in X_test_sequences))
    X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')
    
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding
    
    # Create DataLoader
    train_data = TextDataset(X_train_padded, y_train_encoded)
    train_loader = DataLoader(train_data, batch_size=config.lstm_batch_size, shuffle=True)
    
    # Initialize model
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=config.lstm_embed_dim,
        hidden_dim=config.lstm_hidden_dim,
        output_dim=3,
        n_layers=config.lstm_layers,
        bidirectional=True,
        dropout=config.lstm_dropout
    )
    model.to(device)
    print(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lstm_learning_rate)
    
    # Training metrics
    train_losses = []
    train_accuracies = []
    
    # Training loop
    for epoch in range(config.lstm_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Record metrics
        train_losses.append(running_loss/len(train_loader))
        train_accuracies.append(100 * correct / total)
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{config.lstm_epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%')
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.lstm_epochs+1), train_losses, 'r-')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.lstm_epochs+1), train_accuracies, 'b-')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('lstm_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test_padded, dtype=torch.long).to(device))
        _, LSTM_model_predictions = torch.max(outputs, 1)
    
    # Accuracy
    LSTM_model_score = accuracy_score(y_test_encoded, LSTM_model_predictions.cpu().numpy())
    LSTM_model_score = round(LSTM_model_score*100, 2)
    print(f'LSTM model accuracy: {LSTM_model_score}%')
    
    # Generate probabilities for ROC curve
    probs = F.softmax(outputs, dim=1).cpu().numpy()
    
    # ROC curve
    y_test_binarized = label_binarize(y_test_encoded, classes=np.unique(y_train_encoded))
    n_classes = y_test_binarized.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    class_names = {0: "negative", 1: "neutral", 2: "positive"}
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of LSTM Model')
    plt.legend(loc="lower right")
    plt.savefig('lstm_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC AUC score
    roc_auc_LSTM_model = roc_auc_score(y_test_binarized, probs, average='weighted', multi_class='ovr')
    
    # Convert predictions back to original labels
    decoded_predictions = label_encoder.inverse_transform(LSTM_model_predictions.cpu().numpy())
    decoded_y_test = label_encoder.inverse_transform(y_test_encoded)
    
    # Classification report with decoded labels
    report = classification_report(decoded_y_test, decoded_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop('support', axis=1)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title('Classification Report of LSTM Model')
    plt.savefig('lstm_classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # F1 score
    f1_LSTM_model_report = df_report.loc['weighted avg', 'f1-score']
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(decoded_y_test, decoded_predictions)
    plt.title('Confusion Matrix of LSTM Model')
    plt.savefig('lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return LSTM_model_score, f1_LSTM_model_report, roc_auc_LSTM_model, model

def train_evaluate_bert(X_train, X_test, y_train, y_test, config):
    """Train and evaluate a BERT model"""
    print("\n===== BERT Transformer Model =====")
    
    # Set device and random seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    random.seed(config.bert_seed_val)
    np.random.seed(config.bert_seed_val)
    torch.manual_seed(config.bert_seed_val)
    torch.cuda.manual_seed_all(config.bert_seed_val)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_pretrained_model, 
        do_lower_case=config.bert_do_lower_case
    )
    
    # Tokenize training data
    input_ids_train = []
    attention_masks_train = []
    
    for sent in X_train:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=config.bert_add_special_tokens,
            max_length=config.bert_seq_length,
            pad_to_max_length=config.bert_pad_to_max_length,
            return_attention_mask=config.bert_return_attention_mask,
            return_tensors=config.bert_return_tensors,
        )
        
        input_ids_train.append(encoded_dict['input_ids'])
        attention_masks_train.append(encoded_dict['attention_mask'])
    
    input_ids_train = torch.cat(input_ids_train, dim=0)
    attention_masks_train = torch.cat(attention_masks_train, dim=0)
    labels_train = torch.tensor(y_train_encoded)
    
    # Tokenize test data
    input_ids_test = []
    attention_masks_test = []
    
    for sent in X_test:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=config.bert_add_special_tokens,
            max_length=config.bert_seq_length,
            pad_to_max_length=config.bert_pad_to_max_length,
            return_attention_mask=config.bert_return_attention_mask,
            return_tensors=config.bert_return_tensors,
        )
        
        input_ids_test.append(encoded_dict['input_ids'])
        attention_masks_test.append(encoded_dict['attention_mask'])
    
    input_ids_test = torch.cat(input_ids_test, dim=0)
    attention_masks_test = torch.cat(attention_masks_test, dim=0)
    labels_test = torch.tensor(y_test_encoded)
    
    # Create DataLoaders
    train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.bert_batch_size
    )
    
    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=RandomSampler(test_dataset),
        batch_size=config.bert_batch_size
    )
    
    # Load pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(
        config.bert_pretrained_model,
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
    )
    
    model.to(device)
    print(f"BERT model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(),
                     lr=config.bert_learning_rate,
                     eps=config.bert_epsilon
                     )
    
    total_steps = len(train_dataloader) * config.bert_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training metrics
    loss_values = []
    accuracy_values = []
    
    # Training loop
    for epoch_i in range(config.bert_epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {config.bert_epochs} ========')
        
        # Training
        model.train()
        total_loss = 0
        total_accuracy = 0
        
        for step, batch in enumerate(train_dataloader):
            # Progress update
            if step % 40 == 0 and not step == 0:
                print(f'  Batch {step:>5,} of {len(train_dataloader):>5,}')
            
            # Unpack batch and move to device
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Reset gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            
            # Get loss and logits
            loss = outputs[0]
            total_loss += loss.item()
            
            logits = outputs[1]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_accuracy += flat_accuracy(logits, label_ids)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
        
        # Calculate average metrics
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_accuracy = total_accuracy / len(train_dataloader)
        
        # Store metrics
        loss_values.append(avg_train_loss)
        accuracy_values.append(avg_train_accuracy)
        
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Average training accuracy: {avg_train_accuracy:.2f}")
    
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.bert_epochs+1), loss_values, 'r-')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.bert_epochs+1), accuracy_values, 'b-')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('bert_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluation
    print('\nEvaluating BERT model on test data...')
    model.eval()
    
    predictions = []
    true_labels = []
    
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask
            )
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.append(logits)
        true_labels.append(label_ids)
    
    # Flatten predictions and labels
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)
    
    # Accuracy
    BERT_model_score = accuracy_score(flat_true_labels, flat_predictions)
    BERT_model_score = round(BERT_model_score*100, 2)
    print(f'BERT model accuracy: {BERT_model_score}%')
    
    # Convert predictions back to original labels
    decoded_predictions = label_encoder.inverse_transform(flat_predictions)
    decoded_y_test = label_encoder.inverse_transform(flat_true_labels)
    
    # Classification report
    report = classification_report(decoded_y_test, decoded_predictions, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.drop('support', axis=1)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
    plt.title('Classification Report of BERT Model')
    plt.savefig('bert_classification_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # F1 score
    f1_BERT_model_report = df_report.loc['weighted avg', 'f1-score']
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(decoded_y_test, decoded_predictions)
    plt.title('Confusion Matrix of BERT Model')
    plt.savefig('bert_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC curve
    y_test_binarized = label_binarize(flat_true_labels, classes=np.unique(y_train_encoded))
    n_classes = y_test_binarized.shape[1]
    
    # Convert logits to probabilities
    predictions_tensor = torch.tensor(np.vstack(predictions))
    probabilities = torch.softmax(predictions_tensor, dim=-1).numpy()
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], probabilities[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    class_names = {0: "negative", 1: "neutral", 2: "positive"}
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of BERT Model')
    plt.legend(loc="lower right")
    plt.savefig('bert_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC AUC score
    roc_auc_BERT_model = roc_auc_score(y_test_binarized, probabilities, average='weighted', multi_class='ovr')
    
    return BERT_model_score, f1_BERT_model_report, roc_auc_BERT_model, model

# =============================================================================
# MODEL COMPARISON
# =============================================================================
def compare_models(model_results):
    """Compare all models based on accuracy, F1-score, and ROC-AUC"""
    # Sort by accuracy in descending order
    model_results_acc = model_results.sort_values(by='Accuracy', ascending=False)
    model_results_acc.reset_index(drop=True, inplace=True)
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Accuracy', y='Model', data=model_results_acc, palette='viridis')
    plt.title('Model Comparison by Accuracy')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Models')
    
    # Annotate bars
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.5, p.get_y() + p.get_height()/2, f'{width:.2f}%', 
                ha='left', va='center')
    
    plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sort by F1 score in descending order
    model_results_f1 = model_results.sort_values(by='F1 Score', ascending=False)
    model_results_f1.reset_index(drop=True, inplace=True)
    
    # Plot F1 score comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='F1 Score', y='Model', data=model_results_f1, palette='viridis')
    plt.title('Model Comparison by F1 Score')
    plt.xlabel('F1 Score')
    plt.ylabel('Models')
    
    # Annotate bars
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.2f}', 
                ha='left', va='center')
    
    plt.savefig('model_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Sort by ROC AUC score in descending order (if available)
    if 'ROC AUC' in model_results.columns:
        model_results_auc = model_results.sort_values(by='ROC AUC', ascending=False)
        model_results_auc.reset_index(drop=True, inplace=True)
        
        # Plot ROC AUC comparison
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='ROC AUC', y='Model', data=model_results_auc, palette='viridis')
        plt.title('Model Comparison by ROC AUC Score')
        plt.xlabel('ROC AUC Score')
        plt.ylabel('Models')
        
        # Annotate bars
        for p in ax.patches:
            width = p.get_width()
            plt.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.2f}', 
                    ha='left', va='center')
        
        plt.savefig('model_roc_auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Return the best model by accuracy
    return model_results_acc.iloc[0]['Model']


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function to execute the financial sentiment analysis"""
    print("Starting Financial Sentiment Analysis...")
    
    # Initialize configuration
    config = Config()
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load and preprocess data
    df = load_data(config)
    df = clean_data(df)
    df = preprocess_text(df)
    
    # Visualize data
    visualize_data(df, config)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, X_train_vectorized, X_test_vectorized, vectorizer = prepare_data(df, config)
    
    # Train and evaluate baseline models
    null_accuracy, null_f1, = train_evaluate_null_model(df)
    lr_accuracy, lr_f1, lr_roc_auc, lr_model = train_evaluate_logistic_regression(
        X_train_vectorized, X_test_vectorized, y_train, y_test)
    dt_accuracy, dt_f1, dt_roc_auc, dt_model = train_evaluate_decision_tree(
        X_train_vectorized, X_test_vectorized, y_train, y_test)
    rf_accuracy, rf_f1, rf_model = train_evaluate_random_forest(
        X_train_vectorized, X_test_vectorized, y_train, y_test)
    svm_accuracy, svm_model = train_evaluate_svm(
        X_train_vectorized, X_test_vectorized, y_train, y_test)
    
    # Train and evaluate deep learning models
    fnn_accuracy, fnn_f1, fnn_roc_auc, fnn_model = train_evaluate_fnn(
        X_train_vectorized, X_test_vectorized, y_train, y_test, config)
    cnn_accuracy, cnn_f1, cnn_roc_auc, cnn_model, tokenizer, max_length = train_evaluate_cnn(
        X_train, X_test, y_train, y_test, config)
    lstm_accuracy, lstm_f1, lstm_roc_auc, lstm_model = train_evaluate_lstm(
        X_train, X_test, y_train, y_test, config)
    bert_accuracy, bert_f1, bert_roc_auc, bert_model = train_evaluate_bert(
        X_train, X_test, y_train, y_test, config)
    
    # Compile results
    model_results = pd.DataFrame({
        'Model': ['Null Model', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM',
                 'Feedforward Neural Network', 'Convolutional Neural Network', 'LSTM', 'BERT'],
        'Accuracy': [null_accuracy, lr_accuracy, dt_accuracy, rf_accuracy, svm_accuracy,
                    fnn_accuracy, cnn_accuracy, lstm_accuracy, bert_accuracy],
        'F1 Score': [null_f1, lr_f1, dt_f1, rf_f1, None,
                   fnn_f1, cnn_f1, lstm_f1, bert_f1],
        'ROC AUC': [None, lr_roc_auc, dt_roc_auc, None, None,
                  fnn_roc_auc, cnn_roc_auc, lstm_roc_auc, bert_roc_auc]
    })
    
    # Compare models
    best_model = compare_models(model_results)
    print(f"\nBest model by accuracy: {best_model}")
    
    # Save results to CSV
    model_results.to_csv('model_comparison_results.csv', index=False)
    print("Results saved to model_comparison_results.csv")
    
    print("Financial Sentiment Analysis completed successfully.")


if __name__ == "__main__":
    main()