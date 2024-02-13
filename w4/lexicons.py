
import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import langid
import re
from afinn import Afinn  # Import AFINN
from textblob import TextBlob  # Import TextBlob

file_path = r'C:\Users\student\Desktop\advaiwork\w4\resources\sentiment_analysis_results.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

file_path = r'C:\Users\student\Desktop\advaiwork\w4\resources\youtube_comments_labels.csv'
labeled_data = pd.read_csv(file_path)

def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


# Categorise sentiments for VADER, TextBlob, and AFINN
df['VADER Sentiment'] = df['VADER Compound'].apply(categorize_sentiment)
df['AFINN Sentiment'] = df['AFINN Score'].apply(categorize_sentiment)
df['TextBlob Sentiment'] = df['TextBlob Polarity'].apply(categorize_sentiment)




# Map sentiment labels to numerical values (e.g., 'Positive' -> 2, 'Neutral' -> 1, 'Negative' -> 0)
sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
labeled_data['GroundTruth'] = labeled_data['Sentiment'].map(sentiment_mapping)




# Create a mapping for sentiment labels used in VADER, TextBlob, and AFINN
sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}




# Convert VADER sentiment labels to numerical values
df['VADER Sentiment'] = df['VADER Sentiment'].map(sentiment_mapping)




# Convert TextBlob sentiment labels to numerical values
df['TextBlob Sentiment'] = df['TextBlob Sentiment'].map(sentiment_mapping)




# Convert AFINN sentiment scores to sentiment labels
def categorize_afinn_sentiment(score):
    if score > 0:
        return 2  # Positive
    elif score < 0:
        return 0  # Negative
    else:
        return 1  # Neutral




df['AFINN Sentiment'] = df['AFINN Score'].apply(categorize_afinn_sentiment)


from sklearn.metrics import classification_report, f1_score  # Add 'f1_score' to the import statement


# Define the sentiment classes
classes = [0, 1, 2]


# Define the sentiment classes and the methods
sentiment_methods = ['VADER', 'TextBlob', 'AFINN']
classes = [0, 1, 2]  # 0: Negative, 1: Neutral, 2: Positive




# Initialize dictionaries to store precision, recall, and F1 scores
precision_scores = {}
recall_scores = {}
f1_scores = {}
micro_f1_scores = {}  # Micro-average F1 scores
macro_f1_scores = {}  # Macro-average F1 scores




# Ensure labels are of the same data type (integer)
labeled_data['GroundTruth'] = labeled_data['GroundTruth'].astype(int)




# Calculate the classification reports for each sentiment method
for method in sentiment_methods:
    report = classification_report(labeled_data['GroundTruth'], df[method + ' Sentiment'], labels=classes, output_dict=True)
   
    for label in classes:
        label_str = str(label)
        precision_scores[f'{method} Precision for {label_str}'] = report[label_str]['precision']
        recall_scores[f'{method} Recall for {label_str}'] = report[label_str]['recall']
        f1_scores[f'{method} F1 Score for {label_str}'] = report[label_str]['f1-score']




    # Calculate micro-average and macro-average F1 scores
    micro_f1 = f1_score(labeled_data['GroundTruth'], df[method + ' Sentiment'], average='micro', labels=classes)
    macro_f1 = f1_score(labeled_data['GroundTruth'], df[method + ' Sentiment'], average='macro', labels=classes)
   
    micro_f1_scores[f'{method} Micro F1'] = micro_f1
    macro_f1_scores[f'{method} Macro F1'] = macro_f1


# Print the metrics
for method in sentiment_methods:
    for label in classes:
        label_str = str(label)
        print(f"{method} Precision for {label_str}:", precision_scores[f'{method} Precision for {label_str}'])
        print(f"{method} Recall for {label_str}:", recall_scores[f'{method} Recall for {label_str}'])
        print(f"{method} F1 Score for {label_str}:", f1_scores[f'{method} F1 Score for {label_str}'])

    print(f"{method} Micro F1:", micro_f1_scores[f'{method} Micro F1'])
    print(f"{method} Macro F1:", macro_f1_scores[f'{method} Macro F1'])
