# OS & JSON
import os
import json
# Google API
from google.oauth2 import service_account
from googleapiclient.discovery import build
# VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# MPL & Pandas
import matplotlib.pyplot as plt
import pandas as pd
import langid
import re

# Youtube API

api_key = "AIzaSyDtIy1jOB9e6MxzBThIbYkJBg2uX8Kf7j0"

youtube = build("youtube", "v3", developerKey=api_key)
video_id = '67ICSjD6s5o'

# Function to fetch all English comments from the YouTube video
def get_all_english_video_comments(youtube, **kwargs):
    comments = []
   
    while True:
        results = youtube.commentThreads().list(**kwargs).execute()


        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
           
            # Detect the language of the comment using langid
            lang, _ = langid.classify(comment)
            if lang == 'en':
                comments.append(comment)


        # Check if there are more pages of comments
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
        else:
            break


    return comments


# Preprocess comments by removing special characters and lowercasing
def preprocess_comments(comments):
    preprocessed_comments = []
    for comment in comments:
        # Remove special characters and convert to lowercase
        comment = re.sub(r'[^A-Za-z0-9 ]+', '', comment).lower()
        preprocessed_comments.append(comment)
    return preprocessed_comments


# Get all English comments from the YouTube video
comments = get_all_english_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')


# Preprocess comments
preprocessed_comments = preprocess_comments(comments)


# Sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(comment) for comment in preprocessed_comments]


# Create a DataFrame for visualization
data = {'Comments': preprocessed_comments, 'Sentiment Polarity': sentiments}
df = pd.DataFrame(data)


# Print data with no limit
pd.set_option('display.max_rows', None)
print(df)


# Plot sentiment analysis results
plt.figure(figsize=(8, 6))
plt.hist([s['compound'] for s in sentiments], bins=[-1, -0.5, 0, 0.5, 1], color='lightblue')
plt.title(f'Sentiment Analysis for Video {video_id}')
plt.xlabel('Sentiment Polarity (Compound Score)')
plt.ylabel('Number of Comments')
plt.show()

# def get_all_video_comments(youtube, **kwargs):
#     comments = []
#     while True:
#         results = youtube.commentThreads().list(**kwargs).execute()
#         for item in results['items']:
#             comment = item ['snippet']['topLevelComment']['snippet']['textDisplay']
#             comments.append(comment)
        
#         if 'nextPageToken' in results:
#             kwargs['pageToken'] = results['nextPageToken']
#         else:
#             break

#     return comments

# comments = get_all_video_comments(youtube, part='snippet',videoId=video_id, textFormat='plainText')

# def get_video_comments(youtube, **kwargs):
#     comments = []
#     results = youtube.commentThreads().list(**kwargs).execute()

#     while results:
#         for item in results['items']:
#             comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
#             comments.append(comment)

#         results = youtube.commentThreads().list_next(results,kwargs)
    
#     return comments

# # Get comments from video
# comments = get_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')

