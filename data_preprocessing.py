# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:26:29 2024

@author: win 10
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import re
import string
from wordcloud import WordCloud
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from scikitplot.metrics import plot_confusion_matrix, plot_roc
from scipy import interp
import scipy
print(scipy.__version__)

try:
    df = pd.read_csv('mental_disorders_reddit.csv', encoding='ISO-8859-1')
except UnicodeDecodeError:
    df = pd.read_csv('mental_disorders_reddit.csv', encoding='cp1252')


df.head()
print(df.shape)
df.isnull().sum()
df=df.dropna(how='any')
df['subreddit'].value_counts()


labels =['BPD', 'bipolar', 'depression', 'Anxiety', 'mentalillness','schizophrenia']
sizes = [233119, 167032, 156708,  46666,  44249, 20280]
custom_colours = ['b', 'g','r','c','m','y']

plt.figure(figsize=(20, 6), dpi=227)
plt.subplot(1, 2, 1)
plt.pie(sizes, labels = labels, textprops={'fontsize': 10}, startangle=140, 
       autopct='%1.0f%%', colors=custom_colours, explode=[0,0,0,0,0.05,0])

#plt.subplot(1, 2, 2)
#sns.barplot(x = df['subreddit'].unique(), y = df['subreddit'].value_counts(), palette= 'viridis')

plt.show()





#=========filter 500 samples for these four classes ['BPD', 'bipolar', 'depression', 'Anxiety']==================

import pandas as pd

def load_data(file_path, encoding='ISO-8859-1'):
    try:
        # Load the CSV file with specified encoding
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError as e:
        print(f"Failed to decode with {encoding}: {e}")
        return None

def filter_sample_and_combine(data):
    # Combine 'title' and 'selftext' into a new column 'combined'
    data['combined'] = data['title'].fillna('') + " " + data['selftext'].fillna('')

    # Define the classes to filter and the number of samples per class
    classes = ['BPD', 'bipolar', 'depression', 'Anxiety']
    samples_per_class = 500

    # Initialize an empty DataFrame to store sampled data
    filtered_data = pd.DataFrame()

    # Loop through each class, filter and sample the data
    for subreddit in classes:
        # Filter data for the current class
        class_data = data[data['subreddit'] == subreddit]

        # Sample the data
        # Use `min` to handle cases where there are less than 6000 samples available
        sampled_data = class_data.sample(n=min(samples_per_class, len(class_data)), random_state=42)

        # Append the sampled data to the filtered_data DataFrame
        filtered_data = pd.concat([filtered_data, sampled_data], ignore_index=True)

    # Select only the 'combined' and 'subreddit' columns to save
    final_data = filtered_data[['combined', 'subreddit']]

    # Save the filtered and sampled data to a new CSV file
    final_data.to_csv('combined_samples.csv', index=False)
    print("Data saved to 'combined_samples.csv' successfully.")

# Main execution block
if __name__ == "__main__":
    file_path = 'mental_disorders_reddit.csv'
    data = load_data(file_path)
    if data is not None:
        print("Data filtered successfully.")
        filter_sample_and_combine(data)
    else:
        print("Failed to filter the data.")


#=================count_samples_per_class================================


    
import pandas as pd

def count_samples_per_class(file_path, column_name='subreddit'):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Define the classes you're interested in
    classes = ['BPD', 'bipolar', 'depression', 'Anxiety']

    # Count the number of samples for each class
    class_counts = data[column_name].value_counts().reindex(classes, fill_value=0)

    # Print the count of each class
    print("Number of samples per class:")
    print(class_counts)

# Example usage:
file_path = 'output_combined_samples.csv'  # Replace 'input.csv' with the path to your CSV file
count_samples_per_class(file_path)



#========================data cleaning==========================


import pandas as pd
import re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import contractions
import unicodedata
# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path, encoding='utf-8'):
    # Load the CSV file with specified encoding
    return pd.read_csv(file_path, encoding=encoding)

def normalize_unicode(text):
    # Normalize Unicode characters to the NFC form
    return unicodedata.normalize('NFC', text)

def remove_url(text):
    re_url = re.compile('https?://\S+|www\.\S+')
    return re_url.sub('', text)


def remove_punc(text):
    exclude = string.punctuation
    return text.translate(str.maketrans('', '', exclude))

def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in words if word not in stop_words)


def perform_stemming(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

def expand_contractions(text):
    return contractions.fix(text)

def reencode_text(text, encoding='utf-8'):
    # Encode the text to bytes, then decode it back to string with the specified encoding
    return text.encode(encoding, errors='ignore').decode(encoding)

def clean_text(data):
    # Combine 'title' and 'selftext' into a new column 'combined'
    data['combined'] = data['combined'].fillna('')
    
    # Apply encoding transformation
    data['combined'] = data['combined'].apply(lambda x: reencode_text(x))
    
    # Apply contractions expansion
    data['combined'] = data['combined'].apply(expand_contractions)
    
    # Apply Unicode normalization
    data['combined'] = data['combined'].apply(normalize_unicode)
    
    # Apply cleaning functions
    data['combined'] = data['combined'].apply(remove_url)
    data['combined'] = data['combined'].apply(remove_punc)
    data['combined'] = data['combined'].apply(remove_stopwords)
    data['combined'] = data['combined'].apply(lambda x: x.lower())  # Convert to lowercase
    data['combined'] = data['combined'].apply(perform_stemming)  # Apply stemming
        
    return data


# Example usage
file_path = 'output_combined_samples.csv'  # Replace with your actual file path
data = load_data(file_path)
data = clean_text(data)



#==========word cloud===========================

text = " ".join(data[data['subreddit'] == 'BPD']['combined'])
plt.figure(figsize = (15, 10))
wordcloud = WordCloud(max_words=500, height= 800, width = 1500,  background_color="black", colormap= 'viridis').generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


text = " ".join(data[data['subreddit'] == 'bipolar']['combined'])
plt.figure(figsize = (15, 10))
wordcloud = WordCloud(max_words=500, height= 800, width = 1500,  background_color="black", colormap= 'viridis').generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


text = " ".join(data[data['subreddit'] == 'depression']['combined'])
plt.figure(figsize = (15, 10))
wordcloud = WordCloud(max_words=500, height= 800, width = 1500,  background_color="black", colormap= 'viridis').generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

text = " ".join(data[data['subreddit'] == 'Anxiety']['combined'])
plt.figure(figsize = (15, 10))
wordcloud = WordCloud(max_words=500, height= 800, width = 1500,  background_color="black", colormap= 'viridis').generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


