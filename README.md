# Conversational AI for Mental Disorder Detection

This repository contains the implementation of a Conversational AI system designed to assist in the early detection and classification of mental disorders. The system utilizes transformer-based models, specifically Distil-RoBERTa and GPT-3.5, to engage users in natural conversations and analyze their responses for signs of mental health conditions like anxiety, borderline personality disorder (BPD), depression, and bipolar disorder.The project includes data preprocessing, model training, and a user-friendly interface.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Dataset](#datset)
5. [Structure](#structure)
6. [Results](#results)
7. [Contributing](#contributing)


## Overview
The project combines conversational AI with machine learning to create a diagnostic tool for mental health. The system uses GPT-3.5 to generate responses and Distil-RoBERTa to classify user inputs. It provides a user-friendly interface for interacting with the AI, supporting both text and voice inputs.

## Features
- **Natural Language Understanding**: Uses GPT-3.5 for generating conversational responses.
- **Mental Health Classification**: Detects mental health conditions with high accuracy.
- **Multimodal Input**: Supports text and voice inputs.
- **Data Security**: Ensures privacy and security of user data.

## Installation
To install and set up the project, follow these steps:

1. **Clone the Repository**:
   git clone [repository_url]
   cd [repository_name]
2. **Install Dependencies**:
   pip install -r requirements.txt
3. **Environment Setup**:
   Ensure you have access to a GPU for model training and inference.

## Datset
The project uses two main datasets for training:
1. **Conversational Dataset**: Used to train the GPT-3.5 model for generating human-like responses.
   [Mental Health Conversational Data](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data)
2. **Classification Dataset**: Used to train the Distil-RoBERTa model for classifying mental health conditions.
   [Mental Disorders Identification (Reddit NLP)](https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp)

## Structure
1.  **data_preprocessing.py**: Handles data cleaning and preparation.
2.  **classification_using_distilroberta.py**: Script for model training and evaluation.
3.  **main_file.py**: Main interface for user interaction with the AI system. GPT-Trained model link(ft:gpt-3.5-turbo-0613:bba-university:mental-disorder:8ebDCFla)
4.  **requirements.txt**: List of dependencies required for the project.

## Results
The system achieves high accuracy in classifying mental health conditions, with precision, recall, and F1-score around 95%. Detailed performance metrics and confusion matrices are available in the output logs.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, please open an issue to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

MIT License

Copyright (c) 2024 Diwakar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
   
