# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:04:49 2024

@author: User
"""

from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from torch.optim import AdamW  
# Load the dataset
df = pd.read_csv('cleaned_dataset.csv')

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['disease'], test_size=0.2)

# Initialize the tokenizer from the RoBERTa model
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

# Tokenize the texts
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Encode the labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

# Convert to torch tensors
train_seq = torch.tensor(train_encodings['input_ids'])
train_mask = torch.tensor(train_encodings['attention_mask'])
train_y = torch.tensor(train_labels_encoded, dtype=torch.long)

val_seq = torch.tensor(val_encodings['input_ids'])
val_mask = torch.tensor(val_encodings['attention_mask'])
val_y = torch.tensor(val_labels_encoded, dtype=torch.long)

# Create the DataLoader
batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Load the RoBERTa model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification", num_labels=len(label_encoder.classes_))

# Define the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 20
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training and validation loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_loss / len(train_dataloader)
    
    model.eval()
    predictions, true_labels = [], []
    for batch in val_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs.logits
        logits = logits.detach().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)
    
    predictions = np.argmax(np.concatenate(predictions, axis=0), axis=1)
    true_labels = np.concatenate(true_labels, axis=0)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    print(f'Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')




# Save the model and tokenizer
model.save_pretrained("D:/running/save/path_to_save_model_directory")
tokenizer.save_pretrained("D:/running/save/path_to_save_tokenizer_directory")

#============confusion matrix===============


from sklearn.utils import resample

# Define a function to balance samples per class
def balance_data(X, y, num_samples):
    X_balanced = []
    y_balanced = []
    classes = np.unique(y)
    for cls in classes:
        X_cls = X[y == cls]
        X_cls_resampled = resample(X_cls, replace=True, n_samples=num_samples, random_state=42)
        X_balanced.extend(X_cls_resampled)
        y_balanced.extend([cls] * num_samples)
    return np.array(X_balanced), np.array(y_balanced)

# Balance the validation data
balanced_val_texts, balanced_val_labels = balance_data(val_texts, val_labels, 80)

# Tokenize the balanced validation data
balanced_val_encodings = tokenizer(list(balanced_val_texts), truncation=True, padding=True, max_length=128)
balanced_val_seq = torch.tensor(balanced_val_encodings['input_ids'])
balanced_val_mask = torch.tensor(balanced_val_encodings['attention_mask'])
balanced_val_y = torch.tensor(label_encoder.transform(balanced_val_labels), dtype=torch.long)

# Create the DataLoader for balanced validation data
balanced_val_data = TensorDataset(balanced_val_seq, balanced_val_mask, balanced_val_y)
balanced_val_sampler = SequentialSampler(balanced_val_data)
balanced_val_dataloader = DataLoader(balanced_val_data, sampler=balanced_val_sampler, batch_size=batch_size)

# Evaluation loop with balanced validation data
model.eval()
predictions, true_labels = [], []
for batch in balanced_val_dataloader:
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs.logits
    logits = logits.detach().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

predictions = np.argmax(np.concatenate(predictions, axis=0), axis=1)
true_labels = np.concatenate(true_labels, axis=0)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
accuracy = accuracy_score(true_labels, predictions)
conf_matrix = confusion_matrix(true_labels, predictions)

print(f'Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')

#============= End confusion matrix============================


#===print the class labels==========
# Fit label encoder and get class names
label_encoder.fit(train_labels)
class_names = label_encoder.classes_

# Print class names and labels
print("Class names:", class_names)
print("Class labels:", label_encoder.transform(class_names))


#===============test from input===============
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("save/path_to_save_model_directory")
tokenizer = AutoTokenizer.from_pretrained("save/path_to_save_tokenizer_directory")

# Define a function to classify text
def classify_text(text):
    # Tokenize the text
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Map the predicted class index back to the original label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label

# Test the model on new text data
new_text = "and anhedonia right now with impulses I'm having trouble controlling - I want to spend money despite having none, I'm chain smoking like crazy just to have some routine, I got a new job which is a stressor, I missed an exam and need to get a medical note asap + troubles with my ex and a guy I'm kind of seeing (who has bipolar too) is piling on to it all with a dash of family problems for good measure."
predicted_label = classify_text(new_text)
print("Predicted label:", predicted_label)



#=============Load model and results on unseen data======================

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score

# Load the saved model and tokenizer
model_path = "D:/Paper 10/running_shvm/save/path_to_save_model_directory"
tokenizer_path = "D:/Paper 10/running_shvm/save/path_to_save_tokenizer_directory"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Load the test dataset
data_path = "sampled_mental_disorders_cleaned.csv"  # Update this path to your test data location
data = pd.read_csv(data_path)

# Preprocess and tokenize the data
inputs = tokenizer(data['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Predict using the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Map the disease names to integers
label_map = {classname: i for i, classname in enumerate(model.config.id2label.values())}
true_labels = data['disease'].map(label_map)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions.numpy())
print(f"Accuracy: {accuracy:.4f}")
