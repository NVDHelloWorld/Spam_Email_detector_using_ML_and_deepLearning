# Importing necessary libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Loading Dataset (Replace 'spam_ham_dataset.csv' with your dataset path)
data = pd.read_csv('spam_ham_dataset.csv')

# Checking dataset shape
print("Shape of the dataset:", data.shape)

# Plotting spam label distribution
sns.countplot(x='spam', data=data)
plt.show()

# Downsampling to balance the dataset
ham_msg = data[data.spam == 0]
spam_msg = data[data.spam == 1]
ham_msg = ham_msg.sample(n=len(spam_msg), random_state=42)
balanced_data = ham_msg.append(spam_msg).reset_index(drop=True)
plt.figure(figsize=(8, 6))
sns.countplot(data=balanced_data, x='spam')
plt.title('Distribution of Ham and Spam email messages after downsampling')
plt.xlabel('Message types')
plt.show()

# Text Preprocessing (without removing stopwords)
def remove_punctuations(text):
    punctuations_list = string.punctuation
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')
balanced_data['text'] = balanced_data['text'].apply(lambda x: remove_punctuations(x))

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(balanced_data['text'])
train_sequences = tokenizer.texts_to_sequences(balanced_data['text'])
max_len = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              optimizer='adam')

# Callbacks for model training
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

# Train the model
history = model.fit(train_sequences, balanced_data['spam'], validation_split=0.2,
                    epochs=20, batch_size=32, callbacks=[lr, es])

# Preprocess new data for prediction
new_data=pd.read_csv("spam_ham_dataset.csv")
new_emails=new_data["text"].str.replace("Subject:"," ")

#new_emails = ["Hello, this is a legitimate email.", "Congratulations! You've won a prize!","Hello how are you","( see attached file : hplnol 09 . xls )"]

# Preprocess new email texts
new_emails = [remove_punctuations(text) for text in new_emails]
new_sequences = tokenizer.texts_to_sequences(new_emails)
new_sequences = pad_sequences(new_sequences, maxlen=max_len, padding='post', truncating='post')

# Make predictions
predictions = model.predict(new_sequences)

# Interpret predictions
threshold = 0.5
predicted_labels = (predictions > threshold).astype(int)

# Display results
for text, label in zip(new_emails, predicted_labels):
    label_str = "spam" if label == 1 else "ham"
    print(f"Email Text: '{text}' => Predicted Label: {label_str}")
