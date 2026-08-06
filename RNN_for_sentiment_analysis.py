import pandas as pd

df = pd.read_csv(r"C:\Users\arsha\Desktop\prime ai&ml\deep_learning_3\IMDB Dataset.csv")
# print(df.head())
# print(df.shape)
# print(df.drop_duplicates(inplace=True))
# print(df.shape)
# print(df.isnull().sum())

"""Pre-Processing"""

#lowercase
df["review"] = df["review"].str.lower()

#removing urls
import re

# sample_text = "abc is the word , abc"  #abc => xyz
# new_text = re.sub("abc" , "xyz" , sample_text)

def remove_urls(text):
    text = re.sub(r"http\S+" , "" , text)
    return text

df["review"] = df["review"].apply(remove_urls)

#remove punctuations
def remove_punctuations(text):
    text = re.sub(r"[^A-Za-z0-9\s]" , "" , text)
    return text

df["review"]  =df["review"].apply(remove_punctuations)

#remove html
def remove_html(text):
    text = re.sub(r"<.*?>" , "" , text)
    return text
df["review"] = df["review"].apply(remove_html)

import nltk
"""
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")

    for word in tokens:
        if word in stop_words:
            text = text.replace(word , "")
    return text

df["review"] = df["review"].apply(remove_stopwords)



#stemming
from nltk.stem import PorterStemmer

def stemming(text):
    ps = PorterStemmer()
    stemmed_words = []

    tokens = word_tokenize(text)
    for token in tokens:
        stemmed_tokens = ps.stem(token)
        stemmed_words.append(stemmed_tokens)

    return " ".join(stemmed_words)

df["review"] = df["review"].apply(stemming)

# print(df.head())

#encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["sentiment"] = le.fit_transform(df["sentiment"])
y = df["sentiment"]
# print(df["sentiment"])

# #vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(max_features = 5000)
X = tf.fit_transform(df["review"])

# print(X)

"""DATASET AND DATALOADER"""

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state=42
)

print(type(X_train))

"""TENSORDATASET AND DATALOADERS"""
import torch
from torch.utils.data import TensorDataset , DataLoader

X_train = X_train.toarray()  #convert from sparse matrix to numpy aaray
X_test = X_test.toarray()  #convert from sparsre to numpy array

train_set = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train.values).float()
)
test_set = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test.values).float()
)
train_loader = DataLoader(train_set , batch_size = 64 , shuffle = True)
test_loader  = DataLoader(test_set , batch_size = 64 , shuffle =True)
                   

"""Build our RNN"""

import torch.nn as nn
import torch.optim as optim

class RNN(nn.Module):
    def __init__(self , input_size , hidden_size = 128 , num_layers = 1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #RNN layer
        self.rnn = nn.RNN(input_size , hidden_size , num_layers , batch_first = True)

        #fully connected layers
        self.fc = nn.Linear(hidden_size , 1)

    def forward(self , x):
        #optional => shape(num of layers , batch size , hidden size)
        h0 = torch.zeros(self.num_layers , x.size(0) , self.hidden_size)

        out , _ = self.rnn(x , h0)
        # 1st value = hidden state of all timesteps => (batch , seq_len , hidden_size)
        #2nd value = final hidden state of last timestep

        out = self.fc(out[: , -1 , :])  #-1 for last step
        return out

    input_size = X_train.shape[1]

input_size = X_train.shape[1]

model = RNN(input_size)

criterion = nn.BCELoss()  #binary classification loss when output has 2 value
optimizer = optim.Adam(model.parameters())

"""TRAINING THE RNN"""

epochs = 10
for epoch in range(epochs):
    model.train()

    for Xb , yb in train_loader:
        optimizer.zero_grad()

        Xb = Xb.unsqueeze(1)  #add singleton dimension
        outputs = model(Xb)  # (batch size , 1)

        outputs = torch.sigmoid(outputs.squeeze())  #batch_size => probability

        loss = criterion(outputs , yb)  #compute loss
        loss.backward()  #back propogation
        optimizer.step()  #update weights


    print(f"epoch is {epoch + 1} / {epochs} and loss is {loss.item()}")

"""Evaluation"""

model.eval()
total_value = 0
correct_value = 0

with torch.no_grad():
    for Xb , yb in test_loader:
        Xb = Xb.unsqueeze(1)

        outputs = model(Xb)
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()

        correct_value += (predicted == yb).sum().item()
        total_value += yb.size(0)

    print("total values : " , total_value)
    print("Correct value : " , correct_value)
    print(f"accuracy is {correct_value/total_value * 100}")