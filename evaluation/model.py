import time
import os
import pandas as pd
import tqdm
import re
import torch
import json
# import torch_xla
# import torch_xla.core.xla_model as xm
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
# from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from string import punctuation
import re
from gensim.models import KeyedVectors
import numpy as np
from vocabulary import *

EPOCHS = 15
BATCH_SIZE = 32
EMBEDDING_SIZE = 300
VOCAB_SIZE = 0
TARGET_SIZE = 0
HIDDEN_SIZE = 32
LEARNING_RATE = 0.005
STACKED_LAYERS = 2
EMBEDDING_PATH = '../../dataset/google_news/GoogleNews-vectors-negative300.bin'
GLOVE_EMBEDDING = '../../embeddings/glove.6B.300d.txt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initiate_model_vocab(vocab, tag2idx):
    VOCAB_SIZE = len(vocab.word2index)
    TARGET_SIZE = len(tag2idx)
    
class LSTM(nn.Module):
  def __init__(self, vocab_size, hidden_size, target_size, stacked_layers, weights_matrix, bidirectional):
    super(LSTM, self).__init__()
    self.vocab_size     = vocab_size
    self.hidden_size    = hidden_size
    self.bidirectional  = bidirectional
    self.target_size    = target_size
    self.stacked_layers = stacked_layers
    num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
    self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
    self.embedding.weight.requires_grad = True

    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=self.stacked_layers, batch_first = True, dropout=0.2, bidirectional=bidirectional)
    
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p = 0.2)

    self.FC_concat1 = nn.Linear(2 * 2 * hidden_size if bidirectional else 2 * hidden_size, 128)
    self.FC_concat2 = nn.Linear(128, 64)
    self.FC_concat3 = nn.Linear(64, 32)

    for lin in [self.FC_concat1, self.FC_concat2]:
		    nn.init.xavier_uniform_(lin.weight)
		    nn.init.zeros_(lin.bias)

    self.output = nn.Linear(32, self.target_size)

    self.out = nn.Sequential(
			self.FC_concat1,
			self.relu,
			self.dropout,
			self.FC_concat2,
			self.relu,
      self.FC_concat3,
      self.relu,
			self.dropout,
			self.output
		)

  def forward_once(self, seq, hidden, seq_len):
    embedd_seq = self.embedding(seq)
    packed_seq = pack_padded_sequence(embedd_seq, lengths=seq_len, batch_first=True, enforce_sorted=False)
    output, (hidden, _) = self.lstm(packed_seq, hidden)
    return hidden

  def forward(self, input, premise_len, hypothesis_len):
    premise    = input[0]
    hypothesis = input[1]
    batch_size = premise.size(0)

    h0 = torch.zeros(self.stacked_layers*2 if self.bidirectional else self.stacked_layers, batch_size, self.hidden_size).to(device) # 2 for bidirection 
    c0 = torch.zeros(self.stacked_layers*2 if self.bidirectional else self.stacked_layers, batch_size, self.hidden_size).to(device)

    # hidden = self.init_hidden(batch_size)

    premise    = self.forward_once(premise, (h0, c0), premise_len)
    hypothesis = self.forward_once(hypothesis, (h0, c0), hypothesis_len)
    
    combined_outputs  = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis), dim=2)

    return self.out(combined_outputs[-1])

  # def init_hidden(self, batch_size):
  #   weight = next(self.parameters()).data
  #   hidden = (weight.new(self.stacked_layers, batch_size, self.hidden_size).zero_().to(device),
  #                     weight.new(self.stacked_layers, batch_size, self.hidden_size).zero_().to(device))
  #   return hidden

def multi_acc(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc



def train(model, train_loader, val_loader, criterion, optimizer):  
  total_step = len(train_loader)

  for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    for val in train_loader:
      sentence_pairs, labels = map(list, zip(*val))

      premise_seq    = [torch.tensor(seq[0]).long().to(device) for seq in sentence_pairs]
      hypothesis_seq = [torch.tensor(seq[1]).long().to(device) for seq in sentence_pairs]
      batch = len(premise_seq)

      premise_len    = list(map(len, premise_seq))
      hypothesis_len = list(map(len, hypothesis_seq))

      temp = pad_sequence(premise_seq + hypothesis_seq, batch_first=True)
      premise_seq    = temp[:batch, :]
      hypothesis_seq = temp[batch:, :]
      labels         = torch.tensor(labels).long().to(device)

      model.zero_grad()
      prediction = model([premise_seq, hypothesis_seq], premise_len, hypothesis_len)

      loss = criterion(prediction, labels)
      acc  = multi_acc(prediction, labels)

      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)
    model.eval()
    total_val_acc  = 0
    total_val_loss = 0
    with torch.no_grad():
      for val in val_loader:
        sentence_pairs, labels = map(list, zip(*val))

        premise_seq    = [torch.tensor(seq[0]).long().to(device) for seq in sentence_pairs]
        hypothesis_seq = [torch.tensor(seq[1]).long().to(device) for seq in sentence_pairs]
        batch = len(premise_seq)

        premise_len    = list(map(len, premise_seq))
        hypothesis_len = list(map(len, hypothesis_seq))

        temp = pad_sequence(premise_seq + hypothesis_seq, batch_first=True)
        premise_seq    = temp[:batch, :]
        hypothesis_seq = temp[batch:, :]

        premise_seq    = premise_seq.to(device)
        hypothesis_seq = hypothesis_seq.to(device)
        labels         = torch.tensor(labels).long().to(device)

        model.zero_grad()
        prediction = model([premise_seq, hypothesis_seq], premise_len, hypothesis_len)
        
        loss = criterion(prediction, labels)
        acc  = multi_acc(prediction, labels)

        total_val_loss += loss.item()
        total_val_acc  += acc.item()

    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    torch.cuda.empty_cache()

