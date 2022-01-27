import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
import torch.nn.functional as F
from collections import Counter
import logging
from tqdm import tqdm
import argparse
from copy import deepcopy
from transformers import RobertaModel, AutoTokenizer, RobertaPreTrainedModel, RobertaConfig, AdamW
from sklearn.metrics import f1_score

class Vocab:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        self.idx2word = {value: key for key,value in self.word2idx.items()}
        self.length = 4
    def _add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.length
            self.idx2word[self.length] = word
            self.length+=1
    def _encode_sentence(self, sentence):
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence]
    def _decode_sentence(self, tokens):
        return [self.idx2word[token] for token in tokens]
    def build_vocab(self, sentences, min_freq = 1):
        word_freq = {}
        for sent in sentences:
            for token in sent:
              if token not in word_freq:
                word_freq[token] = 1
              else:
                word_freq[token]+= 1
        for word, freq in word_freq.items():
          if freq>=min_freq:
            self._add_word(word)
    def encode(self, sentences):
        return [self._encode_sentence(sent) for sent in sentences]
    def __len__(self):
        return self.length
    def __call__(self, word):
        return self.word2idx[word]

class EntityVocab:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.idx2label = {}
        self.length = 0
    def _add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.length
            self.idx2token[self.length] = token
            if token.startswith('B-') or token.startswith('I-'):
              token = token[2:]
            self.idx2label[self.length] = token
            self.length+=1
    def _encode_token(self, tokens):
        return [self.token2idx.get(token, 0) for token in tokens]
    def _decode_token(self, tokens):
        return [self.idx2token[token] for token in tokens]
    def _decode_label(self, tokens):
        return [self.idx2label[token] for token in tokens]
    def encode(self, tokens):
        return [self._encode_token(token) for token in tokens]
    def build(self,tokens):
        for tok in tokens:
            for token in tok:
                self._add_token(token)
    def __len__(self):
        return self.length
    def __call__(self, token):
        return self.token2idx[token] 

class NER_Dataset(Dataset):
    def __init__(self, sentences, tokens):
        self.sentences = sentences
        self.tokens = tokens
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, index):
        return self.sentences[index], self.tokens[index]

def create_class_weight(labels_dict):
  total = np.sum(list(labels_dict.values()))
  keys  = labels_dict.keys()
  class_weight = dict()
  num_classes = len(labels_dict)
  for key in keys:
      score = round(total / (num_classes * labels_dict[key]+total/10), 2)
      class_weight[key] = score
  return class_weight    

def reset_logger(logger):
  for handler in logger.handlers[:]:
    logger.removeHandler(handler)

  for f in logger.filters[:]:
    logger.removeFilters(f)

def read_data(data_path):
    sentences = []
    entities = []
    with open(data_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        sent = []
        tok = []
        for line in lines:
            tokens = line.strip().split()
            if len(tokens)==0:
                sentences.append(sent)
                entities.append(tok)
                sent=[]
                tok=[]
                continue
            if len(tokens)!=2:
                print(tokens)
                continue
            sent.append(tokens[0])
            tok.append(tokens[1])
    return sentences, entities

def encode_data(tokenizer, sentences, labels):
  encoded_sentences = []
  encoded_labels = []
  attention_masks = []
  labels_masks = []
  for sent, label in zip(sentences, labels):
    encoded_sent = [0]
    encoded_lab = [-1]
    attention_mask = [1]
    for word, lab in zip(sent, label):
      tokens = tokenizer.encode(word)
      encoded_sent.extend(tokens[1:-1])
      encoded_lab.append(lab)
      encoded_lab.extend([-1 for i in range(len(tokens)-3)])
      attention_mask.extend([1 for i in range(len(tokens)-2)])
    encoded_sent.append(2)
    encoded_lab.append(-1)
    attention_mask.append(1)
    label_mask = [1 if lab>=0 else 0 for lab in encoded_lab]
    encoded_sentences.append(encoded_sent)
    encoded_labels.append(encoded_lab)
    attention_masks.append(attention_mask)
    labels_masks.append(label_mask)
  return {
      'encoded_sentences': encoded_sentences,
      'encoded_labels': encoded_labels,
      'attention_masks': attention_masks,
      'labels_masks': labels_masks
  }

class NER_with_PhoBERT_Dataset(Dataset):
  def __init__(self, data_dict):
    self.input_ids = data_dict['encoded_sentences']
    self.labels = data_dict['encoded_labels']
    self.attention_masks = data_dict['attention_masks']
    self.labels_masks = data_dict['labels_masks']
    self.length = len(self.input_ids)
  def __len__(self):
    return self.length
  def __getitem__(self, idx):
    return self.input_ids[idx], self.labels[idx], self.attention_masks[idx], self.labels_masks[idx]

def _collate_fn_PhoBERT(batch):
  input_ids, labels, attention_masks, labels_masks = zip(*batch)
  max_length = max([len(sent) for sent in input_ids])
  padded_input_ids = deepcopy(input_ids)
  padded_labels = deepcopy(labels)
  padded_attention_masks = deepcopy(attention_masks)
  padded_labels_masks = deepcopy(labels_masks)
  for j in range(len(input_ids)):
    padded_length = max_length-len(padded_input_ids[j])
    padded_input_ids[j].extend([0 for i in range(padded_length)])
    padded_labels[j].extend([-1 for i in range(padded_length)])
    padded_attention_masks[j].extend([0 for i in range(padded_length)])
    padded_labels_masks[j].extend([0 for i in range(padded_length)])
  return torch.LongTensor(padded_input_ids), torch.LongTensor(padded_labels), torch.LongTensor(padded_attention_masks), torch.LongTensor(padded_labels_masks)

