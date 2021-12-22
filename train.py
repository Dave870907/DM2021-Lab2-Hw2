import pandas as pd
import json
from tqdm import tqdm
import csv
from sklearn.metrics import accuracy_score

import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification

from transformers import Trainer, TrainingArguments
import numpy as np
import random

from sklearn.model_selection import train_test_split

import re
import string
import nltk
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

train = []
test = []
path2 = './dm2021-lab2-hw2/data_identification.csv'
with open(path2, newline='') as f:
    rows = csv.reader(f)
    for row in tqdm(rows): 
        if row[1]=='train':
            train.append(row[0])
        elif row[1] == 'test':
            test.append(row[0])

from collections import defaultdict

emotion = defaultdict(list)
path3 = './dm2021-lab2-hw2/emotion.csv'
with open(path3, newline='') as f:
    rows = csv.reader(f)
    for row in rows:
        emotion[row[1]].append(row[0])
emotion_dict = dict(emotion)

emotion_df = pd.read_csv('dm2021-lab2-hw2/emotion.csv')
emotion_df = emotion_df.set_index('tweet_id').T.to_dict()

path = 'dm2021-lab2-hw2/tweets_DM2.json'
with open(path,'r') as f:
    data = json.load(f)

df_data = pd.DataFrame(data)

def hashTag(x):
    # j = json.load(x)
    return x['tweet']['hashtags']

def tweet_id(x):
    # j = json.load(x)
    return x['tweet']['tweet_id']

def text(x):
    # j = json.load(x)
    return x['tweet']['text']

df_data['hashTag'] = df_data['_source'].apply(lambda x : hashTag(x))
df_data['tweet_id'] = df_data['_source'].apply(lambda x :tweet_id(x))
df_data['text'] = df_data['_source'].apply(lambda x:text(x))

df_data = df_data.drop(['_source','_type','_index','_crawldate'],axis=1)

df_data_train = df_data[df_data['tweet_id'].isin(train)]
print(len(df_data_train))
df_data_test = df_data[df_data['tweet_id'].isin(test)]
print(len(df_data_test))

tqdm.pandas(desc = 'pandas bar')

def give_emotion(x):
    return emotion_df[x]['emotion']

df_data_train['label'] = df_data_train['tweet_id'].progress_apply(lambda x :give_emotion(x))

with open('./train.pkl','wb') as f:
    pickle.dump(df_data_train,f)
with open('./test.pkl','wb') as f:
    pickle.dump(df_data_test,f)

# df_data_train = pd.read_pickle('HW_train.pkl')
# df_data_test = pd.read_pickle('HW_test.pkl')

df_data_train = df_data_train.sample(n = 10000)

label_dict = {}
label_name = list(df_data_train['label'].unique())
def generate_label():
    

    for index, label in enumerate(label_name):
        label_dict[label] = index

    df_data_train['num_label'] = df_data_train.label.replace(label_dict)

generate_label()
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

set_seed(1)

model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample
max_length = 512
# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

def read_data(test_size=0.15):
  # download & load 20newsgroups dataset from sklearn's repos
  
  documents = list(df_data_train['text'])
  labels = list(df_data_train['num_label'])
  # split into training & testing a return data as well as label names
  return train_test_split(documents, labels, test_size=test_size), label_name
  
# call the function
(train_texts, valid_texts, train_labels, valid_labels), target_names = read_data()

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = Dataset(train_encodings, train_labels)
valid_dataset = Dataset(valid_encodings, valid_labels)


# load the model and pass to CUDA
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names))


def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=400,               # log & save weights each logging_steps
    save_steps=400,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

trainer.train()
trainer.evaluate()

model_path = "./DM2-bert-base-uncased2"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)


def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return target_names[probs.argmax()]


def inference():

    df_data_test['label'] = df_data_test['text'].apply(lambda x: get_prediction(x))
    submission = df_data_test.drop(['_score','hashTag','text'],axis = 1)
    submission.columns = ['id','emotion']
    submission.to_csv('submission.csv',index=False)

inference()
