# Script to retrieve the Jigsaw text representations from OpenAI API
import math
import pickle

import pandas as pd
import numpy as np

from tqdm import tqdm
from openai import OpenAI



def dump_pkl(content, filename):
    with open(filename, "wb") as file:
        pickle.dump(content, file)


def load_pkl(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def read_key(filename):
    with open(filename) as f:
        return f.read()


client = OpenAI(api_key=read_key('openai.key'))

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


# -

def get_embeddings(texts, model="text-embedding-3-large"):
    texts = [text.replace("\n", " ") for text in texts]
    embs = [x.embedding for x in  client.embeddings.create(input = texts, model=model).data]
    return embs


def load_jigsaw_raw(PATH='../continuous-debiasing/data/jigsaw/train.csv'):
    df = pd.read_csv(PATH)
    
    label_set = [
        'buddhist', 
         'christian', 
         'hindu', 
         'jewish', 
         'muslim', 
         'other_religion'
    ]
    
    rows = []
    for _, row in tqdm(df.iterrows()):
        for label in label_set:
            if not math.isnan(row[label]) and row[label] > 0.:
                rows.append(row)
                break
    return rows

# batch-wise API calling
def chunk_list(lst, chunk_size):
    chunked_list = [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    if len(lst) % chunk_size != 0:
        chunked_list[-1] = lst[-(len(lst) % chunk_size):]
    return chunked_list

data = load_jigsaw_raw()





batched_rows = chunk_list(data, chunk_size=8)

# generate dataset with GPT-4 embeddings
dataset = []
for row_batch in tqdm(batched_rows):
    texts = [x['comment_text'] for x in row_batch]
    embs = get_embeddings(texts)
    dataset.extend([(r, e) for r, e in zip(row_batch, embs)])

dump_pkl(dataset, "data/jigsaw/data_openai.pkl")


