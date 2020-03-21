import json
import pandas as pd
import numpy as np
import re
from konlpy.tag import Komoran
from konlpy.tag import Twitter
from time import time
import pickle
import os
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from konlpy.tag import Kkma,Okt
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from utils.time_checking import timecheck

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# base_path = os.path.abspath('./')
data_path = './data/doc_set.xlsx'
stopwords_path = './data/korean_stopwords.txt'
stopwords = pd.read_csv(stopwords_path)['stopwords'].tolist()
data = pd.read_excel(data_path)
data.head()


# @timecheck
def preprocessing(content, remove_stopwords=False):
    # remove_special_character
    tech_text = re.sub('[^가-힣A-z0-9]', ' ', content)
    tech_text = re.sub(r"\s{2,}", " ", tech_text)
    tech_text = re.sub("\n", " ", tech_text)

    if remove_stopwords:
        tech_words = tech_text.split()
        tech_words = [w for w in tech_words if (w not in stopwords)]
        clean_text = ' '.join(tech_words)
    else:
        clean_text = tech_text

    return tech_text

def TensorTokenizer(data):
    tokenizer = Tokenizer()
    clean_train_content = []
    for txt in data['content']:
        clean_train_content.append(preprocessing(txt, remove_stopwords=True))
    clean_data = pd.DataFrame({'content':clean_train_content, 'new_class':data['new_class'], 'small_class':data['new_small_class']})
    tokenizer.fit_on_texts(clean_train_content)
    text_sequences = tokenizer.text_to_sequences(clean_train_content)

twitter = Okt()

clean_train_content = []
for txt in data['content']:
    clean_train_content.append(preprocessing(txt, remove_stopwords=True))

twitter.pos(clean_train_content[0])

doc0 = [" ".join()]

" ".join(["".join(w) for w, t in twitter.pos(s) if t not in ['Number', "Foreign"]])
["".join(w) for w, t in twitter.pos(s) for s in sent_tokenize(clean_train_content[0:3])]

[w for w, t in twitter.pos(clean_train_content[100]) if t in ["Noun"]]

clean_train_content[0]
len(np.unique(data.big_class))
len(np.unique(data.small_class))
len(np.unique(data.new_class))
len(np.unique(data.new_small_class))
