#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Download news articles from the urls given in News Aggregator Dataset
    ref: https://archive.ics.uci.edu/ml/datasets/News+Aggregator
"""

import io, os, sys
import pandas as pd
import numpy as np
import requests
import zipfile
import time
import csv
import re
import multiprocessing
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

    
# import en_core_web_lg

#------------------------------------------------------------------------------#
# Configuration
#------------------------------------------------------------------------------#
# URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip'

# NEWS_AGGREGATOR_DIR = 'news_aggregator'
# ARTICLES_DIR        = 'articles'

# DEBUG               = True
DEBUG               = False
RE_D                = re.compile('\d')
TOP_N               = 20
TFIDF_THRESHOLD     = 0.1

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
#--------------------------------------------------------------------#

def has_numbers(string, regex_pattern=RE_D):
    return bool(regex_pattern.search(string))

#--------------------------------------------------------------------#

def get_keywords(corpus, nlp = '', top_n=TOP_N, threshold=TFIDF_THRESHOLD):
    t0 = time.time()

    # vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=STOPWORDS) # 这里考虑了双词问题 所以下面采用了名词判断方法
    # 我们暂时用单个词
    vec = TfidfVectorizer(ngram_range=(1, 1), stop_words=STOPWORDS)
    X = vec.fit_transform(corpus)
    print(f"X.shape: {X.shape}")

    terms = np.array(vec.get_feature_names())

    tfidfs, keywords = [], []
    all_keywords = set()
    N = len(corpus)

    for i, text in enumerate(corpus):
        D = X.getrow(i)
        D = np.squeeze(D.toarray())
        ind = np.argsort(D)[::-1]
        ind = ind[:top_n] # 取tfidf高的topn词

        D = D[ind]
        D = D[D > threshold] # 取tfidf在一定阈值以上的词
        kw = terms[ind][:len(D)] 

        # doc = nlp(text) # 词性分析
        # noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split())>1]

        # D  = [d for d, w in zip(D, kw) if (w in noun_phrases or len(w.split())==1) and not has_numbers(w)] 
        # kw = [w for w in kw if (w in noun_phrases or len(w.split())==1) and not has_numbers(w)] 
        # 这里是指如果词在词组中，仅考虑名词词组情况，并且不含数字
        # 这里我们暂时改为仅考虑不含数字
        D  = [d for d, w in zip(D, kw) if not has_numbers(w)] 
        kw = [w for w in kw if not has_numbers(w)] 

        tfidfs.append(D)
        keywords.append(kw)

        for word in kw:
            all_keywords.add(word)

        if i%1000==0 and i>0:
            print(f"({os.getpid()}) Items processed: {i :,}/{N:,}; {(time.time()-t0)/60 :.1f} minutes")

    return terms, keywords, tfidfs, all_keywords

if __name__ == "__main__":
    with open('./data/hit_stopwords.txt', 'r', encoding='utf-8') as fr:
        stopwords = fr.readlines()
        stopwords = [x.strip() for x in stopwords]
        STOPWORDS = set([stopword for stopword in stopwords if stopword])

    with open('./data/hit_stopwords.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    data = [x.strip() for x in lines]
    data = [list(jieba.cut(x)) for x in data]
    corpus = [' '.join(x) for x in data]
    terms, keywords, tfidfs, all_keywords = get_keywords(corpus)
    print(keywords)