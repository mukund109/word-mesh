#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 04:26:25 2018

@author: mukund
"""
import numpy as np
import os
import nltk

project_path = os.path.join(os.path.dirname(os.getcwd()), 'wordmesh')
if not (os.path.isdir(os.path.join(project_path, 'tokenizers'))):
    print('Downloading nltk resource required for POS tagging...')
    nltk.download('punkt', download_dir=project_path, quiet=True)



RELATIONSHIP_METRICS = ['cooccurence']
DISCRETE_PROPERTIES = ['POS']

def _cooccurence_score(text, word1, word2):    
    l1 = _find_all(text, word1)
    l2 = _find_all(text, word2)
    square_sum =0
    for i in l1:
        for j in l2:
            square_sum = square_sum + (i-j)**2

    return (square_sum**(1/2))/(len(l1)*len(l2))

def _find_all(text, substring):
    loc = text.find(substring)
    if loc == -1:
        return []
    else:
        sub_locs = _find_all(text[loc+1:], substring)
        return [loc] + [loc+i+1 for i in sub_locs]

def cooccurence_similarity_matrix(text, wordlist):
    score_func = lambda x,y: _cooccurence_score(text, wordlist[int(x)], wordlist[int(y)])
    vscore_func = np.vectorize(score_func)
    return np.fromfunction(vscore_func, shape=[len(wordlist)]*2)

