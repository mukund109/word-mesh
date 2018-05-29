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
FONTSIZE_BBW = 0.17
FONTSIZE_BBH = 0.25
# fontsize*FONTSIZE_BBW = Width of the bounding box of each character in a plotly graph


def _get_bb_dimensions(words, fontsizes, fontsize_to_bbw=FONTSIZE_BBW,
                       fontsize_to_bbh=FONTSIZE_BBH):

    num_words = len(words)

    num_chars = list(map(len, words))

    widths = [fontsize_to_bbw*fontsizes[i]*num_chars[i] for i in range(num_words)]
    heights = [fontsize_to_bbh*fontsizes[i] for i in range(num_words)]

    return np.array([widths, heights]).swapaxes(0, 1)

def _cooccurence_score(text, word1, word2): 
    text, word1, word2 = text.lower(), word1.lower(), word2.lower()
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

