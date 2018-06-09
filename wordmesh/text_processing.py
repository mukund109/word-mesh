#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 02:56:08 2018

@author: mukund
"""
#Will throw an error if language model has not been downloaded
import en_core_web_sm

import textacy
import textacy.keyterms as keyterms
import numpy as np

def extract_terms_by_score(text, algorithm, num_terms, extract_ngrams, ngrams=(1,2)):
    """
    Need to add support for lemmatization
    """
    #convert raw text into spaCy doc
    doc = textacy.Doc(text, lang='en')
    
    if algorithm=='sgrank':
        ngrams = ngrams if extract_ngrams else (1,)
        keywords_scores = keyterms.sgrank(doc, ngrams=ngrams, n_keyterms=num_terms)
    elif (algorithm=='pagerank') | (algorithm=='textrank'):
        keywords_scores = keyterms.key_terms_from_semantic_network(doc,
                                                                   ranking_algo='pagerank',
                                                                   join_key_words=extract_ngrams,
                                                                   n_keyterms=num_terms)
    else:
        raise NotImplementedError()
       
    keywords = [i[0] for i in keywords_scores]
    scores = np.array([i[1] for i in keywords_scores])
    
    #get pos tags for keywords, if keywords are ngrams, the 
    #pos tag of the last word in the ngram is picked
    ending_tokens = [ngram.split(' ')[-1] for ngram in keywords]
    mapping = _get_pos_mapping(doc)
    pos_tags = [mapping[end] for end in ending_tokens]

    return keywords, scores, pos_tags
   
def _get_pos_mapping(doc):
    mapping = dict()
    for token in doc:
        mapping.update({token.lemma_:token.pos_})
    return mapping

def _get_frequency_mapping(doc):
    """
    Need to take args like 'ngrams', 'normalize'
    and pass them on to to_bag_of_terms
    """
    doc.to_bag_of_terms(as_strings=True)

def extract_terms_by_frequency(text, 
                               num_terms, 
                               pos_filter=['NOUN','ADJ','PROPN'],
                               filter_nums=True, 
                               extract_ngrams = True,
                               ngrams=(1,2)):
    """
    pos_filter : ({'NOUN','PROPN','ADJ','VERB','ADV','SYM',PUNCT'})
    """
    #convert raw text into spaCy doc
    doc = textacy.Doc(text, lang='en')
    
    #get the frequencies of the filtered terms
    ngrams = ngrams if extract_ngrams else (1,)
    frequencies = doc.to_bag_of_terms(ngrams,
                                      as_strings=True, 
                                      include_pos=pos_filter,
                                      filter_nums=filter_nums, 
                                      include_types=['PERSON','LOC','ORG'])
    
    #sort the terms based on the frequencies and 
    #choose the top num_terms terms
    frequencies = list(frequencies.items())
    frequencies.sort(key=lambda x: x[1], reverse=True)
    top_terms = frequencies[:num_terms]
    
    keywords = [tup[0] for tup in top_terms]
    scores = np.array([tup[1] for tup in top_terms])
    
    #get pos tags for keywords, if keywords are ngrams, the 
    #pos tag of the last word in the ngram is picked
    ending_tokens = [ngram.split(' ')[-1] for ngram in keywords]
    mapping = _get_pos_mapping(doc)
    pos_tags = [mapping[end] for end in ending_tokens]
    
    return keywords, scores, pos_tags
    