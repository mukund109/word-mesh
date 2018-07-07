#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 02:56:08 2018

@author: mukund
"""
#Will throw an error if language model has not been downloaded
try:
    print('Loading spaCy\'s language model...')
    import en_core_web_md
    NLP = en_core_web_md.load()
    print('Done')
except ModuleNotFoundError as e:
    msg = 'word-mesh relies on spaCy\'s pretrained language models '\
    'for tokenization, POS tagging and accessing word embeddings. \n\n'\
    'Download the \'en_core_web_md\' model by following the instructions '\
    'given on: \n\nhttps://spacy.io/usage/models \n'
    raise Exception(msg).with_traceback(e.__traceback__) 

import os
import textacy
from textacy.extract import named_entities
from textacy.keyterms import sgrank,key_terms_from_semantic_network
import numpy as np


FILE = os.path.dirname(__file__)
STOPWORDS = set(map(str.strip, open(os.path.join(FILE,'stopwords.txt')).readlines()))
NGRAM_LIMIT = 3

def _text_preprocessing(text):
    """
    Apostrophes not handled properly by spaCy
    https://github.com/explosion/spaCy/issues/685
    """
    return text.replace('’s', '').replace('’m', '')

def _text_postprocessing(doc, keywords, extract_ngrams=False):
    """
    named entities are converted to uppercase
    """
    ents = list(named_entities(doc))
    ents_lemma = [entity.lemma_ for entity in ents]
    
    #if keyword is named entity it will be replaced by its uppercase form
    for i,word in enumerate(keywords.copy()):
        try:
            index = ents_lemma.index(word)
            keywords[i] = ents[index].text
        except ValueError:
            continue
    
    #if there is a redundant space character, it will be removed
    if not extract_ngrams:
        for i,word in enumerate(keywords):
            keywords[i] = word.replace(' ','')
        
    return keywords

def _filter(keywords, scores, filter_stopwords, num_terms):
    #temporary -PRON-, stopwords and ngram filter
    #stopwords are allowed in multigrams
    tempkw, temps = [],[]
    stopwords = STOPWORDS if filter_stopwords else set()
    for i,kw in enumerate(keywords):
        if kw.find('-PRON-')==-1 and (kw not in stopwords) \
        and (kw!='') and (len(kw.split(' '))<=NGRAM_LIMIT) and (kw!='_'):
            tempkw.append(kw)
            temps.append(scores[i])
    
    if len(tempkw)>num_terms:
        tempkw, temps = tempkw[:num_terms], temps[:num_terms]
        
    return tempkw, temps

def extract_terms_by_score(text, algorithm, num_terms, extract_ngrams, 
                           ngrams=(1,2), lemmatize=True, filter_stopwords=True):
    
    if lemmatize:
        normalize = 'lemma'
    else :
        normalize = None
        
    #convert raw text into spaCy doc
    text = _text_preprocessing(text)
    doc = textacy.Doc(text, lang=NLP)
    
    #the number of extracted terms is twice num_keyterms since a lot of them 
    #are filtered out by the filter
    if algorithm=='sgrank':
        ngrams = ngrams if extract_ngrams else (1,)
        keywords_scores = sgrank(doc, normalize=normalize, 
                                 ngrams=ngrams, n_keyterms=num_terms*2)
        
    elif (algorithm=='pagerank') | (algorithm=='textrank'):
        keywords_scores = key_terms_from_semantic_network(doc, 
                                                          normalize=normalize,
                                                          edge_weighting='cooc_freq',
                                                          window_width=5,
                                                          ranking_algo='pagerank',
                                                          join_key_words=extract_ngrams,
                                                          n_keyterms=num_terms*2)

    else:
        keywords_scores = key_terms_from_semantic_network(doc,
                                                          normalize=normalize,
                                                          edge_weighting='cooc_freq',
                                                          window_width=5,
                                                          ranking_algo=algorithm,
                                                          join_key_words=extract_ngrams,
                                                          n_keyterms=num_terms*2)
       
    keywords = [i[0] for i in keywords_scores]
    scores = [i[1] for i in keywords_scores]

    keywords, scores = _filter(keywords, scores, filter_stopwords, num_terms)
    
    scores = np.array(scores)
    #get pos tags for keywords, if keywords are ngrams, the 
    #pos tag of the last word in the ngram is picked
    ending_tokens = [ngram.split(' ')[-1] for ngram in keywords]
    mapping = _get_pos_mapping(doc, normalize)
    pos_tags = [mapping[end] for end in ending_tokens]

    normalized_keywords = keywords.copy()
    keywords = _text_postprocessing(doc, keywords, extract_ngrams)
    return keywords, scores, pos_tags, normalized_keywords
   
def _get_pos_mapping(doc, normalize):
    mapping = dict()
    if normalize=='lemma':
        for token in doc:
                mapping.update({token.lemma_:token.pos_})
    elif normalize=='lower':
        for token in doc:
            mapping.update({token.lower_:token.pos_})
    else:
        for token in doc:
                mapping.update({token.text:token.pos_})
        
    return mapping

def _get_frequency_mapping(doc):
    """
    Need to take args like 'ngrams', 'normalize'
    and pass them on to to_bag_of_terms
    """
    doc.to_bag_of_terms(as_strings=True)

def normalize_text(text, lemmatize):
    if not lemmatize:
        return text
    
    text = _text_preprocessing(text)
    
    #convert raw text into spaCy doc
    text = _text_preprocessing(text)
    doc = textacy.Doc(text, lang=NLP)
    
    #pronouns need to be handled separately
    #https://github.com/explosion/spaCy/issues/962
    lemmatized_strings = []
    for token in doc:
        if token.lemma_ == '-PRON-':
            lemmatized_strings.append(token.lower_)
        else:
            lemmatized_strings.append(token.lemma_)
            
    normalized_text = ' '.join(lemmatized_strings)
    return normalized_text

def extract_terms_by_frequency(text, 
                               num_terms, 
                               pos_filter=['NOUN','ADJ','PROPN'],
                               filter_nums=True, 
                               extract_ngrams = True,
                               ngrams=(1,2), lemmatize=True,
                               filter_stopwords=True):
    """
    pos_filter : {'NOUN','PROPN','ADJ','VERB','ADV','SYM',PUNCT'}
    """
    if lemmatize:
        normalize = 'lemma'
    else :
        normalize = None
        
    #convert raw text into spaCy doc
    text = _text_preprocessing(text)
    doc = textacy.Doc(text, lang=NLP)
    
    #get the frequencies of the filtered terms
    ngrams = ngrams if extract_ngrams else (1,)
    if 'PROPN' in pos_filter:
        frequencies = doc.to_bag_of_terms(ngrams, normalize=normalize,
                                      as_strings=True, 
                                      include_pos=pos_filter,
                                      filter_nums=filter_nums, 
                                      include_types=['PERSON','LOC','ORG'])
    elif 'PROPN' not in pos_filter:
        frequencies = doc.to_bag_of_terms(ngrams, normalize=normalize,
                                          named_entities=False,
                                          as_strings=True, 
                                          include_pos=pos_filter,
                                          filter_nums=filter_nums)
    
    #sort the terms based on the frequencies and 
    #choose the top num_terms terms
    #NOTE: lots of redundant code here, cleanup required
    frequencies = list(frequencies.items())

    keywords = [tup[0] for tup in frequencies]
    scores = [tup[1] for tup in frequencies]
    
    #applying filter
    keywords,scores = _filter(keywords, scores, filter_stopwords, num_terms)
    
    
    frequencies = list(zip(keywords,scores))
    frequencies.sort(key=lambda x: x[1], reverse=True)
    top_terms = frequencies[:num_terms]
    
    keywords = [tup[0] for tup in top_terms]
    scores = np.array([tup[1] for tup in top_terms])
    
    #get pos tags for keywords, if keywords are multi-grams, the 
    #pos tag of the last word in the multi-gram is picked
    ending_tokens = [ngram.split(' ')[-1] for ngram in keywords]
    mapping = _get_pos_mapping(doc, normalize)
    pos_tags = [mapping[end] for end in ending_tokens]
    
    normalized_keywords = keywords.copy()
    keywords = _text_postprocessing(doc, keywords, extract_ngrams)
    return keywords, scores, pos_tags, normalized_keywords
    
            
def get_semantic_similarity_matrix(keywords):
    text = ' '.join(keywords)
    doc = textacy.Doc(text, lang=NLP)
    
    #split the doc into a list of spaCy's 'spans'
    spans = []
    i=0
    for term in keywords:
        delta = len(term.split(' '))
        spans.append(doc[i:i+delta])
        i = i+delta
    
    #find the similarity between each pair of spans
    similarity_scores = []
    for span1 in spans:
        t1_scores = []
        for i,span2 in enumerate(spans):
            t1_scores.append(span1.similarity(span2))
        similarity_scores.append(t1_scores)
        
    #the values of the returned matrix vary from 0 to 1,
    #a smaller number means that the words are similar in meaning
    return (1-np.array(similarity_scores))
