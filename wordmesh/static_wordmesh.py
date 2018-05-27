#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 02:20:39 2018

@author: mukund
"""
import gensim

class StaticWordmesh():
    def __init__(self, text, dimensions=(100, 100),
                 keyword_extractor='textrank', lemmatize=True,
                 pos_filter=('NN', 'JJ', 'RB', 'VB')):
             
        """Word mesh object for generating and drawing STATIC 
        wordmeshes/wordclouds.
        
        Parameters
        ----------
        text : string
            The string of text that needs to be summarized
            
        dimensions : tuple, optional 
            The desired dimensions (height, width) of the wordcloud in pixels
            
        keyword_extractor : string, optional
            You can choose one from the following: ['textrank']
            
        lemmatize : bool, optional
            Whether the text needs to be lemmatized before keywords are
            extracted from it
            
        pos_filter : tuple, optional
            A POS filter can be applied on the keywords. By default, only nouns,
            adjectives, adverbs and verbs can be keywords.
            More more information on the tags used, visit:
            https://www.clips.uantwerpen.be/pages/mbsp-tags
            
        Returns
        -------
        StaticWordMesh
            A word mesh object 
        
        """
        
        self.text = text
        self.extractor = keyword_extractor
        self.dimension_ratio_ = dimensions[0]/float(dimensions[1])
        self.resolution = dimensions
        self.lemmatize = True
        self.keyword_extractor = keyword_extractor
        self.pos_filter = pos_filter
        self._extract_keywords()

    def _extract_keywords(self):
        self.keywords = []
        self.scores = []
        word_scores = None
        
        
        if self.keyword_extractor == 'textrank':
            word_scores = gensim.summarization.keywords(self.text, split=True, 
                                                    scores=True, 
                                                    lemmatize=self.lemmatize,
                                                    pos_filter=self.pos_filter)
        else:
            raise NotImplementedError("Only 'textrank' has been implemented")
            
        for word_score in word_scores:
            self.keywords.append(word_score[0])
            self.scores.append(word_score[1])
