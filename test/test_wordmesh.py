#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:22:42 2018

@author: mukund
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'wordmesh'))

from wordmesh import Wordmesh
import unittest

with open('sample_text.txt') as f:
    test_text = f.read()


class TestWordmesh(unittest.TestCase):
    
    def test_default_constructor(self):
        wm = Wordmesh(test_text)
        self.assertEqual(['stories', 'hall', 'hours', 'dystopias', 'close',
                          'new mother', 'mrs', 'feel', 'sexuality',
                          'eliciting deep', 'indulgence despite', 'woman',
                          'turning', 'wild', 'goes', 'crafting compelling'], 
                         wm.keywords)
        
if __name__ == '__main__':
    #unittest.main()
#    wm = Wordmesh(test_text, pos_filter=['NOUN','ADJ','VERB','PROPN'], 
#                  keyword_extractor='textrank', extract_ngrams=False, num_keywords=70)
#    print(wm.keywords)
#    wm.set_clustering_criteria(by='cooccurence')
#    wm.set_fontcolor(by='pos_tag')
#    wm.generate_embeddings()
#    wm.save_as_html()
#    wm.save_as_html(filename='animated',force_directed_animation=True)
    with open('sample_speech.txt') as f:
         trump_text = f.read()
        
    wm = Wordmesh(trump_text, dimensions=(900, 1500), pos_filter=['ADJ'],keyword_extractor='tf', 
                  extract_ngrams=False, num_keywords=80, lemmatize=False)
    
    wm.set_clustering_criteria('scores')
    wm.set_fontsize('scores')
    wm.set_fontcolor('scores','Reds')
    wm.generate_embeddings()
    wm.save_as_html(force_directed_animation=True)
    print('done')