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
    with open('sample_speech.txt') as f:
         trump_text = f.read()
        
    wm = Wordmesh(trump_text, dimensions=(900, 1500), keyword_extractor='textrank', 
                  extract_ngrams=False, num_keywords=50, lemmatize=True)
    wm.save_as_html(force_directed_animation=False)
    #wm.set_clustering_criteria('cooccurence')
    #wm.set_fontsize('scores')
    #m.set_fontcolor('scores','YlGnBu')
    
    
    print('done')