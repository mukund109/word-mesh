#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:22:42 2018

@author: mukund
"""
import sys
import os

#print(os.path.join(os.path.dirname(os.getcwd()), 'wordmesh'))
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'wordmesh'))

from wordmesh import StaticWordmesh
import unittest

with open('sample_text.txt') as f:
    test_text = f.read()


class TestWordmesh(unittest.TestCase):
    
    def test_static_default_constructor(self):
        wm = StaticWordmesh(test_text)
        self.assertEqual(['stories', 'hall', 'hours', 'dystopias', 'close',
                          'new mother', 'mrs', 'feel', 'sexuality',
                          'eliciting deep', 'indulgence despite', 'woman',
                          'turning', 'wild', 'goes', 'crafting compelling'], 
                         wm.keywords)
        
if __name__ == '__main__':
    #unittest.main()
    wm = StaticWordmesh(test_text, pos_filter=('JJ'))
    print(wm.keywords)
        
        
        
        