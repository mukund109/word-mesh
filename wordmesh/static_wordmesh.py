#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 02:20:39 2018

@author: mukund
"""
from text_processing import extract_terms_by_frequency, extract_terms_by_score
from utils import cooccurence_similarity_matrix as csm
import numpy as np
from sklearn.manifold import MDS
from utils import PlotlyVisualizer
from force_directed_model import ForceDirectedModel
import colorlover as cl

class StaticWordmesh():
    def __init__(self, text, dimensions=(500, 900),
                 keyword_extractor='textrank', num_keywords=15,
                 lemmatize=True, pos_filter=['NOUN','ADJ','PROPN'], 
                 extract_ngrams=True, filter_numbers=True):
             
        """Word mesh object for generating and drawing STATIC 
        wordmeshes/wordclouds.
        
        Parameters
        ----------
        text : string
            The string of text that needs to be summarized
            
        dimensions : tuple, optional 
            The desired dimensions (height, width) of the wordcloud in pixels
            
        keyword_extractor : string, optional
            You can choose one from the following: ['textrank', 'sgrank', 'tf']
            
        lemmatize : bool, optional
            Whether the text needs to be lemmatized before keywords are
            extracted from it
            
        pos_filter : tuple, optional
            A POS filter can be applied on the keywords ONLY when the 
            keyword_extractor has been set to 'tf'. By default, only nouns,
            adjectives and proper nouns can be keywords.
            More more information on the tags used, visit:
            https://www.clips.uantwerpen.be/pages/mbsp-tags
            
        Returns
        -------
        StaticWordMesh
            A word mesh object 
        
        """
        
        self.text = text
        self.dimension_ratio = dimensions[0]/float(dimensions[1])
        self.resolution = dimensions
        self.lemmatize = lemmatize
        self.keyword_extractor = keyword_extractor
        self.pos_filter = pos_filter
        self.extract_ngrams = extract_ngrams
        self.num_keywords = num_keywords
        self.filter_numbers = filter_numbers
        self._extract_keywords()
        self.set_fontsize()
        self.set_fontcolor()
        self.set_clustering_criteria()

    def _extract_keywords(self):

        if self.keyword_extractor == 'tf':
            
            self.keywords, self.scores, self.pos_tags = \
            extract_terms_by_frequency(self.text, self.num_keywords, 
                                       self.pos_filter, self.filter_numbers, 
                                       self.extract_ngrams)
        else:
            self.keywords, self.scores, self.pos_tags = \
            extract_terms_by_score(self.text, self.keyword_extractor,
                                   self.num_keywords, self.extract_ngrams)
            

    def set_fontsize(self, by='scores', custom_sizes=None, 
                     directly_proportional=True, regularization_factor=3):
        """
        This function can be used to pick a metric which decides the font size
        for each extracted keyword. By default, the font size is directly 
        proportional to the 'scores' assigned by the keyword extractor. 
        
        Fonts can be picked by: 'scores', 'word_frequency', 'random', None
        
        You can also choose custom font sizes by passing in a dictionary 
        of word:fontsize pairs using the argument custom_sizes
        
        Parameters
        ----------
        
        by : string or None, optional
            The metric used to assign font sizes. Can be None if custom sizes 
            are being used
        custom_sizes : dictionary or None, optional
            A dictionary with individual keywords as keys and font sizes as
            values. The dictionary should contain all extracted keywords (that 
            can be accessed through the keywords attribute). Extra words will
            be ignored
        directly_proportional : bool, optional
            Controls whether font sizes are directly or inversely proportional
            to the value of the chosen metric
        regularization_factor : int, optional
            Determines the ratio max(fontsizes)/min(fontsizes). Fontsizes are
            scaled linearly so as to achieve this ratio. This helps prevent 
            extreme values.
            
        Returns
        -------
        
        numpy array
            Array of normalized font sizes, normalized such that the maximum 
            is 1. There is a one-one correspondence between these and the 
            extracted keywords
        """
        
        if by=='scores' and directly_proportional:
            self.fontsizes_norm = self.scores/self.scores.sum()
            
        else:
            raise NotImplementedError()
            
        #applying regularization
        k = regularization_factor
        mx = self.fontsizes_norm.max()
        mn = self.fontsizes_norm.min()
        
        a = mx*(k-1)/((mx-mn)*k)
        b = mx*(mx-mn*k)/((mx-mn)*k)
        
        self.fontsizes_norm = a*self.fontsizes_norm + b
        self.fontsizes_norm = self.fontsizes_norm/self.fontsizes_norm.sum()
            
        return self.fontsizes_norm
            
    def set_fontcolor(self, by='random', custom_colors=None):
        """
        This function can be used to pick a metric which decides the font color
        for each extracted keyword. By default, the font size is assigned 
        randomly 
        
        Fonts can be picked by: 'random', 'word_frequency', None
        
        You can also choose custom font colors by passing in a dictionary 
        of word:fontcolor pairs using the argument custom_sizes, where 
        fontcolor is an (R, G, B) tuple
        
        Parameters
        ----------
        
        by : string or None, optional
            The metric used to assign font sizes. Can be None if custom colors 
            are being used
        custom_colors : dictionary or None, optional
            A dictionary with individual keywords as keys and font colors as
            values (these should be RGB tuples). The dictionary should contain
            all extracted keywords (that can be accessed through the keywords 
            attribute). Extra words will be ignored
            
        Returns
        -------
        
        numpy array
            A numpy array of shape (num_keywords, 3).
        """
        
        if by=='random':
            tone = np.random.choice(list(cl.flipper()['seq']['3'].keys()))
            self.fontcolors = np.random.choice(list(cl.flipper()['seq']\
                                                    ['3'][tone]), 
                                                    len(self.keywords))
            
        else:
            raise NotImplementedError()
            
        return self.fontcolors
            
    def set_clustering_criteria(self, by='cooccurence', 
                          custom_similarity_matrix=None):
        """
        This function can be used to define the criteria for clustering of
        different keywords in the wordcloud. By default, clustering is done
        based on the tendency of words to frequently occur together in the
        text i.e. the 'cooccurence' criteria is used for clustering
        
        The following pre-defined criteria can be used: 'cooccurence',
        'semantic_similarity', 'pos_tag'
        
        You can also define a custom criteria
        
        Parameters
        ----------
        
        by : string or None, optional
            The pre-defined criteria used to cluster keywords
            
        custom_similarity_matrix : numpy array or None, optional
            A 2-dimensional array with shape (num_keywords, num_keywords)
            The entry a[i,j] defines the similarity between keyword[i] and 
            keyword[j]. Words that are similar will be clustered together
            
        Returns
        -------
        
        numpy array
            the similarity_matrix, i.e., a numpy array of shape (num_keywords, 
            num_keywords).
        """
        if by=='cooccurence':
            self.similarity_matrix = csm(self.text, self.keywords)
            
        else:
            raise NotImplementedError()
            
        self._generate_embeddings()
        return self.similarity_matrix
    
    
    def _generate_embeddings(self, store_as_attribute=True):
        mds = MDS(2, dissimilarity='precomputed').\
                             fit_transform(self.similarity_matrix)
                           
        self._visualizer = PlotlyVisualizer(words = self.keywords,
                                            fontsizes_norm =self.fontsizes_norm, 
                                            height = self.resolution[0],
                                            width = self.resolution[1], 
                                            textcolors=self.fontcolors)
        
        bbd = self._visualizer.bounding_box_dimensions
        
        fdm = ForceDirectedModel(mds, bbd, num_iters=100)
        self.force_directed_model = fdm
        
        if store_as_attribute:
            self.embeddings = fdm.equilibrium_position()
            self._bounding_box_width_height = bbd
            self._mds = mds
        
        return fdm.equilibrium_position()
        
    def _get_all_fditerations(self, num_slides=100):
        all_pos = self.force_directed_model.all_centered_positions
        num_iters = self.force_directed_model.num_iters
        
        step_size = num_iters//num_slides
        slides = []

        for i in range(num_iters%step_size, num_iters, step_size):
            slides.append(all_pos[i])
            
        return np.stack(slides)
        

    def save_as_html(self, filename='wordmesh.html', force_directed_animation=False):
        """
        Temporary
        """  
        if force_directed_animation:
            all_positions = self._get_all_fditerations()
            self._visualizer.save_wordmesh_as_html(all_positions, filename, 
                                                   animate=True)
        else:
            self._visualizer.save_wordmesh_as_html(self.embeddings, filename)
            
    def get_mpl_figure(self):
        #embeddings = self._generate_embeddings(backend='mpl',
        #                                       store_as_attribute=False)
        #fig = _get_mpl_figure(embeddings, self.keywords, 
        #                      self.fontsizes_norm*100)
        #return fig, embeddings 
        raise NotImplementedError()
        