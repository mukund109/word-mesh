#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 02:20:39 2018

@author: mukund
"""
from text_processing import extract_terms_by_frequency, extract_terms_by_score, normalize_text
from utils import cooccurence_similarity_matrix as csm
from utils import regularize
import numpy as np
from sklearn.manifold import MDS
from utils import PlotlyVisualizer
from force_directed_model import ForceDirectedModel
import colorlover as cl

class Wordmesh():
    def __init__(self, text, dimensions=(500, 900),
                 keyword_extractor='textrank', num_keywords=35,
                 lemmatize=True, pos_filter=None, 
                 extract_ngrams=True, filter_numbers=True):
             
        """Wordmesh object for generating and drawing wordmeshes/wordclouds.
        
        Parameters
        ----------
        text : string
            The string of text that needs to be summarized.
            
        dimensions : tuple, optional 
            The desired dimensions (height, width) of the wordcloud in pixels.
            
        keyword_extractor : {'textrank', 'sgrank', 'divrank', 'bestcoverage', 'tf'}, optional
            The algorithm used for keyword extraction. 'tf' refers to simple
            term frequency based extraction.
            
        num_keywords : int, optional
            The number of keywords to be extracted from the text. In some cases,
            if the text length is too short, fewer keywords might
            be extracted without a warning being raised.
            
        lemmatize : bool, optional
            Whether the text needs to be lemmatized before keywords are
            extracted from it
            
        pos_filter : list of str, optional
            Supported pos tags-{'NOUN','PROPN','ADJ','VERB','ADV','SYM','PUNCT'}.
            A POS filter can be applied on the keywords ONLY when the 
            keyword_extractor has been set to 'tf'.
            
        extract_ngrams : bool, optional
            Whether 2 or 3 grams should be extracted.
        
        filter_numbers : bool, optional
            Whether extracted keywords can be numbers.
            
            
        Returns
        -------
        Wordmesh
            A word mesh object 
        
        """
        #The pos_filer has only been implemented for 'tf' based extraction
        if (keyword_extractor!='tf') and (pos_filter is not None):
            msg = '\'pos_filter\' is only available for \'tf\' based keyword extractor'
            raise ValueError(msg)
        elif pos_filter is None:  
            pos_filter = ['NOUN','ADJ','PROPN']
        
        self.text = text
        self.resolution = dimensions
        self.lemmatize = lemmatize
        self.keyword_extractor = keyword_extractor
        self.pos_filter = pos_filter
        self.extract_ngrams = extract_ngrams
        self.num_keywords = num_keywords
        self.filter_numbers = filter_numbers
        self.apply_delaunay = True
        self._extract_keywords()
        self.set_fontsize()
        self.set_fontcolor()
        self.set_clustering_criteria()

    def _extract_keywords(self):

        if self.keyword_extractor == 'tf':
            
            self.keywords, self.scores, self.pos_tags, n_kw = \
            extract_terms_by_frequency(self.text, self.num_keywords, 
                                       self.pos_filter, self.filter_numbers, 
                                       self.extract_ngrams)
        else:
            self.keywords, self.scores, self.pos_tags, n_kw = \
            extract_terms_by_score(self.text, self.keyword_extractor,
                                   self.num_keywords, self.extract_ngrams)
        #self.normalized_keywords are all lemmatized if self.lemmatize is True,
        #unlike self.keywords which contain capitalized named entities
        self.normalized_keywords = n_kw
            

    def set_fontsize(self, by='scores', custom_sizes=None, 
                     apply_regularization=True, regularization_factor=3):
        """
        This function can be used to pick a metric which decides the font size
        for each extracted keyword. The font size is directly 
        proportional to the 'scores' assigned by the keyword extractor. 
        
        Fonts can be picked by: 'scores', 'word_frequency', 'constant', None
        
        You can also choose custom font sizes by passing in a dictionary 
        of word:fontsize pairs using the argument custom_sizes
        
        Parameters
        ----------
        
        by : string or None, optional
            The metric used to assign font sizes. Can be None if custom sizes 
            are being used
        custom_sizes : list of float or numpy array or None, optional
            A list of font sizes. There should be a one-to-one correspondence
            between the numbers in the list and the extracted keywords (that 
            can be accessed through the keywords attribute). Note that this list
            is only used to calculate relative sizes, the actual sizes depend
            on the visualization tool used. Alse it is advised that you turn 
            regularizatin off when applying custom font sizes.
        apply_regularization : bool, optional
            Determines whether font sizes will be regularized to prevent extreme 
            values which might lead to a poor visualization
        regularization_factor : int, optional
            Determines the ratio max(fontsizes)/min(fontsizes). Fontsizes are
            scaled linearly so as to achieve this ratio. This helps prevent 
            extreme values.
            
        Returns
        -------
        
        None
        """
        
        if custom_sizes is not None:
            assert len(custom_sizes)==len(self.keywords)
            self.fontsizes_norm = np.array(custom_sizes)
        elif by=='scores':
            self.fontsizes_norm = self.scores/self.scores.sum() 
        elif by=='constant':
            self.fontsizes_norm = np.full(len(self.keywords), 1)
        else:
            raise ValueError()
            
            
        #applying regularization
        if apply_regularization:
            self.fontsizes_norm = regularize(self.fontsizes_norm, 
                                             regularization_factor)
        
        #normalize
        self.fontsizes_norm = self.fontsizes_norm/self.fontsizes_norm.sum()

            
    def set_fontcolor(self, by='random', colorscale='YlGnBu', 
                      custom_colors=None):
        """
        This function can be used to pick a metric which decides the font color
        for each extracted keyword. By default, the font size is assigned 
        randomly 
        
        Fonts can be picked by: 'random', 'scores', 'pos_tag' None
        
        You can also choose custom font colors by passing in a dictionary 
        of word:fontcolor pairs using the argument custom_sizes, where 
        fontcolor is an (R, G, B) tuple
        
        Parameters
        ----------
        
        by : str or None, optional
            The metric used to assign font sizes. Can be None if custom colors 
            are being used
        colorscale: str or None, optional
            One of [Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues].
            When by=='scores', this will be used to determine the colorscale.
        custom_colors : dictionary or None, optional
            A dictionary with individual keywords as keys and font colors as
            values (these should be RGB tuples). The dictionary should contain
            all extracted keywords (that can be accessed through the keywords 
            attribute). Extra words will be ignored
            
        Returns
        -------
        
        None
        """
        
        if by=='random':
            tone = np.random.choice(list(cl.flipper()['seq']['3'].keys()))
            self.fontcolors = np.random.choice(list(cl.flipper()['seq']\
                                                    ['3'][tone]), 
                                                    len(self.keywords))
                
        elif by=='scores':

            scales = {**cl.scales['8']['div'], **cl.scales['8']['seq']}
            #Even though, currently all colorscales in 'scales.keys()' can be 
            #used, only the ones listed in the doc can be used for creating a 
            #colorbar in the plotly plot
            
            assert colorscale in ['Greys','YlGnBu', 'Greens', 'YlOrRd', 
                                  'Bluered', 'RdBu', 'Reds', 'Blues']
            colors = scales[colorscale]
            colors.reverse()
            
            #The keywords are binned based on their scores
            mn, mx = self.scores.min(), self.scores.max()
            bins = np.linspace(mn,mx,8)
            indices = np.digitize(self.scores, bins)-1
            
            self.fontcolors = [colors[i] for i in indices]
            
        elif by=='pos_tag':
            c = cl.scales['5']['qual']['Set2'] + ['rgb(254,254,254)', 'rgb(254,254,254)']
            tags = ['NOUN','PROPN','ADJ','VERB','ADV','SYM','PUNCT']
            mapping = {tag:c[i] for i,tag in enumerate(tags)}
            self.fontcolors = list(map(mapping.get, self.pos_tags))
            
        else:
            raise ValueError()

            
    def set_clustering_criteria(self, by='cooccurence', 
                          custom_similarity_matrix=None, 
                          apply_regularization=True):
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
        self.normalized_text = normalize_text(self.text)
        if by=='cooccurence':
            sm = csm(self.normalized_text,
                                         self.normalized_keywords)
        elif by=='random':
            num_kw = len(self.keywords)
            sm = np.full((num_kw,num_kw), 240)
            self.apply_delaunay = False
            self.similarity_matrix= sm
            return sm
        
        elif by=='scores':
            mat = np.outer(self.scores, self.scores.T)+1
            sm = 1/(np.absolute(mat-mat.mean()))
        else:
            raise ValueError()
            
        #apply regularization
        if apply_regularization:
            shape = sm.shape
            temp = regularize(sm.flatten(), 5)
            sm = temp.reshape(shape)
        
        #normalize
        sm = sm*200/np.mean(sm)
        
        self.similarity_matrix= sm
        
        return self.similarity_matrix
    
    def generate_embeddings(self):
        self._generate_embeddings()
    
    def _generate_embeddings(self, store_as_attribute=True):
        mds = MDS(2, dissimilarity='precomputed').\
                             fit_transform(self.similarity_matrix)
                           
        self._visualizer = PlotlyVisualizer(words = self.keywords,
                                            fontsizes_norm =self.fontsizes_norm, 
                                            height = self.resolution[0],
                                            width = self.resolution[1], 
                                            textcolors=self.fontcolors)
        
        bbd = self._visualizer.bounding_box_dimensions
        
        fdm = ForceDirectedModel(mds, bbd, num_iters=100,
                                 apply_delaunay=self.apply_delaunay)
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
        try:
            self._visualizer
        except AttributeError:
            self.generate_embeddings()
            
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
        