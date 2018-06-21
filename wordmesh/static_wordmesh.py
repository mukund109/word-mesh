#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 02:20:39 2018

@author: mukund
"""
from text_processing import extract_terms_by_frequency, extract_terms_by_score
from text_processing import normalize_text, get_semantic_similarity_matrix
from utils import cooccurence_similarity_matrix as csm
from utils import regularize
import numpy as np
from sklearn.manifold import MDS
from utils import PlotlyVisualizer
from force_directed_model import ForceDirectedModel
import colorlover as cl

FONTSIZE_REG_FACTOR = 3
CLUSTER_REG_FACTOR = 4
NUM_ITERS = 100
SIMILARITY_MEAN = 400

class Wordmesh():
    def __init__(self, text, dimensions=(500, 900),
                 keyword_extractor='textrank', num_keywords=35, 
                 lemmatize=True, pos_filter=None, 
                 extract_ngrams=True, filter_numbers=True,
                 filter_stopwords=True):
             
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
            Whether bi or tri-grams should be extracted.
        
        filter_numbers : bool, optional
            Whether numbers should be filtered out
            
        filter_stopwords: bool, optional
            Whether stopwords should be filtered out
        Returns
        -------
        Wordmesh
            A word mesh object 
        
        """
        if (keyword_extractor=='divrank'):
            raise NotImplementedError('divrank is currently unstable')
        #The pos_filer has only been implemented for 'tf' based extraction
        if (keyword_extractor!='tf') and (pos_filter is not None):
            
            msg = '\'pos_filter\' is only available for \'tf\'' \
            'based keyword extractor. This is an issue with textacy' \
            'and will be fixed in the future'

            raise NotImplementedError(msg)
        elif pos_filter is None:  
            pos_filter = ['NOUN','ADJ','PROPN']
            
        #textacy's functions are unstable when the following condition is met
        if (keyword_extractor!='tf') and extract_ngrams:
            msg = 'Currently, extracting ngrams using graph based methods ' \
            'is not advisable. This is due to underlying issues ' \
            'with textacy which will be fixed in the future.'
            raise NotImplementedError(msg)
            
            
        self.text = text
        self.resolution = dimensions
        self.lemmatize = lemmatize
        self.keyword_extractor = keyword_extractor
        self.pos_filter = pos_filter
        self.extract_ngrams = extract_ngrams
        self.num_keywords = num_keywords
        self.filter_numbers = filter_numbers
        self.filter_stopwords = filter_stopwords
        self.apply_delaunay = True
        self._extract_keywords()
        self.set_visualization_params(dimensions=dimensions)
        self.set_fontsize()
        self.set_fontcolor()
        self.set_clustering_criteria()

    def _extract_keywords(self):

        if self.keyword_extractor == 'tf':
            
            self.keywords, self.scores, self.pos_tags, n_kw = \
            extract_terms_by_frequency(self.text, self.num_keywords, 
                                       self.pos_filter, self.filter_numbers, 
                                       self.extract_ngrams,
                                       lemmatize=self.lemmatize,
                                       filter_stopwords = self.filter_stopwords)
        else:
            self.keywords, self.scores, self.pos_tags, n_kw = \
            extract_terms_by_score(self.text, self.keyword_extractor,
                                   self.num_keywords, self.extract_ngrams,
                                   lemmatize=self.lemmatize,
                                   filter_stopwords = self.filter_stopwords)
        #self.normalized_keywords are all lemmatized if self.lemmatize is True,
        #unlike self.keywords which contain capitalized named entities
        self.normalized_keywords = n_kw
            

    def set_fontsize(self, by='scores', custom_sizes=None, 
                     apply_regularization=True, 
                     regularization_factor=FONTSIZE_REG_FACTOR):
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
        
        #raise flag indicating that the fontsizes have been modified
        self._flag_fontsizes = True

            
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
            colors = scales[colorscale].copy()
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
            
        #raise flag to indicate that the fontcolors have been modified
        self._flag_fontcolors = True

            
    def set_clustering_criteria(self, by='random', 
                          custom_similarity_matrix=None, 
                          apply_regularization=True):
        """
        This function can be used to define the criteria for clustering of
        different keywords in the wordcloud. By default, clustering is done
        based on the tendency of words to frequently occur together in the
        text i.e. the 'cooccurence' criteria is used for clustering
        
        The following pre-defined criteria can be used: 'cooccurence',
        'meaning', 'pos_tag'
        
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
            sm = np.ones((num_kw,num_kw))
            self.apply_delaunay = False
        
        elif by=='scores':
            mat = np.outer(self.scores, self.scores.T)
            sm = 1/np.absolute(mat-(mat**(1/16)).mean())
            
        elif by=='meaning':
            sm = get_semantic_similarity_matrix(self.keywords)
        else:
            raise ValueError()
            
        #apply regularization
        if apply_regularization:
            shape = sm.shape
            temp = regularize(sm.flatten(), CLUSTER_REG_FACTOR)
            sm = temp.reshape(shape)
            
        #standardise
        sm = sm*SIMILARITY_MEAN/np.mean(sm)
        
        self.similarity_matrix= sm
        
        #raise a flag indicating that the clustering criteria has been modified
        self._flag_clustering_criteria = True

    def set_visualization_params(self, bg_color='black', dimensions=None):
        """
        Set other visualization parameters
        """
        self.bg_color = bg_color
        if dimensions is not None:
            self.resolution = dimensions
        self._flag_vis = True
        
    def generate_embeddings(self):
        self._generate_embeddings()
    
    def _generate_embeddings(self):
        
        if self._flag_clustering_criteria:
            mds = MDS(2, dissimilarity='precomputed').\
                                 fit_transform(self.similarity_matrix)
            self._mds = mds
            
        if self._flag_fontsizes or self._flag_fontcolors or self._flag_vis:
            self._visualizer = PlotlyVisualizer(words = self.keywords,
                                                fontsizes_norm =self.fontsizes_norm, 
                                                height = self.resolution[0],
                                                width = self.resolution[1], 
                                                textcolors=self.fontcolors,
                                                bg_color = self.bg_color)
            self._bounding_box_width_height = self._visualizer.bounding_box_dimensions
        
        if self._flag_fontsizes or self._flag_clustering_criteria:
            bbd = self._bounding_box_width_height
            fdm = ForceDirectedModel(self._mds, bbd, num_iters=NUM_ITERS,
                                     apply_delaunay=self.apply_delaunay)
            self.force_directed_model = fdm
            self.embeddings = fdm.equilibrium_position()
            
        #turn off all flags
        self._flag_clustering_criteria = False
        self._flag_fontsizes = False
        self._flag_fontcolors = False

        
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
        #generate embeddings if any of the wordmesh parameters have been modified
        if self._flag_clustering_criteria or self._flag_fontsizes or self._flag_fontcolors or self._flag_vis:
            self.generate_embeddings()
            
        if force_directed_animation:
            all_positions = self._get_all_fditerations()
            self._visualizer.save_wordmesh_as_html(all_positions, filename, 
                                                   animate=True)
        else:
            self._visualizer.save_wordmesh_as_html(self.embeddings, filename)
            
        