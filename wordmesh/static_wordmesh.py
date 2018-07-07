#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 02:20:39 2018

@author: mukund
"""
from .text_processing import extract_terms_by_frequency, extract_terms_by_score
from .text_processing import normalize_text, get_semantic_similarity_matrix
from .utils import cooccurence_similarity_matrix as csm
from .utils import regularize
import numpy as np
from sklearn.manifold import MDS, TSNE
from .utils import PlotlyVisualizer
from .force_directed_model import ForceDirectedModel
import colorlover as cl
import pandas as pd

FONTSIZE_REG_FACTOR = 3
CLUSTER_REG_FACTOR = 4
NUM_ITERS = 100
SIMILARITY_MEAN = 400
NOTEBOOK_MODE = False
        
class Wordmesh():
    """
    Wordmesh object for generating and drawing wordmeshes/wordclouds.
    
    Attributes
    ----------
    text : str
        The text used to extract the keywords.
        
    keywords : list of str
        The keywords extracted from the text.
        
    scores : numpy array
        The scores assigned by the keyword extraction algorithm.
        
    pos_tags : list of str
        The pos_tags corresponding to the keywords.
        
    embeddings : numpy array
        An array of shape (num_keywords, 2), giving the locations of the 
        keywords on the canvas.
        
    bounding_box_width_height : numpy array
        An array of shape (num_keywords, 2) gives the width and height of
        each keyword's bounding box. The coordinates of the centre of 
        the box can be accessed through the 'embeddings' attribute.
        
    similarity_matrix : numpy array
        The similarity matrix with shape (num_keywords, num_keywords), is 
        proportional to the 'dissimilarity' between the ith and jth keywords. 
        The matrix may have been regularized to prevent extreme values.
        
    fontsizes_norm : numpy array
        The normalized fontsizes, the actual fontsizes depend on the 
        visualization. These may have been regularized to avoid extreme values.
        
    fontcolors : list of str
        The fontcolors as rgb strings. This format was chosen since it is 
        supported by plotly.
    
    """
    def __init__(self, text, dimensions=(500, 900),
                 keyword_extractor='textrank', num_keywords=70, 
                 lemmatize=True, pos_filter=None, 
                 extract_ngrams=False, filter_numbers=True,
                 filter_stopwords=True):
             
        """        
        Parameters
        ----------
        text : string
            The string of text that needs to be summarized.
            
        dimensions : tuple, optional 
            The desired dimensions (height, width) of the wordcloud in pixels.
            
        keyword_extractor : {'textrank', 'sgrank', 'bestcoverage', 'tf'}, optional
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
            Filters out all keywords EXCEPT the ones with these pos tags.
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
            
            msg = '\'pos_filter\' is only available for \'tf\' ' +\
            'based keyword extractor. This is an issue with textacy ' +\
            'and will be fixed in the future'

            raise NotImplementedError(msg)
        elif pos_filter is None:  
            pos_filter = ['NOUN','ADJ','PROPN']
            
        #textacy's functions are unstable when the following condition is met
        #They just churn out terrible ngrams
        if (keyword_extractor!='tf') and extract_ngrams:
            msg = 'Currently, extracting ngrams using graph based methods ' +\
            'is not advisable. This is due to underlying issues ' +\
            'with textacy which will be fixed in the future. '+\
            'For now, you can set \'extract_ngrams\' to False.'
            raise NotImplementedError(msg)
            
        if len(text)<=10:
            raise ValueError("The text cannot have less that 10 characters.")
            
        self.text = text
        self._resolution = dimensions
        self._lemmatize = lemmatize
        self._keyword_extractor = keyword_extractor
        self._pos_filter = pos_filter
        self._extract_ngrams = extract_ngrams
        self._num_required_keywords = num_keywords
        self._filter_numbers = filter_numbers
        self._filter_stopwords = filter_stopwords
        
        #If textacy throws the following error while extracting keywords,
        #it means that the text is too short
        try:
            self._extract_keywords()
        except ValueError as e:
            if str(e).find('must contain at least 1 term')==-1:
                raise
            else:
                self.keywords = []
        
        #Text too short to extract keywords
        if len(self.keywords)<2 and self._keyword_extractor!='tf':
            msg = 'Text is too short to extract any keywords using ' + \
            '\'{}\'. Try switching to \'tf\' based extraction.'.format(self._keyword_extractor)
            raise ValueError(msg)
        elif len(self.keywords)<2:
            raise ValueError('Text is too short to extract any keywords.')
            
        #Cannot apply delaunay triangulation on less than 4 points
        if len(self.keywords)<4:
            self._apply_delaunay = False
        else:
            self._apply_delaunay = True
            
        self.set_visualization_params()
        self.set_fontsize()
        self.set_fontcolor()
        self.set_clustering_criteria()

    def _extract_keywords(self):

        if self._keyword_extractor == 'tf':
            
            self.keywords, self.scores, self.pos_tags, n_kw = \
            extract_terms_by_frequency(self.text, self._num_required_keywords, 
                                       self._pos_filter, self._filter_numbers, 
                                       self._extract_ngrams,
                                       lemmatize=self._lemmatize,
                                       filter_stopwords = self._filter_stopwords)
        else:
            self.keywords, self.scores, self.pos_tags, n_kw = \
            extract_terms_by_score(self.text, self._keyword_extractor,
                                   self._num_required_keywords, self._extract_ngrams,
                                   lemmatize=self._lemmatize,
                                   filter_stopwords = self._filter_stopwords)
        #self._normalized_keywords are all lemmatized if self._lemmatize is True,
        #unlike self.keywords which contain capitalized named entities
        self._normalized_keywords = n_kw
            

    def set_fontsize(self, by='scores', custom_sizes=None, 
                     apply_regularization=True, 
                     regularization_factor=FONTSIZE_REG_FACTOR):
        """
        This function can be used to pick a metric which decides the font size
        for each extracted keyword. The font size is directly 
        proportional to the 'scores' assigned by the keyword extractor. 
        
        Fonts can be picked by: 'scores', 'constant', None
        
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
            on the visualization tool used.
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

            
    def set_fontcolor(self, by='scores', colorscale='YlOrRd', 
                      custom_colors=None):
        """
        This function can be used to pick a metric which decides the font color
        for each extracted keyword. By default, the font color is assigned 
        based on the score of each keyword.
        
        Fonts can be picked by: 'random', 'scores', 'pos_tag', 'clustering_criteria'
        
        You can also choose custom font colors by passing in a list of 
        (R,G,B) tuples with values for each component falling in [0,255].
        
        Parameters
        ----------
        
        by : str or None, optional
            The metric used to assign font sizes. Can be None if custom colors 
            are being used
        colorscale: str or None, optional
            One of [Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues].
            When by=='scores', this will be used to determine the colorscale.
        custom_colors : list of 3-tuple, optional
            A list of RGB tuples. Each tuple corresponding to the color of
            a keyword.
            
        Returns
        -------
        None
        """
        if custom_colors is not None:
            assert len(custom_colors) == len(self.keywords)
            if isinstance(custom_colors[0], str):
                self.fontcolors = custom_colors
            else:
                self.fontcolors = []
                for rgb in custom_colors:
                    assert len(rgb)==3
                    self.fontcolors.append('rgb'+str(rgb))
            
        elif by=='random':
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
            tags = ['NOUN','PROPN','ADJ','VERB','ADV','SYM','ADP']
            mapping = {tag:c[i] for i,tag in enumerate(tags)}
            self.fontcolors = list(map(mapping.get, self.pos_tags))
            
        elif by=='clustering_criteria':
            mds = MDS(3, dissimilarity='precomputed').\
                                 fit_transform(self.similarity_matrix)
            mds = mds-mds.min()
            mds = mds*205/mds.max() + 50
            self.fontcolors = ['rgb'+str(tuple(rgb)) for rgb in mds]
            
        else:
            raise ValueError()
            
        #raise flag to indicate that the fontcolors have been modified
        self._flag_fontcolors = True

            
    def set_clustering_criteria(self, by='scores', 
                          custom_similarity_matrix=None, 
                          apply_regularization=True, 
                          clustering_algorithm = 'MDS', delaunay_factor=None):
        """
        This function can be used to define the criteria for clustering of
        different keywords in the wordcloud. By default, clustering is done
        based on the keywords' scores, with keywords having high scores in the
        centre.
        
        The following pre-defined criteria can be used: 'cooccurence',
        'meaning', 'scores', 'random'
        
        You can also define a custom_similarity_matrix. 
        
        Parameters
        ----------
        
        by : string or None, optional
            The pre-defined criteria used to cluster keywords
            
        custom_similarity_matrix : numpy array or None, optional
            A 2-dimensional array with shape (num_keywords, num_keywords)
            The entry a[i,j] is proportional to the 'dissimilarity' between
            keyword[i] and keyword[j]. Words that are similar will be grouped
            together on the canvas.
            
        apply_regularization : bool, optional
            Whether to regularize the similarity matrix to prevent extreme 
            values.
            
        clustering_algorithm : {'MDS', 'TSNE'}, optional
            The algorithm used to find the initial embeddings based on the 
            similarity matrix.
        Returns
        -------
        None
        """
        if custom_similarity_matrix is not None:
            sm = custom_similarity_matrix
        elif by=='cooccurence':
            self._normalized_text = normalize_text(self.text,
                                                  lemmatize=self._lemmatize)
            sm = csm(self._normalized_text,
                                         self._normalized_keywords)
        elif by=='random':
            num_kw = len(self.keywords)
            sm = np.random.normal(400, 90, (num_kw,num_kw))
            sm = 0.5*(sm + sm.T)
            self._apply_delaunay = False
        
        elif by=='scores':
            mat = np.outer(self.scores, self.scores.T)
            #sm = 1/np.absolute(mat-(mat**(1/16)).mean())
            sm = np.absolute(mat.max()-mat) + 1
            
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
        if clustering_algorithm not in ['MDS','TSNE']:
            raise ValueError('Only the following clustering algorithms \
                             are supported: {}'.format(['MDS','TSNE']))
        self._clustering_algorithm = clustering_algorithm
        self._delaunay_factor = delaunay_factor
        
        #raise a flag indicating that the clustering criteria has been modified
        self._flag_clustering_criteria = True

    def set_visualization_params(self, bg_color='black'):
        """
        Set other visualization parameters
        
        Parameters
        ----------
        
        bg_color: 3-tuple of int, optional
            Sets the background color, takes in a tuple of (R,G,B) \
            color components
        """
        if isinstance(bg_color, str):
            self._bg_color = bg_color
        else:
            assert(len(bg_color)==3)
            self._bg_color = 'rgb'+str(bg_color)
            
            
        self._flag_vis = True
        
    def recreate_wordmesh(self):
        """
        Can be used to change the word placement in case the current
        one isn't suitable. Since the steps involved in the creation of the
        wordmesh are random, the result will come out looking different every 
        time.
        """
        
        #raise all the clustering flag, so as to run the MDS algorithm again
        self._flag_clustering_criteria = True
        self._generate_embeddings()
    
    def _generate_embeddings(self):
        
        if self._flag_clustering_criteria:
            
            mds = MDS(2, dissimilarity='precomputed').\
                                 fit_transform(self.similarity_matrix)
            self._initial_embeds = mds
            
            if self._clustering_algorithm == 'TSNE':
                self._initial_embeds = TSNE(metric='precomputed', 
                                            perplexity=3, init=mds).\
                                            fit_transform(self.similarity_matrix)
            
        if self._flag_fontsizes or self._flag_fontcolors or self._flag_vis:
            self._visualizer = PlotlyVisualizer(words = self.keywords,
                                                fontsizes_norm =self.fontsizes_norm, 
                                                height = self._resolution[0],
                                                width = self._resolution[1], 
                                                textcolors=self.fontcolors,
                                                bg_color = self._bg_color)
            self.bounding_box_width_height = self._visualizer.bounding_box_dimensions
        
        if self._flag_fontsizes or self._flag_clustering_criteria:
            bbd = self.bounding_box_width_height
            fdm = ForceDirectedModel(self._initial_embeds, bbd, num_iters=NUM_ITERS,
                                     apply_delaunay=self._apply_delaunay,
                                     delaunay_multiplier=self._delaunay_factor)
            self._force_directed_model = fdm
            self.embeddings = fdm.equilibrium_position()
            
        #turn off all flags
        self._flag_clustering_criteria = False
        self._flag_fontsizes = False
        self._flag_fontcolors = False
      
    def _get_all_fditerations(self, num_slides=10):
        all_pos = self._force_directed_model.all_centered_positions
        num_iters = self._force_directed_model.num_iters
        
        step_size = num_iters//num_slides
        slides = []

        for i in range(num_iters%step_size, num_iters, step_size):
            slides.append(all_pos[i])
            
        return np.stack(slides)
        
    def save_as_html(self, filename='wordmesh.html', 
                     force_directed_animation=False, notebook_mode=NOTEBOOK_MODE):
        """
        Save the plot as an html file.
        
        Parameters
        ----------
        
        filename: str, (default='wordmesh.html')
            The path of the html file 
            
        force_directed_animation: bool, optional
            Setting this to True lets you visualize the force directed algorithm
            
        notebook_mode: bool, optional
            Set this to True to view the plot in a jupyter notebook. 
            The file will NOT be saved when notebook_mode is True.
        """  
        #generate embeddings if any of the wordmesh parameters have been modified
        if self._flag_clustering_criteria or self._flag_fontsizes or self._flag_fontcolors or self._flag_vis:
            self._generate_embeddings()
            
        if force_directed_animation:
            all_positions = self._get_all_fditerations()
            self._visualizer.save_wordmesh_as_html(all_positions, filename, 
                                                   animate=True,
                                                   notebook_mode=notebook_mode)
        else:
            self._visualizer.save_wordmesh_as_html(self.embeddings, filename,
                                                   notebook_mode=notebook_mode)
            
    def plot(self, force_directed_animation=False):
        """
        Can be used to plot the wordmesh inside a jupyter notebook
        
        Parameters
        ----------
        
        force_directed_animation : bool, optional
            Setting this to True lets you visualize the force directed algorithm
        """
        self.save_as_html(force_directed_animation=force_directed_animation,
                          notebook_mode=True)
        
            
class LabelledWordmesh(Wordmesh):
    """
    Create a wordmesh from labelled text. This can be used when the text 
    is composed of several sections, each having a label associated with it.
    It can also be used to compare two different sources of text. 
    
    Attributes
    ----------
    text : pandas DataFrame
        The 'text' and its corresponding 'label'
        
    keywords : list of str
        The keywords extracted from the text.
        
    scores : numpy array
        The scores assigned by the keyword extraction algorithm.
        
    pos_tags : list of str
        The pos_tags corresponding to the keywords.
        
    labels : list of int
        The labels corresponding to each keyword.
        
    embeddings : numpy array
        An array of shape (num_keywords, 2), giving the locations of the 
        keywords on the canvas.
        
    bounding_box_width_height : numpy array
        An array of shape (num_keywords, 2) gives the width and height of
        each keyword's bounding box. The coordinates of the centre of 
        the box can be accessed through the 'embeddings' attribute.
        
    similarity_matrix : numpy array
        The similarity matrix with shape (num_keywords, num_keywords), is 
        proportional to the 'dissimilarity' between the ith and jth keywords. 
        The matrix may have been regularized to prevent extreme values.
        
    fontsizes_norm : numpy array
        The normalized fontsizes, the actual fontsizes depend on the 
        visualization. These may have been regularized to avoid extreme values.
        
    fontcolors : list of str
        The fontcolors as rgb strings. This format was chosen since it is 
        supported by plotly.
    
    """
            
    def __init__(self, labelled_text, dimensions=(500, 900),
                 keyword_extractor='textrank', num_keywords=35, 
                 lemmatize=True, pos_filter=None, 
                 extract_ngrams=False, filter_numbers=True,
                 filter_stopwords=True):
        """
        Parameters
        ----------
        
        labelled_text: list of (int, str) 
            Here the 'int' is the label associated with the text
        
        dimensions : tuple, optional 
            The desired dimensions (height, width) of the wordcloud in pixels.
            
        keyword_extractor : {'textrank', 'sgrank', 'bestcoverage', 'tf'}, optional
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
            Filters out all keywords EXCEPT the ones with these pos tags.
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
        LabelledWordmesh
            A LabelledWordmesh object, which inherits from Wordmesh
        """
        if not (isinstance(labelled_text, list) or isinstance(labelled_text, pd.DataFrame)):
            raise ValueError('labelled_text can only be a list or a pandas \
                             dataframe, not a {}'.format(type(labelled_text)))
        if isinstance(labelled_text, list):
            assert len(labelled_text)!=0
            assert len(labelled_text[0])==2
            labelled_text = pd.DataFrame(labelled_text, 
                                         columns=['label','text'])
            
        assert labelled_text.shape[1]==2
        assert (labelled_text.columns==['label','text']).all()
        
        #NOTE: code is not optimised, holds unnecessary copies of the text
        #will need to use suitable data structures in the future
        
        labelled_text['text'] = labelled_text['text'] + ' \n '
        #dicitonary of label:text
        self.text_dict = labelled_text.groupby('label')['text'].sum().to_dict()
        
        if len(self.text_dict)>8:
            raise ValueError('Only up to 8 unique labels are allowed right now')
        
        super().__init__(labelled_text, dimensions=dimensions, 
                 keyword_extractor=keyword_extractor,
                 num_keywords=num_keywords, lemmatize=lemmatize,
                 pos_filter=pos_filter, extract_ngrams=extract_ngrams,
                 filter_numbers=filter_numbers, 
                 filter_stopwords=filter_stopwords)
        
    def _extract_keywords(self):
        
        self.keywords, self.pos_tags, self.labels = [],[],[]
        self._normalized_keywords = []
        self.scores = np.array([])
        
        for key in self.text_dict:
            if self._keyword_extractor == 'tf':
                
                kw, sc, pos, n_kw = \
                extract_terms_by_frequency(self.text_dict[key], 
                                           self._num_required_keywords, 
                                           self._pos_filter, 
                                           self._filter_numbers, 
                                           self._extract_ngrams,
                                           lemmatize=self._lemmatize,
                                           filter_stopwords=self._filter_stopwords)
            else:
                kw, sc, pos, n_kw = \
                extract_terms_by_score(self.text_dict[key], 
                                       self._keyword_extractor,
                                       self._num_required_keywords, 
                                       self._extract_ngrams,
                                       lemmatize=self._lemmatize,
                                       filter_stopwords = self._filter_stopwords)
                
            self.keywords = self.keywords + kw
            self.scores = np.concatenate((self.scores, sc))
            self.pos_tags = self.pos_tags + pos
            self.labels = self.labels + [key]*len(kw)
            
            #self._normalized_keywords are all lemmatized if self._lemmatize is True,
            #unlike self.keywords which contain capitalized named entities
            self._normalized_keywords = self._normalized_keywords + n_kw
            

    def set_fontcolor(self, by='label', colorscale='Set3', 
                      custom_colors=None):
        """
        This function can be used to pick a metric which decides the font color
        for each extracted keyword. By default, the font color is assigned 
        based on the score of each keyword.
        
        Fonts can be picked by: 'label', 'random', 'scores', 'pos_tag', 'clustering_criteria'
        
        You can also choose custom font colors by passing in a list of 
        (R,G,B) tuples with values for each component falling in [0,255].
        
        Parameters
        ----------
        
        by : str or None, optional
            The metric used to assign font sizes. Can be None if custom colors 
            are being used
        colorscale: str or None, optional
            One of [Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues].
            When by=='scores', this will be used to determine the colorscale.
        custom_colors : list of 3-tuple, optional
            A list of RGB tuples. Each tuple corresponding to the color of
            a keyword.
            
        Returns
        -------
        None
        """
        
        if by=='label' and (custom_colors is None):
            scales = cl.scales['8']['qual']
            #All colorscales in 'scales.keys()' can be used
            
            assert colorscale in ['Pastel2','Paired','Pastel1',
                                  'Set1','Set2','Set3','Dark2','Accent']
            colors = scales[colorscale].copy()
            colors.reverse()
            
            color_mapping={key:colors[i] for i,key in enumerate(self.text_dict)}
            fontcolors =  list(map(color_mapping.get, self.labels))
            
            Wordmesh.set_fontcolor(self, custom_colors=fontcolors)
            
        else:
            #change default colorscale to a quantitative one
            colorscale = 'YlGnBu' if (colorscale=='Set3') else colorscale
            Wordmesh.set_fontcolor(self, by=by, colorscale=colorscale,
                                   custom_colors=custom_colors)
        
    def set_clustering_criteria(self, by='scores', 
                                custom_similarity_matrix=None, 
                                apply_regularization=True,
                                clustering_algorithm='MDS'):
        """
        This function can be used to define the criteria for clustering of
        different keywords in the wordcloud. By default, clustering is done
        based on the keywords' scores, with keywords having high scores in the
        centre.
        
        The following pre-defined criteria can be used: 'cooccurence',
        'meaning', 'scores', 'random'
        
        You can also define a custom_similarity_matrix. 
        
        Parameters
        ----------
        by : string or None, optional
            The pre-defined criteria used to cluster keywords
            
        custom_similarity_matrix : numpy array or None, optional
            A 2-dimensional array with shape (num_keywords, num_keywords)
            The entry a[i,j] is proportional to the 'dissimilarity' between
            keyword[i] and keyword[j]. Words that are similar will be grouped
            together on the canvas.
            
        apply_regularization : bool, optional
            Whether to regularize the similarity matrix to prevent extreme 
            values.
        
        clustering_algorithm : {'MDS', 'TSNE'}, optional
            The algorithm used to find the initial embeddings based on the 
            similarity matrix.
        Returns
        -------
        None
        """
        
        if by=='cooccurence':
            
            normalized = self.text
            normalized['text'] = normalized['text']\
            .apply(lambda x: normalize_text(x, lemmatize=self._lemmatize))
            
            #self._normalized_text = list(normalized.itertuples(False, None))
            
            sm = csm(self._normalized_text, 
                     self._normalized_keywords, 
                     labelled=True, labels=self.labels)
            Wordmesh.set_clustering_criteria(self, custom_similarity_matrix=sm, 
                                             apply_regularization=apply_regularization,
                                             clustering_algorithm=clustering_algorithm)
        else:
            Wordmesh.set_clustering_criteria(self, by=by, 
                                             custom_similarity_matrix=custom_similarity_matrix,
                                             apply_regularization=apply_regularization,
                                             clustering_algorithm=clustering_algorithm)
