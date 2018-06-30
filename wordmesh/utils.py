#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 04:26:25 2018

@author: mukund
"""

import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

PLOTLY_FONTSIZE_BBW = 0.6
PLOTLY_FONTSIZE_BBH = 0.972+0.088


class PlotlyVisualizer():
    
    def __init__(self, words, fontsizes_norm, height, width, 
                 filename='temp-plot.html', title=None, textcolors='white',
                 hovertext=None, axis_visible=False, bg_color='black', 
                 title_fontcolor='white', title_fontsize='auto', 
                 title_font_family='Courier New, monospace', bb_padding=0.08,
                 boundary_padding_factor=1.1):
        
        """
        Parameters
        ----------
        """
        self.words = words
        self.fontsizes_norm = fontsizes_norm
        self.height = height
        self.width = width
        self.title = title
        self.textcolors = textcolors
        self.hovertext = hovertext
        self.axis_visible = axis_visible
        self.bg_color = bg_color
        self.title_fontcolor = title_fontcolor
        self.title_fontsize = title_fontsize
        self.title_font_family = title_font_family
        self.padding = bb_padding
        self.boundary_padding = boundary_padding_factor
        self.bounding_box_dimensions, self.real_fontsizes = self.get_bb_dimensions()
        
# fontsize*FONTSIZE_BBW = Width of the bounding box of each character in a plotly graph
    def _get_zoom(self, coordinates):
        bbd = self.bounding_box_dimensions
        
        x_left = np.min((coordinates[:, 0]-bbd[:,0]/2))
        x_right = np.max((coordinates[:, 0]+bbd[:,0]/2))
        y_bottom = np.min((coordinates[:, 1]-bbd[:,1]/2))
        y_top = np.max((coordinates[:,1]+bbd[:,1]/2))
        
        zoom = max((x_right-x_left)/self.width, (y_top-y_bottom)/self.height)
        return zoom*self.boundary_padding
       
    def get_bb_dimensions(self):
        
        num_chars = np.array([len(word) for word in self.words])
        square_side_length = self.fontsizes_norm*150*(len(self.words)**(1/2))

        bb_widths = (PLOTLY_FONTSIZE_BBW+self.padding)*square_side_length*num_chars
        bb_heights = (PLOTLY_FONTSIZE_BBH+self.padding*2)*square_side_length
        return np.array([bb_widths, bb_heights]).swapaxes(0, 1), square_side_length
    
    def _get_layout(self, labels=[], zoom=1):
        
        steps = []
        for label in labels:
            step = dict(method = 'animate',
                        args = [[label]],
                        label = label
                        )
            steps.append(step)
            
        top_padding = 0 if (self.title is None) else self.height/8
        self.title_fontsize = self.height/20 if (self.title_fontsize=='auto') else self.title_fontsize
        
        layout={'height':self.height, 
                'width':self.width,
                'titlefont':{'color':self.title_fontcolor, 
                        'size':self.title_fontsize},
                #'paper_bgcolor':self.bg_color,
                'paper_bgcolor':'white',
                'plot_bgcolor':self.bg_color, 
                'xaxis': {'range': [-self.width*zoom/2, self.width*zoom/2], 
                          'autorange': False, 
                          'visible':self.axis_visible, 
                          'autotick':False, 
                          'dtick':10},
                'yaxis': {'range': [-self.height*zoom/2, self.height*zoom/2], 
                          'autorange': False, 
                          'visible':self.axis_visible, 
                          'autotick':False, 
                          'dtick':10},
                'margin':go.Margin(
                                l=0,
                                r=0,
                                b=0,
                                t=top_padding,
                                pad=0
                            ),
                'hovermode':'closest',
                'title': self.title,
                'sliders': [{'steps':steps}]
               }
        
        return layout
    
    def _get_trace(self, coordinates, 
                  textfonts="Courier New, monospace", marker_opacity=0, 
                  showlegend=False, legendgroup='default_legend', zoom=1):
        
        coordinates = np.array(coordinates) 
        
            
        trace = go.Scatter(
            
                    #displays hoverinfo when hovering over keyword
                    #by default, shows all text and colors it the color of the keyword
                    hoverinfo = 'skip' if (self.hovertext==None) else 'text',
                    hovertext = self.hovertext,
            
                    #Sets the legend group for this trace. 
                    #Traces part of the same legend group hide/show at the 
                    #same time when toggling legend items. 
                    showlegend = showlegend,
                    legendgroup = legendgroup,
                    name = legendgroup,
            
                    #'ids' assigns id labels to each datum. These ids can be used
                    #for object constancy of data points during animation. 
                    #However, the following line of code has the effect of 
                    #not displaying duplicate keywords which is allowed
                    #in a LabelledWordmesh object.
                    #ids = self.words,
            
                    x = coordinates[:,0],
                    y = coordinates[:,1],
                    
                    
                    mode = 'markers+text',
                    marker = dict(symbol='square', 
                                  opacity=marker_opacity, color = 'white', 
                                  size=self.real_fontsizes),
            
                    text = self.words,
                    textposition = 'centre',
                    textfont = dict(family = "Courier New, monospace",
                                    size = self.real_fontsizes*(1/zoom),
                                    color = self.textcolors)
                )
        
        return trace
    
    def generate_figure(self, traces, labels, layout):
        frames = [{'data':[traces[i]], 'name':labels[i]} for i in range(len(traces))]
        figure={'data': [traces[0]],
                'layout': layout,
                'frames': frames
                 }
        
        return figure
    
    def save_wordmesh_as_html(self, coordinates, filename='temp-plot.html', 
                              animate=False, autozoom=True, notebook_mode=False):

        zoom = 1
        labels = ['default label']
        traces = []
        if animate:
            for i in range(coordinates.shape[0]):
                
                traces.append(self._get_trace(coordinates[i]))
                labels = list(map(str,range(coordinates.shape[0])))
                
        else:

            if autozoom:
                zoom = self._get_zoom(coordinates)
            traces = [self._get_trace(coordinates, zoom=zoom)]
            
        layout = self._get_layout(labels, zoom=zoom)
            
        fig = self.generate_figure(traces, labels, layout)
        
        if notebook_mode:
            py.init_notebook_mode(connected=True)
            py.iplot(fig, filename=filename, show_link=False)
        else:
            py.plot(fig, filename=filename, auto_open=False, show_link=False)
    

def _cooccurence_score(text, word1, word2): 
    #text, word1, word2 = text.lower(), word1.lower(), word2.lower()
    l1 = _find_all(text, word1)
    l2 = _find_all(text, word2)

    distance =0
    for i in l1:
        for j in l2:
            distance = distance + abs(i-j)

    return distance/(len(l1)*len(l2)+1)

def _cooccurence_score2(text, word1, word2):
    l1 = _find_all(text, word1)
    l2 = _find_all(text, word2)
    avg = _smallest_cooc_distances(l1, l2) + \
                                    _smallest_cooc_distances(l2, l1)
    return avg

def _smallest_cooc_distances(list1, list2):
    #The method above is equivalent to the following:
    
    smallest_distance = 10000000
    sum_=0
    for i in list1:
        for j in list2:
            smallest_distance = min(smallest_distance, abs(i-j))
        sum_ += smallest_distance
    
    
    return sum_/len(list2)

def _find_all(text, substring, offset=0):
    loc = text.find(substring)
    if loc == -1:
        return []
    else:
        sub_locs = _find_all(text[loc+1:], substring)
        return [offset+loc] + [offset+loc+i+1 for i in sub_locs]
    
def _find_all_labelled(labelled_text, substring, substring_label):

    labelled_text['offset'] = labelled_text['text'].apply(len)
    labelled_text['offset'] = labelled_text['offset'].shift(1).fillna(0).cumsum()
       
    locations = labelled_text['text'].str.find(substring)
    return labelled_text[~(locations==-1) & (labelled_text['label']==substring_label)]['offset']

    #The code above is equivalent to the following
    """
    start = 0
    locations = []
    for label, text in labelled_text:
        if label==substring_label:
            loc = [start+i for i in _find_all(text, substring)]
            locations += loc
        start += len(text)
    """
    return locations

def _cooccurence_score_labelled(labelled_text, word1, word2, label1, label2):
    l1 = _find_all_labelled(labelled_text, word1, label1)
    l2 = _find_all_labelled(labelled_text, word2, label2)
      
    avg = _smallest_cooc_distances(l1, l2)+_smallest_cooc_distances(l2, l1)
    return avg
    
def cooccurence_similarity_matrix(text, wordlist, labelled=False, labels=None):
    """ 
    Finds the cooccurence score of every pair of words. Currently it 
    uses a heuristic, and is slow, so might change to a more robust
    method later on.
    """
    if not labelled:
        score_func = lambda x,y: _cooccurence_score2(text, wordlist[int(x)], wordlist[int(y)])
        vscore_func = np.vectorize(score_func)
        return np.fromfunction(vscore_func, shape=[len(wordlist)]*2)
    else:
        score_func = lambda x,y: _cooccurence_score_labelled(text, 
                                                             wordlist[int(x)],
                                                             wordlist[int(y)],
                                                             labels[int(x)],
                                                             labels[int(y)])
        vscore_func = np.vectorize(score_func)
        return np.fromfunction(vscore_func, shape=[len(wordlist)]*2)

def regularize(arr, factor):
    arr = np.array(arr)
    assert arr.ndim == 1
    
    #applying regularization
    mx = arr.max()
    mn = arr.min()
    
    if (mx==mn):
        return arr
    
    a = mx*(factor-1)/((mx-mn)*factor)
    b = mx*(mx-mn*factor)/((mx-mn)*factor)
    
    return a*arr + b