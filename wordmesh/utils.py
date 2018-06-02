#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 04:26:25 2018

@author: mukund
"""
import numpy as np
import os
import nltk
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
plt.ioff()

current_path = os.path.dirname(__file__)
if not (os.path.isdir(os.path.join(current_path, 'tokenizers'))):
    print('Downloading nltk resource required for POS tagging...')
    nltk.download('punkt', download_dir=current_path, quiet=True)



RELATIONSHIP_METRICS = ['cooccurence']
DISCRETE_PROPERTIES = ['POS']
#FONTSIZE_BBW = 0.17
#FONTSIZE_BBH = 0.25
PLOTLY_FONTSIZE_BBW = 0.18
PLOTLY_FONTSIZE_BBH = 0.28

#MPL_FONTSIZE_BBW = 0.005405405405405406
#MPL_FONTSIZE_BBH = 0.002

class PlotlyVisualizer():
    
    def __init__(self, words, fontsizes_norm, height, width, 
                 filename='temp-plot.html', title=None, textcolors='white',
                 hovertext=None, axis_visible=False, bg_color='black', 
                 title_fontcolor='white', title_fontsize='auto', 
                 title_font_family='Courier New, monospace', bb_padding=0.1):
        
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
        self.bounding_box_dimensions, self.real_fontsizes = self.get_bb_dimensions()
        
        
# fontsize*FONTSIZE_BBW = Width of the bounding box of each character in a plotly graph
    def _get_extent(coordinates, bbd):
        x_left = np.min((coordinates[:, 0]-bbd[:,0]/2), axis=1)
        x_right = np.max((coordinates[:, 0]+bbd[:,:,0]/2), axis=1)
        y_bottom = np.min((coordinates[:, 1]-bbd[:,1]/2), axis=1)
        y_top = np.max((coordinates[:,1]+bbd[:,1]/2), axis=1)
        return x_left, x_right, y_bottom, y_top
    
    def get_bb_dimensions(self):
    
        
        radius = min([self.height, self.width])/2
        circle_area = 3.1415*radius*radius
        ideal_area_per_word = circle_area/len(self.words)
        
        #the fontsizes are scaled such that the total area oocupied by the 
        #keywords is equal to circle_area
        num_chars = np.array([len(word) for word in self.words])
        current_area_per_word = ((self.fontsizes_norm**2)*num_chars).sum()
    
        prop_factor = ideal_area_per_word/current_area_per_word
        square_side_length = self.fontsizes_norm*(prop_factor**(1/2))*3
        
        bb_widths = (0.6+self.padding)*square_side_length*num_chars
        bb_heights = (0.972+0.088+self.padding)*square_side_length
        return np.array([bb_widths, bb_heights]).swapaxes(0, 1), square_side_length
    
    def _get_layout(self, labels=[]):
        
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
                'paper_bgcolor':self.bg_color,
                'plot_bgcolor':self.bg_color, 
                'xaxis': {'range': [-self.width/2, self.width/2], 
                          'autorange': False, 
                          'visible':self.axis_visible, 
                          'autotick':False, 
                          'dtick':10},
                'yaxis': {'range': [-self.height/2, self.height/2], 
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
                  showlegend=False, legendgroup='default_legend'):
        
        coordinates = np.array(coordinates) 
        
            
        trace = go.Scatter(
            
                    #displays hoverinfo when hovering over keyword
                    #by default, shows all text and colors it the color of the keyword
                    hoverinfo = 'skip' if (self.hovertext==None) else 'text',
                    hovertext = self.hovertext,
            
                    #Sets the legend group for this trace. 
                    #Traces part of the same legend group hide/show at the same time when toggling legend items. 
                    showlegend = showlegend,
                    legendgroup = legendgroup,
                    name = legendgroup,
            
                    #Assigns id labels to each datum. These ids for object constancy of data points during animation. 
                    ids = self.words,
            
                    x = coordinates[:,0],
                    y = coordinates[:,1],
                    
                    
                    mode = 'markers+text',
                    marker = dict(symbol='square', 
                                  opacity=marker_opacity, color = 'white', 
                                  size=self.real_fontsizes),
            
                    text = self.words,
                    textposition = 'centre',
                    textfont = dict(family = "Courier New, monospace",
                                    size = self.real_fontsizes,
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
    
    def save_wordmesh_as_html(self, coordinates, filename='temp-plot.html', animate=False):
        
        labels = ['default label']
        traces = []
        if animate:
            for i in range(coordinates.shape[0]):
                traces.append(self._get_trace(coordinates[i]))
                labels = list(map(str,range(coordinates.shape[0])))
                
        else:
            traces = [self._get_trace(coordinates)]
            
        layout = self._get_layout(labels)
            
        fig = self.generate_figure(traces, labels, layout)
        py.plot(fig, filename=filename, auto_open=False, show_link=False)
    
#def _get_mpl_figure(coordinates, words, fontsizes):
#    fig, ax = plt.subplots(figsize=(10,10))
#    
#    bbd = _get_bb_dimensions(words, fontsizes, backend='mpl')
#    bbd = np.flip(bbd, 1)
#    
#    range_max = (np.abs(coordinates)+bbd/2).max()
#
#    ax.set_xlim(left = -range_max, right = range_max)
#    ax.set_ylim(bottom = -range_max, top = range_max)
#    fontsizes = fontsizes/range_max
#    
#    for i,word in enumerate(words):
#        ax.text(coordinates[i][0], coordinates[i][1], word, 
#                dict(fontsize=fontsizes[i], family='monospace'),
#                    horizontalalignment='center', verticalalignment='center')
#    return fig

def _cooccurence_score(text, word1, word2): 
    text, word1, word2 = text.lower(), word1.lower(), word2.lower()
    l1 = _find_all(text, word1)
    l2 = _find_all(text, word2)

    square_sum =0
    for i in l1:
        for j in l2:
            square_sum = square_sum + (i-j)**2

    return (square_sum**(1/2))/(len(l1)*len(l2))

def _find_all(text, substring):
    loc = text.find(substring)
    if loc == -1:
        return []
    else:
        sub_locs = _find_all(text[loc+1:], substring)
        return [loc] + [loc+i+1 for i in sub_locs]

def cooccurence_similarity_matrix(text, wordlist):
    score_func = lambda x,y: _cooccurence_score(text, wordlist[int(x)], wordlist[int(y)])
    vscore_func = np.vectorize(score_func)
    return np.fromfunction(vscore_func, shape=[len(wordlist)]*2)

#def _save_wordmesh_as_html(coordinates, words, fontsizes, height, width, 
#                           filename='temp-plot.html',
#                           title=None, textcolors='white', 
#                           hovertext=None, axis_visible=False, bg_color='black',
#                           title_font_color='white', title_font_size='auto', 
#                           title_font_family='Courier New, monospace', 
#                           animate=False):
#    
#    labels = ['default label']
#    traces = []
#    if animate:
#        for i in range(coordinates.shape[0]):
#            traces.append(get_trace(coordinates[i], words, fontsizes, 
#                                    textcolors, hovertext=hovertext))
#            labels = list(map(str,range(coordinates.shape[0])))
#            
#    else:
#        traces = [get_trace(coordinates, words, fontsizes)]
#        
#    layout = get_layout(height, width, labels, title, axis_visible, bg_color, 
#                        title_font_color, title_font_size, title_font_family)
#        
#    fig = generate_figure(traces, labels, layout)
#    py.plot(fig, filename=filename, auto_open=False, show_link=False)