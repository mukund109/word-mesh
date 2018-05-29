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

project_path = os.path.join(os.path.dirname(os.getcwd()), 'wordmesh')
if not (os.path.isdir(os.path.join(project_path, 'tokenizers'))):
    print('Downloading nltk resource required for POS tagging...')
    nltk.download('punkt', download_dir=project_path, quiet=True)



RELATIONSHIP_METRICS = ['cooccurence']
DISCRETE_PROPERTIES = ['POS']
#FONTSIZE_BBW = 0.17
#FONTSIZE_BBH = 0.25
FONTSIZE_BBW = 0.17
FONTSIZE_BBH = 0.28
# fontsize*FONTSIZE_BBW = Width of the bounding box of each character in a plotly graph


def _get_bb_dimensions(words, fontsizes, fontsize_to_bbw=FONTSIZE_BBW,
                       fontsize_to_bbh=FONTSIZE_BBH):

    num_words = len(words)

    num_chars = list(map(len, words))

    widths = [fontsize_to_bbw*fontsizes[i]*num_chars[i] for i in range(num_words)]
    heights = [fontsize_to_bbh*fontsizes[i] for i in range(num_words)]

    return np.array([widths, heights]).swapaxes(0, 1)

def get_layout(title, labels, show_axis):
    
    steps = []
    for label in labels:
        step = dict(method = 'animate',
                    args = [[label]],
                    label = label
                    )
        steps.append(step)
    
    layout={'font':{'color':'white'} ,'paper_bgcolor':'black', 'plot_bgcolor':'black', 
            'xaxis': {'range': [0, 400], 'autorange': False, 'visible':show_axis, 'autotick':False, 'dtick':5},
            'yaxis': {'range': [0, 200], 'autorange': False, 'visible':show_axis, 'autotick':False, 'dtick':5},
            'title': title,
            'sliders': [{'steps':steps}]
           }
    
    return layout

def get_trace(coordinates, words, sizes, textcolors='white', debug=False):
    coordinates = np.array(coordinates)
    
    trace = go.Scatter(
                x = coordinates[:, 0],
                y = coordinates[:, 1],#.max() - coordinates[1],
                mode = 'markers+text',
                marker = dict(opacity=0),
                text = words,
                textposition = 'centre',
                hoverinfo = 'none',
                textfont = dict(family = "Courier New, monospace",
                                size = sizes,
                                color = textcolors)
            )
    if debug:           
        trace = go.Scatter(
                    x = coordinates[:,0],
                    y = coordinates[:,1],#.max() - coordinates[1],
                    mode = 'markers+text',
                    marker = dict(opacity=100),
                    text = words,
                    textposition = 'centre',
                    textfont = dict(family = "Courier New, monospace",
                                    size = sizes,
                                    color = textcolors)
                )
    
    return trace

def generate_figure(traces, labels, title='WordCloud', show_axis=False):
    frames = [{'data':[traces[i]], 'name':labels[i]} for i in range(len(traces))]
    figure={'data': [traces[0]],
            'layout': get_layout(title, labels, show_axis),
            'frames': frames
             }
    
    return figure

def _save_wordmesh_as_html(coordinates, words, fontsizes, debug=False):

    traces = [get_trace(coordinates, words, fontsizes, debug)]
    fig = generate_figure(traces, ['default label'], show_axis=True)
    py.plot(fig)

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

