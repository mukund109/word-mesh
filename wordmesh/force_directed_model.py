#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:30:38 2018

@author: mukund
"""

import numpy as np
from scipy.spatial import Delaunay

def _fv_attraction(point, other_points, multiplier=1, 
                   inverse_distance_proportionality=False):
    
    """
    other_points can be a single point or an ndarray with shape: (num_points, 2)
    """
    x1, y1 = point
    
    if isinstance(other_points , np.ndarray):
        x2, y2 = other_points[:, 0], other_points[:, 1]
    else: 
        x2, y2 = other_points
    
    d = ((x2-x1)**2 + (y2-y1)**2)**(1/2)
    
    x_component = (x2-x1)
    y_component = (y2-y1)
    
    if inverse_distance_proportionality:
        x_component = x_component/d
        y_component = y_component/d
    
    if isinstance(other_points , np.ndarray):  
        force_vectors = np.asarray([multiplier*x_component, multiplier*y_component]).swapaxes(0,1)
    else:
        force_vectors = (multiplier*x_component, multiplier*y_component)
    
    return force_vectors

def _fv_collision(point, box_size, other_points, other_box_sizes, multiplier=20):
    """
    box_size = (width, height)
    other_box_sizes shape = (num_points, 2)
    """
    
    b_x1, b_y1 = point[0]-box_size[0]/2, point[1]-box_size[1]/2
    b_x2, b_y2 = point[0]+box_size[0]/2, point[1]+box_size[1]/2
    
    o_x1 = other_points[:, 0] - other_box_sizes[:,0]/2 
    o_y1 = other_points[:, 1] - other_box_sizes[:,1]/2 
    o_x2 = other_points[:, 0] + other_box_sizes[:,0]/2 
    o_y2 = other_points[:, 1] + other_box_sizes[:,1]/2 
    
    inter_x1 = np.maximum(o_x1, b_x1)
    inter_x2 = np.minimum(o_x2, b_x2)
    inter_y1 = np.maximum(o_y1, b_y1)
    inter_y2 = np.minimum(o_y2, b_y2)
    
    
    force_vector = np.ones(shape=(inter_x1.shape[0], 2), dtype=np.float32)
    
    force_vector[inter_x1>inter_x2, :] = 0
    force_vector[inter_y1>inter_y2, :] = 0
    
    overlapping_area = (inter_x2-inter_x1)*(inter_y2-inter_y1)
    overlapping_area = np.stack([overlapping_area, overlapping_area], axis=1)
    
    
    force_vector = force_vector*(-1*_fv_attraction(point, 
                                                   other_points, 
                                                   multiplier, 
                                                   True))*overlapping_area
    
    return force_vector  

def _delaunay_force(point_index, coordinates, simplices, multiplier=30):
        
    #get simplices which contain said point
    mask = np.any(simplices==point_index, axis=1)
    line_segs = coordinates[simplices[mask]].reshape(-1, 2)
    line_segs = line_segs[~(line_segs== coordinates[point_index])].reshape(-1,4)

    #find points at which given point bisects the triangle side opposite to it
    
    x1 = line_segs[:, 0]
    y1 = line_segs[:, 1]
    x2 = line_segs[:, 2]
    y2 = line_segs[:, 3]
    
    xp, yp = coordinates[point_index]
    m = (y2-y1)/(x2-x1)
    
    
    X = (m*(yp-y1) + (m**2)*x1 + xp)/(m**2 + 1)
    Y = (y1 + m*(xp-x1) + (m**2)*yp)/(m**2 + 1)
    
    # find the force due to these
    force_vector = -_fv_attraction(coordinates[point_index], 
                                   np.stack([X,Y]).swapaxes(0,1),
                                           multiplier, True)
    
    return force_vector


def _update_positions(current_positions, bounding_box_dimensions, simplices, 
                      descent_rate):
    """
    Performs a single iteration of force directed displacement for every word
    """
    updated_positions = current_positions.copy()

    bbd = bounding_box_dimensions
    
    num_particles = current_positions.shape[0]
    
    for i in range(num_particles):
        
        this_particle = updated_positions[i]
        other_particles = updated_positions[~(np.arange(num_particles)==i)]
        
        this_bbd = bbd[i]
        other_bbds = bbd[~(np.arange(num_particles)==i)]
        
        #Calculates all three forces on ith particle due to all other particles
        aforce = _fv_attraction(this_particle, other_particles)
        cforce = _fv_collision(this_particle, this_bbd, other_particles, 
                               other_bbds)
        dforce = _delaunay_force(i, updated_positions, simplices)
        
        total_force = np.sum(cforce+aforce, axis=0) + np.sum(dforce, axis=0)
        
        #updated_position = current_position + alpha*force
        #Not exactly Newtonian but works
        updated_positions[i] = updated_positions[i] + descent_rate*total_force
                         
    return updated_positions
        
def equilibrium_positions(current_positions, bounding_box_dimensions):
    """
    The equilibrium positions are calculated by applying the force directed 
    algorithm on the particles surrounded by the given bounding boxes
    
    Parameters
    ----------
    
    current_positions: numpy array
        A 2-D numpy array of shape (num_particles, 2), giving the x and y 
        coordinates of all particles
    
    bounding_box_dimensions: numpy array
        A 2-D numpy array of shape (num_particles, 2), giving the height and
        width of each bounding box
    
    Returns
    -------
    
    numpy array:
        A 2-D numpy array of shape (num_particles, 2) containing the x and y
        coordinates of the equilibrium positions of the particles
    """
    positions = current_positions.copy()
    simplices = Delaunay(positions).simplices
    
    #initial descent rate is a fraction of the distance between the centre and
    #furthest point, this was decided by trial and error
    max_radial_distance = np.max(np.sum(positions**2, axis=1)**(1/2))
    initial_dr = max_radial_distance/40000    
    
    
    for i in range(50):
        positions = _update_positions(positions, bounding_box_dimensions,
                                      simplices, initial_dr*(1-i/50))
        
    return positions
        


