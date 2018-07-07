#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:30:38 2018

@author: mukund
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
import random

ATTRACTION_MULTIPLIER = 1
COLLISION_MULTIPLIER = 5
COLLISION_CAP = 1000
DELAUNAY_MULTIPLIER = 3
DR_CONSTANT = 150

def _fv_attraction(point, other_points, multiplier=ATTRACTION_MULTIPLIER, 
                   inverse_distance_proportionality=False, normalised=False):
    
    """
    other_points can be a single point or an ndarray with shape: (num_points, 2)
    """
    x1, y1 = point
    x2, y2 = other_points[:, 0], other_points[:, 1]
    d = ((x2-x1)**2 + (y2-y1)**2)**(1/2)
    
    x_component = (x2-x1)
    y_component = (y2-y1)
    

    if inverse_distance_proportionality:
        x_component = x_component/(d+1)
        y_component = y_component/(d+1)
    
    force_vectors = np.asarray([multiplier*x_component, multiplier*y_component]).swapaxes(0,1)
    
    #normalised to account for variable number of keywords
    if normalised:
        force_vectors = force_vectors/(other_points.shape[0]+1)
    
    return force_vectors

def _fv_collision(point, box_size, other_points, other_box_sizes, 
                  multiplier=COLLISION_MULTIPLIER):
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
    
    #cap on how large the collision force vector is allowed to be
    force_vector[force_vector>COLLISION_CAP]=COLLISION_CAP
    force_vector[force_vector<-COLLISION_CAP]=-COLLISION_CAP
    return force_vector  

def _delaunay_force(point_index, current_positions, simplices, 
                    initial_positions, multiplier=DELAUNAY_MULTIPLIER):
    
    #get simplices which contain said point
    
    
    mask = np.any(simplices==point_index, axis=1)
    line_segs = current_positions[simplices[mask]].reshape(-1, 2)
    line_segs = line_segs[~(line_segs== current_positions[point_index])].reshape(-1,4)

    initial_line_segs = initial_positions[simplices[mask]].reshape(-1, 2)
    initial_line_segs = initial_line_segs[~(initial_line_segs== initial_positions[point_index])].reshape(-1,4)
    
    #find points at which given point bisects the triangle side opposite to it
    
    x1 = line_segs[:, 0]
    y1 = line_segs[:, 1]
    x2 = line_segs[:, 2]
    y2 = line_segs[:, 3]
    
    xp, yp = current_positions[point_index]
    m = (y2-y1)/(x2-x1)
    mp = (yp-y1)/(xp-x1)
    
    X = (m*(yp-y1) + (m**2)*x1 + xp)/(m**2 + 1)
    Y = (y1 + m*(xp-x1) + (m**2)*yp)/(m**2 + 1)
    
    #initial slopes
    x1_i = initial_line_segs[:, 0]
    y1_i = initial_line_segs[:, 1]
    x2_i = initial_line_segs[:, 2]
    y2_i = initial_line_segs[:, 3] 
    
    m_i = (y2_i-y1_i)/(x2_i-x1_i)
    
    xp_i, yp_i = initial_positions[point_index]
    mp_i = (yp_i-y1_i)/(xp_i-x1_i)
    
    #Calculate sign of force
    sign = np.sign(m-mp)*np.sign(m_i-mp_i)
    sign = np.repeat(sign[:, np.newaxis], 2, axis=1)
    # find the force due to these
    #force vector = sign * force of repulsion
    force_vector = sign*(-1)*_fv_attraction(current_positions[point_index], 
                                   np.stack([X,Y]).swapaxes(0,1),
                                           multiplier, True)
    
    return force_vector


def _update_positions(current_positions, bounding_box_dimensions, simplices, 
                      initial_positions, descent_rate, momentum=None, apply_delaunay=True):
    """
    Performs a single iteration of force directed displacement for every word
    """
    updated_positions = current_positions.copy()
    bbd = bounding_box_dimensions
    num_particles = current_positions.shape[0]
    
    force_memory = np.ndarray(shape=(num_particles, 2))
    
    for i in random.sample(list(range(num_particles)), num_particles):
        
        this_particle = updated_positions[i]
        other_particles = updated_positions[~(np.arange(num_particles)==i)]
        
        this_bbd = bbd[i]
        other_bbds = bbd[~(np.arange(num_particles)==i)]
        
        #Calculates all three forces on ith particle due to all other particles
        aforce = _fv_attraction(this_particle, other_particles, normalised=True)
        cforce = _fv_collision(this_particle, this_bbd, other_particles, 
                               other_bbds)
        
        if apply_delaunay:
            dforce = _delaunay_force(i, updated_positions, simplices,
                                     initial_positions)
            total_force = np.sum(cforce+aforce, axis=0) + np.sum(dforce, axis=0)
            
        else:
            total_force = np.sum(cforce+aforce, axis=0) 
        
        if momentum is not None:
            total_force = total_force + momentum[i]
            
        #updated_position = current_position + alpha*force
        #Not exactly Newtonian but works
        updated_positions[i] = updated_positions[i] + descent_rate*total_force
        force_memory[i] = total_force
                         
    return updated_positions, force_memory
        
class ForceDirectedModel():
    
    def __init__(self, positions, bounding_box_dimensions,
                 num_iters=1000, apply_delaunay=True, delaunay_multiplier=None):
        
        """
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
            a ForceDirectedModel object
        """
        if delaunay_multiplier is not None:
            DELAUNAY_MULTIPLIER = delaunay_multiplier
        self.num_particles = positions.shape[0]
        self.initial_positions = positions
        self.bounding_box_dimensions = bounding_box_dimensions
        self.num_iters = num_iters
        self.apply_delaunay = apply_delaunay
        
        if apply_delaunay:
            self.simplices = Delaunay(positions).simplices
        else:
            self.simplices = None
            
        self.all_positions = self._run_algorithm()
        self.all_centered_positions = self._centered_positions()
        
    def equilibrium_position(self, centered=True):
        """
        The equilibrium positions are calculated by applying the force directed 
        algorithm on the particles surrounded by the given bounding boxes
        
        centered: bool, optional
            If you want the bounding boxes to be centered around the origin,
            you can set this to true.
        
        Returns
        -------
        
        numpy array:
            A 2-D numpy array of shape (num_particles, 2) containing the
            x and y coordinates of the equilibrium positions of the particles
        """
        if centered:
            return self.all_centered_positions[-1]
        else:
            return self.all_positions[-1]
    
    def _run_algorithm(self):
        

        position_i = self.initial_positions.copy()
        simplices = self.simplices
        bbd = self.bounding_box_dimensions
        
        
        all_positions = np.ndarray(shape=(self.num_iters, self.num_particles, 2))
        all_positions[0] = position_i
        
        #make it a function of max radial distance or something
        avg_dist = pdist(position_i).sum(0).sum()/(self.num_particles**2)
        initial_dr = avg_dist/(DR_CONSTANT*self.num_iters)
        
        force_memory = np.zeros((self.num_particles, 2))
        
        for i in range(1,self.num_iters):
            position_i, force_memory = _update_positions(position_i, 
                                                         bbd, simplices, 
                                                         self.initial_positions,
                                                         initial_dr*(1-i*i/(self.num_iters*self.num_iters)),
                                                         apply_delaunay=self.apply_delaunay)#))
            all_positions[i] = position_i
            
        return all_positions

    def _centered_positions(self):
        
        bbd = self.bounding_box_dimensions.copy()
        bbd = np.repeat(bbd[np.newaxis, :, :], self.num_iters, axis=0)
        all_pos = self.all_positions

        x_left = np.min((all_pos[:, :, 0]-bbd[:,:,0]/2), axis=1)
        x_right = np.max((all_pos[:, :, 0]+bbd[:,:,0]/2), axis=1)
        y_bottom = np.min((all_pos[:, :, 1]-bbd[:,:,1]/2), axis=1)
        y_top = np.max((all_pos[:, :, 1]+bbd[:,:,1]/2), axis=1)
        
        centered_positions = all_pos.copy()
        
        #broadcasting; com=center of mass
        com_x = np.repeat(((x_right+x_left)/2)[:, np.newaxis], self.num_particles, axis=1)
        com_y = np.repeat(((y_top+y_bottom)/2)[:, np.newaxis], self.num_particles, axis=1)
        
        centered_positions[:, :, 0] = all_pos[:, :, 0] - com_x
        centered_positions[:, :, 1] = all_pos[:, :, 1] - com_y  
        
        return centered_positions
    
        

