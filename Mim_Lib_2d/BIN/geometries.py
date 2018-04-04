# -*- coding: utf-8 -*-
"""
functions about geometries

@author: Yi Zhang （张仪）, Created on Thu Oct 26 17:13:31 2017
    Aerodynamics
    Faculty of Aerospace Engineering
    TU Delft
"""
import numpy as np

# %% DISTANCE BETWEEN TWO POINTS
def distance_between_two_point(pt1, pt2):
    """
    Distance between two points
    """
    return np.sqrt((pt1[1]-pt2[1])**2 + (pt1[0]-pt2[0])**2)

# %% FIT INTO A STRAIGHT LINE FROM TWO POINTS
def gamma_straightline(start_point, end_point):
    """
    #SUMMARY: fit two points into a straight line
    # INPUTS: point1, point2 :: 1-d array
    """
    assert np.shape(start_point) == np.shape(end_point) == (2,)
    x1, y1 = start_point
    x2, y2 = end_point
    # o in [0, 1]
    def gamma(o): return x1 + o*(x2-x1), y1 + o*(y2-y1)
    def dgamma(o): return (x2-x1) * np.ones(np.shape(o)), (y2-y1) * np.ones(np.shape(o))
    return gamma, dgamma

# %% Anti clock wise angle between two lines
def anti_Clockwise_angle_between_two_lines(origin, pt1, pt2):
    """
    #SUMMARY: from origin to pt1: the first line,
              from origin to pt2: the second line.
    """
    if pt1 == origin or pt2 == origin:
        return 0.5 * np.pi
    else:
        start_theta = angle(origin, pt1)
        end_theta = angle(origin, pt2)
        if end_theta < start_theta: end_theta += 2*np.pi
        return end_theta - start_theta

# %% ANGLE BETWEEN TWO LINES
def angle(origin, pt):
    """
    #SUMMARY: Angle between the vector from origin to pt and the x-direction vector
    """
    x1, y1 = (1, 0)
    x2, y2 = (pt[0]-origin[0], pt[1]-origin[1])
    inner_product = x1*x2 + y1*y2
    len1 = np.hypot(x1, y1)
    len2 = np.hypot(x2, y2)
    if y2 < 0:
        return 2*np.pi - np.arccos(inner_product/(len1*len2))
    else:
        return np.arccos(inner_product/(len1*len2))

# %% FIT INTO anti-clock-wise ARC WITH:center, start_point, end_point
def gamma_arc_Anti_ClockWise(center, start_point, end_point):
    """
    #SUMMARY: fit two center and radius int arc (up half)
              the return arc is ALWAYS anti-clock-wise!!
    # INPUTS:
    """
    x0, y0 = center
    x1, y1 = start_point
    x2, y2 = end_point
    r = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    assert np.abs(r - (np.sqrt((x2-x0)**2 + (y2-y0)**2))) < 10e-15, 'center is not at proper place'
    start_theta = angle(center, start_point)
    end_theta = angle(center, end_point)
    if end_theta < start_theta: end_theta += 2*np.pi
    # o in [0, 1]
    def gamma(o):
        theta = o*(end_theta-start_theta) + start_theta
        return x0 + r*np.cos(theta), y0 + r*np.sin(theta)
    def dgamma(o):
        theta = o*(end_theta-start_theta) + start_theta
        return -r*np.sin(theta) * (end_theta-start_theta), r*np.cos(theta) * (end_theta-start_theta)
    return gamma, dgamma

# %% FIT INTO clock-wise ARC WITH:center, start_point, end_point
def gamma_arc_ClockWise(center, start_point, end_point):
    """
    #SUMMARY: fit two center and radius int arc (up half)
              the return arc is ALWAYS clock-wise!!
    # INPUTS:
    """
    x0, y0 = center
    x1, y1 = start_point
    x2, y2 = end_point
    r = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    assert np.abs(r - (np.sqrt((x2-x0)**2 + (y2-y0)**2))) < 10e-15, 'center is not at proper place'
    start_theta = angle(center, start_point)
    end_theta = angle(center, end_point)
    # o in [0, 1]
    if end_theta > start_theta: end_theta -= 2*np.pi

    def gamma(o):
        theta = o*(end_theta-start_theta) + start_theta
        return x0 + r*np.cos(theta), y0 + r*np.sin(theta)
    def dgamma(o):
        theta = o*(end_theta-start_theta) + start_theta
        return -r*np.sin(theta) * (end_theta-start_theta), r*np.cos(theta) * (end_theta-start_theta)
    return gamma, dgamma
