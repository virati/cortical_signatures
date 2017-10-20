import math
import numpy as np
from scipy.sparse import csgraph


def read_eeg():
    """

    Reads spatial coordinate data from GSN-HydroCel-257 file

    Returns:
        Array of spatial coorinates + identifier
        [(x1, y1, z1, id1), ... , (x257, y257, z257, id257)]

    """

    # index: node identifier
    # (x, y, z)
    channels = []

    # read coordinate data
    f = open("GSN-HydroCel-257.rtf", "r")

    line = f.readline()[:-4] # remove odd rtf characters
    while line != "":
        # avoid rtf formatting errors,
        # make sure looking at actual line of data
        if line[0] != "E":
            line = f.readline()
            continue

        i = 0 # position in line

        identifier = "" # parse identifier
        while line[i] != " ":
            identifier += line[i]
            i += 1
        i += 1

        x = "" # parse x
        while line[i] != " ":
            x += line[i]
            i += 1
        i += 1

        y = "" # parse y
        while line[i] != " ":
            y += line[i]
            i += 1
        i += 1

        z = "" # parse z
        while i < len(line) and line[i] != " ":
            z += line[i]
            i += 1

        channels.append((float(x), float(y), float(z), identifier))
        line = f.readline()[:-4]

    return channels




def load_neighbors_within_radius(channels, radius):
    """

    Each has everything as its neighbor
    Algorithm peels non-neighbors off

    Returns arrays of neighbors for each EEG within
    some radius (inclusive)

    """
    # Index is channel
    # Value is binary "neighbor in range of radius or no"
    neighbors = [[1 for i in range(len(channels))] for i in range(len(channels))]

    # eliminate non-neighbors for both
    for i, chan in enumerate(channels):
        for j, neighbor in enumerate(channels):
            if i != j and neighbors[i][j] != 0:
                dist = math.sqrt((chan[0] - neighbor[0])**2 +\
                                (chan[1] - neighbor[1])**2 +\
                                (chan[2] - neighbor[2])**2)

                if dist < radius: continue

            # otherwise, rule out for both
            neighbors[i][j] = 0
            neighbors[j][i] = 0

    return neighbors


def get_neighbors_within_radius(home, channels, radius):
    """

    Calculates neighbors for one node from scratch using brute-force approach.

    Returns:
        - List of neighbors for specified channel at a given radius

    """
    adjacent = []
    for i, chan in enumerate(channels):
        if home != i and neighbors[home][i] != 0:

            dist = math.sqrt((chan[0] - neighbor[0])**2 +\
                            (chan[1] - neighbor[1])**2 +\
                            (chan[2] - neighbor[2])**2)

            if dist < radius: adjacent.append(chan[i][3])

    return adjacent


def get_dist_matrix(channels):
    """
     Calculate distance from each to each
     Returns:
        - Distance matrix
    """
    # Index is channel, value is distance
    dists = [[0 for i in range(len(channels))] for i in range(len(channels))]

    for i, chan1 in enumerate(channels):
        for j, chan2 in enumerate(channels):
            if i != j and dists[i][j] == 0: # don't recalc if self or done, i think the i != j is redundant
                dists[i][j] = math.sqrt((chan1[0] - chan2[0])**2 +\
                                    (chan1[1] - chan2[1])**2 +\
                                    (chan1[2] - chan2[2])**2)
                dists[j][i] = dists[i][j] # avoid recalc

    return dists


def sub_means(readings):
    """
    Returns the readings from each electrode minus the averaged of other readings 
    (Note: not based on distances)
    """
    for i in range(readings):
        for j in range(readings):
            if i != j: readings[i] -= readings[j] / (len(readings)-1)
    return readings


def sub_means_weighted_distances(dists, readings, f = lambda d: 1/(d+1)):
    for i in range(readings):
        for j in range(readings):
            if i != j:
                d = dists[i][j]
                readings[i] -= readings[j] * f(d) / (len(readings)-1) # add a weight depedent on distance
    return readings



def process(readings):
    """ Return reading from each electrode based on our distance weighting function. """
    weigh = lambda d: 1/(d+1) # specify some function to map distance onto a weight
    return map(sub_means_weighted_distances, dists, readings, weigh)




print get_dist_matrix(read_eeg())
#neighbors = get_neighbors_within_radius(10, read_eeg(), 10, load_neighbors_within_radius())


""" TODO:
1. How do I want to graphically represent this?
"""
