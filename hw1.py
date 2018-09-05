# Student Name: Pat Dayton
# University of Texas at Dallas
# Course: CS 6301-505
# Instructor: Dr. Arthur Redfern
# Homework #1
# September 5, 2018

import numpy as np
from random import seed
from random import randint


input_feature_type = "sequential"  # {random, sequential}
input_feature_map_size = [3, 3, 3]  # Ni, Lr, Lc (channels, rows cols)
input_feature_map_zero_padding = [1, 1, 1, 1]  # Pl, Pr, Pt, Pb
input_feature_map_up_sampling_factor = [1, 1]  # Ur, Uc
filter_coefficient_type = "random"  # {random, sequential}
filter_coefficient_size = [3, 3, 3, 3]  # No, Ni, Fr, Fc
filter_coefficient_up_sampling_factor = [1, 1]  # Dr, Dc
output_feature_map_channels = 1  # No
output_feature_map_down_sampling_factor = [1, 1]  # Sr, Sc


def generate_data(feature_type, feature_map_size):

    # Generate correctly size tensor of zeros
    tensor = np.zeros((feature_map_size[0],
                       feature_map_size[1],
                       feature_map_size[2]))

    # seed random number generator
    seed(1)

    # set initial value for sequential tensor
    j = 0

    #  if feature type is random, return random array with random integers
    if feature_type == "random":
        with np.nditer(tensor, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = randint(0, 100)  # update this for randint range
        return tensor

    #  if feature type is random, return random array with random integers
    elif feature_type == "sequential":
        with np.nditer(tensor, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = j
                j += 1  # update this for randint range
        return tensor

    # otherwise there is a mistake in the function input
    else:
        print('feature_type must be a string with value \'random\' or \'sequential\'')
        return [[[]]]


###############################################################################################
input_tensor = generate_data(input_feature_type, input_feature_map_size)
print()
print(input_tensor)
