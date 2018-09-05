# Student Name: Pat Dayton
# University of Texas at Dallas
# Course: CS 6301-505
# Instructor: Dr. Arthur Redfern
# Homework #1
# September 5, 2018

import numpy as np

input_feature_type = "random"  # {random, sequential}
input_feature_map_size = [3, 3, 3]  # Ni, Lr, Lc (channels, rows cols)
input_feature_map_zero_padding = [1, 1, 1, 1]  # Pl, Pr, Pt, Pb
input_feature_map_up_sampling_factor = [1, 1]  # Ur, Uc
filter_coefficient_type = "random"  # {random, sequential}
filter_coefficient_size = [3, 3, 3, 3]  # No, Ni, Fr, Fc
filter_coefficient_up_sampling_factor = [1, 1]  # Dr, Dc
output_feature_map_channels = 1  # No
output_feature_map_down_sampling_factor = [1, 1]  # Sr, Sc


x = np.zeros((5, 5, 3))

print(x)
