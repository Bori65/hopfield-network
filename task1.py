 #!/usr/bin/python3

import numpy as np
import random

from hopfield import Hopfield
from util import *

#t  est the noise-correcting recall ability of the network – for every input pattern:
"""
corrupt the input with different amounts of discrete noise k∈{0,7,14,21} select k random pixel positions and flip them (+1↔−1) let the network relax until reaching a fixed point or a limit cycle for each step, plot:
    the amount of overlap of the current configuration with each stored pattern
    the energy of the current configuration
"""

## Load data
dataset = "letters.txt"

patterns, dim = prepare_data_num(dataset)
input_pat = np.copy(patterns)

noise = [0,7,14,21]

plot_states(patterns, "Training patterns")


model = Hopfield(dim)
model.train(patterns)


letters = {0:'X', 1:'H', 2:'O', 3:'Z'}

for i in range(len(patterns)):

    letter_pattern = patterns[i]
    
    for k in noise:
        
        error_positions = np.random.randint(0, len(letter_pattern), size=k)
        for e_p in error_positions:
            letter_pattern[e_p] *= -1

        plot_states([letter_pattern], f"Corrupted input (original letter: {letters[i]}, num. of corrupted pisels: {str(k)})")

        _, S, E, overlaps, t = model.run_sync(letter_pattern, input_pat)
        plot_states(S, f"Synchronous run (original letter: {letters[i]}, num. of corrupted pisels: {str(k)})")
        
        plot_overlaps(overlaps, t, text = f"Original letter: {letters[i]}, k = {str(k)}", letter_num = 4)
        plot_energy(E, t, text = f"Original letter: {letters[i]}, k = {str(k)}", letter_num = 4)

