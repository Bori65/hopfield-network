 #!/usr/bin/python3


import numpy as np
import random

from hopfield import Hopfield
from util import *

#investigate the dynamics of the network – for ≥10000 random input patterns:
"""
    count the number of inputs leading to
        true attractors – stable states identical to input patterns (or negatives)
        spurious attractors – stable states not matching any input pattern
        limit cycles – periodic trajectories
    plot the 10 most frequent final states or cycles
        merge duplicates that differ only in the sign or in the phase of the cycle
"""


## Load data
dataset = "letters.txt"

patterns, dim = prepare_data_num(dataset)

model = Hopfield(dim)
model.train(patterns)

k = 10000

result_types = {0:"True attractors", 1:"Spurious attractors", 2:"Limit cycle"}
result_number = {0:0, 1:0, 2:0}


final_states = []
state_count = {}

for i in range(k):
    if ((not (i % 500)) and i > 0):
        print(f"Current iteration: {i} / {k}")

    rnd_pattern = np.random.choice([-1, 1], size=dim)

    atr, S, E, O, _ = model.run_sync(rnd_pattern, patterns)
    s_in = S[-1]
    if(s_in[0]==-1):
        s_in *= -1
    final_states.append(s_in)

    result_number[atr]+=1

for i in range(3):
    print(f"{result_types[i]}: {result_number[i]}")


most_common = np.unique(final_states, return_counts=True, axis=0)
occurences = zip(most_common[1], list(map(list,most_common[0])))
sorted_occ = sorted(list(occurences))[-10:]
sorted_occ = list(map(lambda x: np.array(x[1]), sorted_occ))

plot_freq(sorted_occ)

