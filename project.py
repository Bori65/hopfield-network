import os

#test the noise-correcting recall ability of the network – for every input pattern:
"""
corrupt the input with different amounts of discrete noise k∈{0,7,14,21} select k random pixel positions and flip them (+1↔−1) let the network relax until reaching a fixed point or a limit cycle for each step, plot:
    the amount of overlap of the current configuration with each stored pattern
    the energy of the current configuration
"""
print("Lunching task 1")
os.system('python task1.py')

#investigate the dynamics of the network – for ≥10000 random input patterns:
"""
    count the number of inputs leading to
        true attractors – stable states identical to input patterns (or negatives)
        spurious attractors – stable states not matching any input pattern
        limit cycles – periodic trajectories
    plot the 10 most frequent final states or cycles
        merge duplicates that differ only in the sign or in the phase of the cycle
"""
print("Lunching task 2")
os.system('python task2.py')

