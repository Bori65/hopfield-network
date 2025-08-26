
#Použila som kód z cvičení:
# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import matplotlib
matplotlib.use('TkAgg') # fixme if plotting doesn`t work (try 'Qt5Agg' or 'Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import atexit
import os
import time
import functools
import collections



## Utilities

def prepare_data_num(filename):
    patterns = []
    with open(filename) as f:
        n = int(f.readline())
        
        for i in range(0,n):
            
            row, col = list(map(int, f.readline().split()))
            dim = row*col
            
            letter = np.zeros((row,col))
            for r in range(0, row):
                letter[r] = np.array(list(map(int, f.readline().split())))
                
            patterns.append(letter.flatten())
            
    util_setup(col, row)
    
    print(f"Number of patterns: {n} of size: {row} x {col}")
    
    return patterns, dim

def plot_freq(S):
    plt.figure(2, figsize=(10, 5))

    for i in range(len(S)):
        s=S[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(s.reshape((height, width)), interpolation="nearest", cmap="gray")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

def prepare_data(filename):
    patterns = []
    with open(filename) as f:
        count, width, height = [int(x) for x in f.readline().split()] # header
        dim = width*height

        for _ in range(count):
            f.readline() # skip empty line
            x = np.empty((height, width))
            for r in range(height):
                x[r,:] = np.array(list(f.readline().strip())) == '#'
            patterns.append(2*x.flatten()-1) # flatten to 1D vector, rescale {0,1} -> {-1,+1}

    util_setup(width, height)
    return patterns, dim

def vector(array, row_vector=False):
    '''
    Construts a column vector (i.e. matrix of shape (n,1)) from given array/numpy.ndarray, or row
    vector (shape (1,n)) if row_vector = True.
    '''
    v = np.array(array)
    if np.squeeze(v).ndim > 1:
        raise ValueError('Cannot construct vector from array of shape {}!'.format(v.shape))
    return v.reshape((1, -1) if row_vector else (-1, 1))


def add_bias(X):
    '''
    Add bias term to vector, or to every (column) vector in a matrix.
    '''
    if X.ndim == 1:
        return np.concatenate((X, [1]))
    else:
        pad = np.ones((1, X.shape[1]))
        return np.concatenate((X, pad), axis=0)


def timeit(func):
    '''
    Profiling function to measure time it takes to finish function.
    Args:
        func(*function): Function to meassure
    Returns:
        (*function) New wrapped function with meassurment
    '''
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('Function [{}] finished in {:.3f} s'.format(func.__name__, elapsed_time))
        return out
    return newfunc



## Interactive drawing

def clear():
    plt.clf()


def interactive_on():
    plt.ion()
    plt.show(block=False)
    time.sleep(0.1)


def interactive_off():
    plt.ioff()
    plt.close()


def redraw():
    # plt.gcf().canvas.draw()   # fixme: uncomment if interactive drawing does not work
    plt.waitforbuttonpress(timeout=0.001)
    time.sleep(0.001)


def keypress(e):
    if e.key in {'q', 'escape'}:
        os._exit(0) # unclean exit, but exit() or sys.exit() won't work
    if e.key in {' ', 'enter'}:
        plt.close() # skip blocking figures


def use_keypress(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', keypress)


## Non-blocking figures still block at end
def finish():
    plt.show(block=True) # block until all figures are closed


atexit.register(finish)

def plot_overlaps(overlaps, t, text = "", k=None, letter_num = 4):
    fig = plt.figure(figsize=(5, 5))
    plt.title("Pattern overlap")

    st = fig.suptitle(text, fontsize=12)
    st.set_y(0.9)
    fig.subplots_adjust(top=0.7)
    
    letters= ['X', 'H', 'O', 'Z']
    #colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.1), (0.5, 0.5, 0.5)]
    colors = ["gold", "navy", "darkgreen", "crimson"]
    
    iterations = np.arange(letter_num)
    
    for i, o in enumerate(overlaps):
        pos = i%letter_num

        if i > (letter_num - 1):
            plt.plot([(i - letter_num) // letter_num, i // letter_num], np.multiply([overlaps[i - letter_num], o], 100), c=colors[pos], label=letters[pos])
        
    plt.ylabel("% of overlap")
    plt.xlabel("# of iteration")

    plt.xticks(np.arange(len(overlaps)/4))
    
    #print(overlaps)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, letters[:letter_num])

    
    plt.show()

## Plotting
palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']

width  = None
height = None

def util_setup(w, h):
    global width, height
    width  = w
    height = h

def plot_energy(E, t, text = "", k=None, letter_num = 4):
    
    plt.plot(np.arange(len(E)), E)

    plt.title("Energy")
    
    plt.ylabel("Energy")
    plt.xlabel('# of iteration')
    
    plt.xticks(np.arange(len(E)))

    plt.tight_layout()

    plt.show()

def plot_state(s, energys=None, index=None, max_eps=None, rows=1, row=1, size=2, aspect=2, title=None, block=True):
    if plot_state.fig is None:
        plot_state.fig = plt.figure(figsize=(size,size*rows) if energys is None else ((1+aspect)*size,size*rows))
        use_keypress(plot_state.fig)

        gs = gridspec.GridSpec(rows, 2, width_ratios=[1, aspect])
        plot_state.grid = {(r,c): plt.subplot(gs[r,c]) for r in range(rows) for c in range(2 if energys else 1)}

        plt.subplots_adjust()
        plt.tight_layout()

    plot_state.fig.show() # foreground, swith plt.(g)cf

    ax = plot_state.grid[row-1,0]
    ax.clear()
    ax.set_title('State')
    ax.imshow(s.reshape((height, width)), cmap='gray', interpolation='nearest', vmin=-1, vmax=+1)
    ax.set_xticks([])
    ax.set_yticks([])

    if index:
        ax.scatter(index%width, index//width, s=150)

    if energys is not None:
        ax = plot_state.grid[row-1,1]
        ax.clear()
        ax.set_title('Energy = {:.2f}'.format(energys[-1] if len(energys) > 0 else 0))
        ax.plot(energys)

        if max_eps:
            ax.set_xlim(0, np.maximum(max_eps-1, len(energys)))

        ylim = ax.get_ylim()
        ax.vlines(np.arange(0, len(energys), width*height)[1:], ymin=ylim[0], ymax=ylim[1], color=[0.8]*3, lw=1)
        ax.set_ylim(ylim)

    plt.gcf().canvas.set_window_title(title or 'State')
    plt.show(block=block)

plot_state.fig = None


def plot_states(S, title=None, block=True):
    plt.figure(2, figsize=(10,4))
    use_keypress()
    plt.clf()

    for i, s in enumerate(S):
        plt.subplot(1, len(S), i+1)
        plt.imshow(s.reshape((height, width)), cmap='gray', interpolation='nearest', vmin=-1, vmax=+1)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.gcf().canvas.set_window_title(title or 'States')
    plt.show(block=block)
    
