
#Použila som kód z cvičení:
# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, Juraj Holas, Peter Gergel, Endre Hamerlik, Štefan Pócoš, Iveta Bečková 2017-2024

import itertools
import numpy as np
import random as rd

from util import *

eps_hard_limit = 100



class Hopfield():

    def __init__(self, dim):
        self.dim  = dim
        self.beta = None  # if beta is None: deterministic run, else stochastic run


    def train(self, patterns):
        '''
        Compute weight matrix analytically
        '''
        self.W = np.zeros((self.dim, self.dim))
        for p in patterns:
            for i in range(self.dim):
                for j in range(self.dim):
                    if(i!=j):
                        self.W[i,j] += (p[i]*p[j])/self.dim



    def energy(self, s):
        '''
        Compute energy for given state
        '''
        #− 1/2 ∑_j ∑_(i!=j) w_(i,j) s_i s_j
        #print(s)
        my_sum = 0
        for i in range(self.dim):
            for j in range(self.dim):
                if(i!=j):
                    my_sum+=self.W[i,j]*s[i]*s[j]
        
        return (-1/2)*my_sum



    def forward_one_neuron(self, s, neuron_index):
        '''
        Perform forward pass in asynchronous dynamics - compute output of selected neuron
        '''
        net = self.W@s 
        

        if self.beta is None:
            # Deterministic transition
            return np.array(list(map(lambda x: 1 if x>=0 else -1, net)))[neuron_index]
        else:
            # Stochastic transition (Output for each neuron is either -1 or 1!)
            prob = self.beta
            return np.array(list(map(lambda x: 1 if (rd.random()<(1/(1+np.e**((-1)*prob*x)))) else -1, net)))[neuron_index]


    def forward_all_neurons(self, s):
        '''
        Perform forward pass in synchronous dynamics - compute output of all neurons
        '''
        net = self.W@s

        if self.beta is None:
            # Deterministic transition
            return np.array(list(map(lambda x: 1 if x>=0 else -1, net)))
        else:
            # Stochastic transition (Output for each neuron is either -1 or 1!)
            prob = self.beta
            return np.array(list(map(lambda x: 1 if (rd.random()<(1/(1+np.e**(-prob*x[i])))) else -1, net)))



    def run_sync(self, x, orgig_patterns):
        '''
        Run model in synchronous dynamics. One input vector x will produce
        series of outputs (states) s_t.
        '''
        s = x.copy()
        e = self.energy(s)
        s_length = len(s)
        
        S = [s]
        E = [e]

        overlaps = []

        for pat in orgig_patterns:
            overlaps.append(np.max((np.sum(np.equal(pat, s)), np.sum(np.not_equal(pat, s)))) / s_length)

        
        for t in itertools.count():
            
            s = self.forward_all_neurons(s)
            e = self.energy(s)
            s_length = len(s)
            
            S.append(s)
            E.append(e)
            

            for pat in orgig_patterns:
                overlaps.append(np.max((np.sum(pat != s), np.sum(pat == s))) / s_length)


            if ((s==S[-2]).all()):
                
                if np.max(overlaps) == 1.0000:
                    return 0, S, E, overlaps, t + 1
                
                else:
                    return 1, S, E, overlaps, t + 1
                
            for ar in S[-2::-1]:
                
                if (s == ar).all():
                    
                    return 2, S, E, overlaps, t + 1




    def run_async(self, x, eps=None, beta_s=None, beta_f=None, row=1, rows=1, trace=False):
        '''
        Run model in asynchronous dynamics. One input vector x will produce
        series of outputs (states) s_t.
        '''
        s = x.copy()
        e = self.energy(s)
        E = [e]

        title = 'Running: asynchronous {}'.format('stochastic' if beta_s is not None else 'deterministic')

        for ep in range(eps):
            ## Set beta for this episode
            if beta_s is None:
                # Deterministic -> no beta
                self.beta = None
                print('Ep {:2d}/{:2d}:  deterministic'.format(ep+1, eps))
            else:
                # Stochastic -> schedule for beta (or temperature)
                self.beta = beta_s * ((beta_f/beta_s) ** (ep/(eps-1)))
                print('Ep {:2d}/{:2d}:  stochastic, beta = 1/T = {:7.4f}'.format(ep+1, eps, self.beta))

            ## Compute new state for each neuron individually
            for i in np.random.permutation(self.dim):
                s[i] = self.forward_one_neuron(s, neuron_index=i) # update state of selected neuron
                e = self.energy(s)
                E.append(e)

                # Plot
                if trace:
                    plot_state(s, energys=E, index=i, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=False)
                    redraw()

            # Terminate deterministically when stuck in a local/global minimum (loops generally don't occur)
            if self.beta is None:
                if np.all(self.forward_all_neurons(s) == s):
                    print('Reached local/global minimum after {} episode{}, terminating.'.format(ep+1, 's' if ep > 0 else ''))
                    break

        # Plot
        if not trace:
            plot_state(s, energys=E, index=None, max_eps=eps*self.dim, row=row, rows=rows, title=title, block=True)

        print('Final state energy = {:.2f}'.format(self.energy(s)))
