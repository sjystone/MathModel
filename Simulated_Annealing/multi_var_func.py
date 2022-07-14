"""
---* created at 2022.7.11 by Stone *---

use Simulated Annealing algorithm to solve multiple variable function optimization problem

"""

import numpy as np
import random
import matplotlib.pyplot as plt
import copy


class SimulatedAnnealing():
    """
    apply for simulated annealing algorithm of multiple variable function optimization problem
    """

    def __init__(self, var_num, var_min, var_max, temp_init, temp_final, markov, alpha, lr):

        self.var_num = var_num                                  # number of variable
        self.var_min = [var_min for _ in range(var_num)]        # lower limit of search space
        self.var_max = [var_max for _ in range(var_num)]        # higher limit of search space
        self.temp_init = temp_init                              # initial temperature
        self.temp_final = temp_final                            # final temperature
        self.markov  = markov                                   # length of markov
        self.alpha = alpha                                      # cooling parameter
        self.lr = lr                                            # length of search step

        self.fx_best = None                                     # best solution
        self.X_best = None                                      # best solution

        self.temp_ = []                                         # process of temperature
        self.fx_ = []                                           # process of function value
        self.fx_best_ = []                                      # process of the best function value
        self.P_inferior_acc_ = []                               # process of inferior acceptation probability


    def func(self, X):
        """
        define the target function
        """
        fx = 0.0
        for i in range(len(X)):
            fx += X[i] * np.sin(np.sqrt(np.abs(X[i])))
        fx = 418.9829 * len(X) - fx
        return fx

    def lr_decay(self):
        self.lr = 0.99 * self.lr

    def solve(self, rand_seed=None):

        if rand_seed != None:
            random.seed(rand_seed)

        # --- random initialize the solution --- #

        X_init = np.zeros((self.var_num))
        for i in range(self.var_num):
            X_init[i] = random.uniform(self.var_min[i], self.var_max[i])

        # --- initialize --- #

        fx_init = self.func(X_init)
        fx_curr = fx_init
        self.fx_best = fx_init
        print('initial function value is:', fx_init)

        X_curr = copy.deepcopy(X_init)
        temp_curr = self.temp_init

        # --- solve --- #

        itr = 0
        while temp_curr >= self.temp_final:

            high_quality_num = 0
            inferior_acc_num = 0
            inferior_ref_num = 0

            for _ in range(self.markov):

                # random choose one variable and apply disturbance to it
                X_new = copy.deepcopy(X_curr)
                ith = random.randint(0, self.var_num - 1)
                X_new[ith] = X_new[ith] + self.lr * (self.var_max[ith] - self.var_min[ith]) * random.normalvariate(0, 1)
                # limit the variable within a certain range
                X_new[ith] = max(min(X_new[ith], self.var_max[ith]), self.var_min[ith])

                # calc value of fx and energy difference
                fx_new = self.func(X_new)
                delta_E = fx_new - fx_curr

                # accept the new solution according to Metropolis
                if fx_new < fx_curr:
                    high_quality_num += 1
                    accepted = True         # accept the high quality solution
                else:
                    # calc the transition probability of tolerant solution
                    P_accepted = np.exp(-delta_E / temp_curr)

                    if P_accepted > random.random():
                        accepted = True     # accept the low quality solution
                        inferior_acc_num += 1
                    else:
                        accepted = False    # refuse the low quality solution
                        inferior_ref_num += 1

                # save the new solution
                if accepted:
                    X_curr = copy.deepcopy(X_new)
                    fx_curr = fx_new

                    # save the best solution
                    if fx_new < self.fx_best:
                        self.fx_best = fx_new
                        self.X_best = copy.deepcopy(X_new)
                        # apply lr decay
                        self.lr_decay()

            # probability of accepting inferior solution
            P_inferior_acc = inferior_acc_num / (inferior_acc_num + inferior_ref_num)
            self.P_inferior_acc_.append(P_inferior_acc)
            self.fx_.append(fx_curr)
            self.fx_best_.append(self.fx_best)
            self.temp_.append(temp_curr)

            if itr % 10 == 0:
                print('iter:{} | temperature:{} | Best value of func:{}'
                      .format(itr, round(temp_curr, 3), round(self.fx_best, 3)))
            itr += 1

            # --- cooling ---
            temp_curr  = temp_curr * self.alpha




if __name__ == '__main__':

    # SA = SimulatedAnnealing(var_num=10, var_min=-1000, var_max=1000,
    #                         temp_init=100, temp_final=1, markov=100, alpha=0.98, lr=0.5)
    SA = SimulatedAnnealing(var_num=10, var_min=-500, var_max=500,
                            temp_init=100, temp_final=1, markov=100, alpha=0.98, lr=0.4)
    SA.solve(123456789101112)

    print('Solution:', SA.X_best)
    print('value:', SA.fx_best)

    plt.figure(dpi=200)
    plt.plot(SA.fx_best_, label='best function value')
    plt.plot(SA.fx_, label='function value')
    plt.legend()
    plt.figure(dpi=200)
    plt.plot(SA.temp_, label='temperature')
    plt.legend()
    plt.figure(dpi=200)
    plt.plot(SA.P_inferior_acc_, label='probability of inferior acceptation')
    plt.legend()
    plt.show()






