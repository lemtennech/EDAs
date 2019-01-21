# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 11:39:51 2018

@author: sami
"""

import random
import numpy as np


k = np.loadtxt("knapsack/p05_c.txt", dtype = int)#data[0,:] # knapsack capacity
data = np.loadtxt("data.txt", dtype = int)
w = np.loadtxt("knapsack/p05_w.txt", dtype = int)#data[0,:]
val = np.loadtxt("knapsack/p05_p.txt", dtype = int)#data[0,:]


n = len(val)


def initial(pop):
    popul = []
    while len(popul) < pop:
        i = 0
        p = [int] * n
        while i < n:
            p[i] = random.randint(0, 1)
            i = i + 1

        if (p not in popul): #and (calc_values(p) < k):
            popul.append(p)
    print("init", popul)
    return (popul)


def selection(population):
    fit_values = []
    fit_prop = []
    sum_values = 0
    prob = 0
    mating_pool = []
    for p in population:
        val = calc_values(p)  # claculate fitness value for every chromosome
        fit_values.append(val)
        sum_values += val

    for g in fit_values:  # for make roulette wheel selection
        prob += (g / sum_values) * 100
        fit_prop.append(prob)

    #for i in range((len(population) // 2)+1):
        prob_select = random.uniform(0.0, 100)

    for pro in fit_prop:
        if (pro > prob_select):
            mating_pool = population[fit_prop.index(pro)]
            #population.remove(population[fit_prop.index(pro)])
            break

    return mating_pool

def select(population):


    mat = []
    for i in range((len(population) // 2)+1):
        mat.append(selection(population))

    return mat


def mutation(population):
    for p in population:
        pm = random.uniform(0, 1)  # probability mutation
        if (pm < 0.1):
            x = random.randint(0, len(p) - 1)
            if p[x] == 0:
                p[x] = 1
            else:
                p[x] = 0
    return population


def calc_values(v):
    
    value = 0
    i = 0
    if calc_weight(v) > k :
        value = np.amin(val)
        
    for m in v:
        if m == 1:
            value = value + val[i]

        i = i + 1

    return value


def calc_weight(wei):
    weight = 0

    i = 0
    for m in wei:

        if m == 1:
            weight = weight + w[i]

        i = i + 1

    return weight


def crossover(m):
    gen2 = []
    if (len(m) == 1):
        gen2.append(m[0])

    for a in range(len(m)):
        for b in range(len(m)):
            if a <= b:
                break

            x = random.randint(1, len(m[a]) - 2)  # choose randomly crossover point
            f1 = m[a][:x] + m[b][x:]  # produce offspring f1 and f2 from parents a and b
            f2 = m[b][:x] + m[a][x:]
            if (calc_weight(f1) <= k):
                gen2.append(f1)
            if (calc_weight(f2) <= k):
                gen2.append(f2)

    return gen2


def elemination(gen): # for delete the individuals where the weight is big then capacity


    to_remove = []
    i = 0
    for p in gen:

        if calc_weight(p) > k:
            to_remove.append(i)
        # gen.remove(p)
        i = i + 1

    for i in reversed(to_remove):
        gen.pop(i)

    return (gen)


def elitism(gen): # function to save the best individual in current generation
    elit = 0
    i = 0
    for g in gen:
        temp = calc_values(g)
        if temp > elit:
            elit = temp
            i = gen.index(g)
    print(elit)
    return gen[i]


def evaluate(gen):
    max_val = 0
    i = 0
    for g in gen:
        temp = calc_values(g)
        if temp > max_val:
            max_val = temp

    return max_val


def replacement(gen, gen2, t):



    gen2 = gen2 + gen

    f = []


    for j in range(t) :

        temp = 0
        tp = []

        for i in gen2:
            p = calc_values(i)
            if p > temp :
                temp = p
                tp = i

        if tp not in f :
            f.append(tp)

        for x in gen2:
            if x == tp:
                gen2.remove(x)

    return (f)



def ga(population_size):
    gen = initial(population_size)

    for i in range(200):
        gen = elemination(gen)

        #print(gen)
        elit = elitism(gen)

        p = evaluate(gen)

        mat = select(gen) # mat = mating pool

        gen2 = crossover(mat)
        gen2 = mutation(gen2)

        gen2 = elemination(gen2) # function to remove

        gen = replacement(gen2,gen,population_size) # to produce next generation
        if elit not in gen:
            gen.append(elit)


    return (elit)


print(ga(10), "optimal solution")
