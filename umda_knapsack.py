# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 14:17:44 2018

@author: sami
"""

import random
import numpy as np


k = 165 # knapsack capacity
data = np.loadtxt("data.txt", dtype = int)
w = data[0,:]
val = data[1,:]


n = len(val)


def initial(pop):
	popul = []
	while len(popul) < pop:
		i = 0
		p = [int] * n
		while i < n:
			p[i] = random.randint(0, 1)
			i = i + 1

		if (p not in popul) and (calc_values(p) < k):
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
        val = calc_values(p)  # claculate fitness value for every indiv
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

def calc_values(v):
	
	value = 0
	i = 0
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
            i = g
    print(elit)
    return i


def evaluate(gen):
    max_val = 0
    for g in gen:
        temp = calc_values(g)
        if temp > max_val:
            max_val = temp

    return max_val

def modeling(population):


    population = np.asarray(population)
#    print(population)
    vector_prob = []
    for  column in population.T: #range(len(population)):
        vector_prob.append((np.sum(column))/np.size(population,0))
#    print("vector ",vector_prob)
    return(vector_prob)

def sampling(vector_prob, m):

    gen = []
    for j in range(m//2):
        indiv = np.zeros(len(vector_prob))
#        print(vector_prob)
        for i in vector_prob:
            x = np.random.rand(1)
            #print("rand = ",x, "i = ",i)
            if i >= x:
                indiv[vector_prob.index(i)] = 1
#        print("ind ",indiv)
        gen.append(indiv)

    return(gen)

def umda(population_size):

    gen = initial(population_size)

    for i in range(1000):
#        print("gen",gen)
        gen = elemination(gen)
      
        #print(gen)
        elit = elitism(gen)

        p = evaluate(gen)

        mat = select(gen) # mat = mating pool

        vector_prob = modeling(mat)
        gen2 = sampling(vector_prob, population_size)
        gen = gen2 + mat
#        print(gen)
#        gen = elemination(gen2) # function to remove
             
#        gen = gen2.append(mat)        #gen = replacement(gen2,gen,population_size) # to produce next generation
#        if elit not in gen2:
        gen.append(elit)


    return (elit)

print(umda(30), "optimal solution")

