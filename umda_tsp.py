#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:44:14 2019

@author: sami
"""

import random
import numpy as np

data = np.loadtxt("tspd1.txt", dtype = int)

def create_individual():
    individual = np.random.permutation(np.arange(np.size(data, 0)))
    individual = np.append(individual, individual[0])
    return(individual)

def initial(population_size):
    population = []
    for i in range(population_size):
        population.append(create_individual())

    population = np.asarray(population)
    print(population)
    return(population)

p = initial(100)


def evaluate(individual):
    d = 0
    for i in range(np.size(individual)-1):
        d = d + data[individual[i], individual[i+1]]


    return(d)



def selection(population):
    selected_set = []
    c = test(population)
    s = np.argsort(c)
    for i in range(len(s)//2):
        selected_set.append(p[s[i]])


    return(selected_set)

def modiling(selected_set):
    selected_set = np.asarray(selected_set)
    prob = np.zeros(np.size(data, 0))
    model =np.zeros((np.size(data, 0), np.size(data, 0)) )
    for i in range(np.size(data, 0)):
        print(selected_set[:, i])
        
        for j in range(np.size(data, 0)):
            d = ((selected_set[:, i] == j).sum())/np.size(selected_set, 0)
#            print(d, " +++ i = ", i, " +++ j : ", j)
            model[i, j] = d
        print(prob)
        np.append(model, prob)
    return(model)


def test(pop):
    fit = []

    for i in pop:
        fit.append(evaluate(i))
    return(fit)

print("-----------   iniitl pop ----------------------")
x = test(p)

print(p)
print(x)
print("--------------- selection ------------------")
n = selection(p)
print(n)
p1 = test(n)
print(p1)
print("------------ modiling --------------")
np.asarray(n)
mod = modiling(n)

print(mod)
n = np.asarray(n)




