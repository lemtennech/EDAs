#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:39:40 2019

@author: sami
"""
import random
import numpy as np

data = np.loadtxt("tspd1.txt", dtype = int)
p_size = np.size(data, 0)

def create_individual():
    individual = np.random.permutation(np.arange(np.size(data, 0)))
    individual = np.append(individual, individual[0])
    return(individual)
d = create_individual()
def initial(population_size):
    population = []
    for i in range(population_size):
        population.append(create_individual())

    population = np.asarray(population)
    print(population)
    return(population)


def evaluate(individual):
    d = 0
    for i in range(np.size(individual)-1):
        d = d + data[individual[i], individual[i+1]]


    return(d)


def selection(population):
    selected_set = []
    c = test(population)
#    print(c,"ccccc")
    s = np.argsort(c)
    for i in range(len(s)//2):
        selected_set.append(population[s[i]])
    return(selected_set)


    
def delta(indiv, i, j):
    delta = 0 
    indiv = np.asarray(indiv)
    l = np.size(indiv, 1)
    for x in indiv:
#        print(x, delta)
        
        for k in range(l):
            if(x[k] == i and x[(k+1)% l]  ==j):
                delta = delta + 1
    return delta

#building Edge Histogram Matrix
    
def modeling_ehm(selected_set):
    selected_set = np.asarray(selected_set)
    model =np.zeros((p_size, p_size) )
    for i in range(p_size):
      
        
        for j in range(p_size):
            if (i == j):
                model[i, j] = 0
            else:
                model[i, j] = delta(selected_set, i, j)+delta(selected_set, j, i)+0.1
            
    return(model)
  
def sampling(ehm):
    prob = 0
    selected = 0
    vect_prob = []

    for i in ehm:
            
        prob += i / np.sum(ehm)
        vect_prob.append(prob)


    prob_select = random.uniform(0.0, 1)

    
    for pro in vect_prob:
        if (pro >= prob_select):
            selected = vect_prob.index(pro)
            break
    return selected
            
            
   


def create_indiv(model1):
    model=np.copy(model1)
    
    individual = []
    
    for i in range(p_size):
        if i == 0:
            individual.append( random.randint(0, p_size-1))
#            print(individual[0])
            model[:,individual[0]]= 0
        else:
            individual.append(sampling(model[i-1,:]))
            model[:,individual[i]]=0
            
    individual = np.append(individual, individual[0])
#    print(individual)
            
    return individual 
 
    
def test(pop):
    fit = []

    for i in pop:
        fit.append(evaluate(i))
    return(fit)
    
    
def ehsba(population_size):
    gen1 = initial(population_size)
    for i in range(50):
        selected_set = selection(gen1)
        model = modeling_ehm(selected_set)
        for j in range(population_size):
           
            mod =np.copy(model)
            gen1[j] = create_indiv(mod)
        print (min(map(evaluate,gen1)))
    return gen1

ehsba(250)