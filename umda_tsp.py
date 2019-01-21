#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:44:14 2019

@author: sami
"""

import random
import numpy as np

data = np.loadtxt("tspd3.txt", dtype = int)

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
        selected_set.append(population[s[i]])
    return(selected_set)


def modeling(selected_set):
    selected_set = np.asarray(selected_set)
    prob = np.zeros(np.size(data, 0))
    model =np.zeros((np.size(data, 0), np.size(data, 0)) )
    for i in range(np.size(data, 0)):
      
        
        for j in range(np.size(data, 0)):
            d = ((selected_set[:, i] == j).sum())/np.size(selected_set, 0)
#            print(d, " +++ i = ", i, " +++ j : ", j)
            model[i, j] = d
       
        np.append(model, prob)
    return(model)
    
def adjusting(model, col):
   
    k = 0
    for i in model:
        indice = np.where(i != 0 )
        
        if (i[col] != 0):
            k = i[col]
           
            i[col] = 0
            for j in range(np.size(i)):
                if (i[j] != 0):
#                    print(j)
                    i[j] = i[j]+ (k / (np.size(indice)-1))
                    
                              
    return(model)
                
 
def sampling(model):
    prob = 0
    selected = 0
    vect_prob = []

    for i in model:
            
        prob += i
        vect_prob.append(prob)

    prob_select = random.uniform(0.00, 1)
    
    for pro in vect_prob:
        if (pro >= prob_select):
            selected = vect_prob.index(pro)
            break
    return selected
            
            
    


def create_indiv(model):
    
    individual = []#np.zeros(np.size(data, 0))
    for i in model:
#        print("to sample ", i)
        k = sampling(i)
        individual.append(k)
        adjusting(model, k)
    individual = np.append(individual, individual[0])
    return individual 
    
    
def test(pop):
    fit = []

    for i in pop:
        fit.append(evaluate(i))
    return(fit)
    
    
def umda(population_size):
    gen1 = initial(population_size)
    for i in range(100):
        selected_set = selection(gen1)
        print("++++++++", selected_set)
        model = modeling(selected_set)
        print("///// model ",i," ", model)
        for j in range(population_size):
            mod =np.copy(model)
            gen1[i] = create_indiv(mod)
            print(gen1[i])
            
    return gen1
gen1 = umda(200)
x = test(gen1)
            
        

#print("-----------   iniitl pop ----------------------")
#x = test(p)
#
#print(p)
#print(x)
#print("--------------- selection ------------------")
#n = selection(p)
#print(n)
#p1 = test(n)
#print(p1)
#print("------------ modiling --------------")
#np.asarray(n)
#mod = modiling(n)
#
#print(mod)
#n = np.asarray(n)
#print("q sa;l", mod[0,:])
#indiv = create_indiv(mod)
#
#
#
#
