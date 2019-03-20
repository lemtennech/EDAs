#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:22:00 2019

@author: sami
"""

import random
import numpy as np
from deap import base
from deap import creator
import matplotlib.pylab as plt
from deap import tools, algorithms, benchmarks



class deap_ehsba_tsp:
    
    def __init__(self, size):
        self.size = size
        self.data = np.loadtxt("tspd1.txt", dtype = int)
        self.population = toolbox.population(n=10)
#        self.selected_f = np.append(np.array([[999999.0] for i in range(5)]),self.population, axis=1)
        self.Model = np.zeros((np.size(self.data, 0), np.size(self.data, 0)) )
        self.update(self.population)
        self.selected = []
#        print("model",self.Model)
       
        
    
    def create_individual(self):
        individual = np.random.permutation(np.arange(np.size(self.data, 0)))
        individual = np.append(individual, individual[0])
        return(individual)
    
    def initial(self, population_size):
        population = []
        for i in range(population_size):
            population.append(self.create_individual())
    
        population = np.asarray(population)
#        print(population)
        
        return(population)
    
    
    def evaluate(self, individual):
        d = 0
#        print(individual)
        for i in range(np.size(individual)-1):
#            print("individual[i] " ,individual[i])
            
            d = d + self.data[individual[i], individual[i+1]]
    
#        print(d)
        return d,
    
    
    
    def delta(self, indiv, i, j):
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
    
    def modeling_ehm(self, selected_set):
        selected_set = np.asarray(selected_set)
        model =np.zeros((p_size, p_size) )
        for i in range(p_size):
          
            
            for j in range(p_size):
                if (i == j):
                    model[i, j] = 0
                else:
                    model[i, j] = self.delta(selected_set, i, j)+0.1#self.delta(selected_set, j, i)+0.1
                
        return(model)
     
    def update(self, population):
#        print("pop", population)
        self.selected = tools.selBest(population, k=25, fit_attr='fitness')
        selected_set = np.array(self.selected)
#        print(selected_set)
#        print("selected_set", selected_set)
        prob = np.zeros(np.size(self.data, 0))
#        self.model =np.zeros((np.size(self.data, 0), np.size(self.data, 0)) )
        for i in range(np.size(self.data, 0)):
          
            
            for j in range(np.size(self.data, 0)):
                if (i == j):
                    self.Model[i, j] = 0
                else:
                    self.Model[i, j] = self.delta(selected_set, i, j)+self.delta(selected_set, j, i)+0.1
        
            
    def sampling(self, ehm):
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
            
    
    
    def create_indiv(self):
        model=np.copy(self.Model)
    
        individual = []
    
        for i in range(np.size(self.data, 0)):
            if i == 0:
                individual.append( random.randint(0, np.size(self.data, 0))-1)
    #            print(individual[0])
                model[:,individual[0]]= 0
            else:
                individual.append(self.sampling(model[individual[i-1],:]))
                model[:,individual[i]]=0
                
        individual = np.append(individual, individual[0])
#        print(individual)
            
        return individual 
        
        
    def test(self, pop):
        fit = []
    
        for i in pop:
            fit.append(self.evaluate(i))
        return(fit)
        
    def generate(self, ind_init):
        return [ind_init(self.create_indiv()) for _ in range(75)] + self.selected

IND_SIZE =26
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

    

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
  

    
   
    #
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  
def main():
   #    toolbox.population(n=100)
#    print("   ",toolbox.individual())  
    ehsba_tsp = deap_ehsba_tsp(100)
    
    toolbox = base.Toolbox()
    toolbox.register("evaluate", ehsba_tsp.evaluate)
    
    toolbox.register("generate", ehsba_tsp.generate, creator.Individual)

 
    toolbox.register("update", ehsba_tsp.update)
#    toolbox.register("evaluate", umda.evaluate)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
#    stats.register("avg", np.mean)
#    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaGenerateUpdate(toolbox, ngen=200, stats=stats)
    
   
    
    
    
if __name__ == "__main__":
    main()
    
    