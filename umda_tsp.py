#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:44:14 2019

@author: sami
"""

import random
import numpy as np
from deap import base
from deap import creator
import matplotlib.pylab as plt
from deap import tools, algorithms, benchmarks



class umda_tsp:
    
    def __init__(self, size):
        self.size = size
        self.data = np.loadtxt("tspd1.txt", dtype = int)
        self.population = toolbox.population(n=10)
#        self.selected_f = np.append(np.array([[999999.0] for i in range(5)]),self.population, axis=1)
        self.Model = np.zeros((np.size(self.data, 0), np.size(self.data, 0)) )
        self.update(self.population)
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
    
    
    def selection(self, population):
        pop = tools.selBest(population, k=10, fit_attr='fitness')
        selected_set = np.array(pop)
#        selected_set = []
#        c = self.test(population)
#        s = np.argsort(c)
#        for i in range(len(s)//2):
#            selected_set.append(population[s[i]])
        return(selected_set)
    
    
    def update(self, population):
#        print("pop", population)
        selected = tools.selBest(population, k=5, fit_attr='fitness')
        selected_set = np.array(selected)
#        print("selected_set", selected_set)
        prob = np.zeros(np.size(self.data, 0))
#        self.model =np.zeros((np.size(self.data, 0), np.size(self.data, 0)) )
        for i in range(np.size(self.data, 0)-1):
          
            
            for j in range(np.size(self.data, 0)-1):
                d = ((selected_set[:, i] == j).sum())/np.size(selected_set, 0)
#                print(d, " +++ i = ", i, " +++ j : ", j)
                self.Model[i, j] = d
        
            
        
    def adjusting(self, model, col):
       
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
                    
     
    def sampling(self, model):
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
                
                
        
    
    
    def create_indiv(self):
#        print("model befor ", self.Model)
        individual = []#np.zeros(np.size(data, 0))
#        print("model", model)
        model = np.copy(self.Model)
        for i in self.Model:
    #        print("to sample ", i)
            k = self.sampling(i)
            individual.append(k)
            self.adjusting(model, k)
        individual = np.append(individual, individual[0])
#        print("indiv creer ", individual)
        return individual 
        
        
    def test(self, pop):
        fit = []
    
        for i in pop:
            fit.append(self.evaluate(i))
        return(fit)
        
    def generate(self, ind_init):
        return [ind_init(self.create_indiv()) for _ in range(self.size)]
   
def main():
    IND_SIZE =25
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
  

    
   
    #
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#    toolbox.population(n=100)
#    print("   ",toolbox.individual())  
    umda = umda_tsp(100)
    
    toolbox = base.Toolbox()
    toolbox.register("evaluate", umda.evaluate)
    
    toolbox.register("generate", umda.generate, creator.Individual)

 
    toolbox.register("update", umda.update)
#    toolbox.register("evaluate", umda.evaluate)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
#    stats.register("avg", np.mean)
#    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaGenerateUpdate(toolbox, ngen=30, stats=stats)
   
    
    
    
if __name__ == "__main__":
    main()
    
    

#    gen1 = initial(population_size)
#    for i in range(100):
##        selected_set = alg.selection(gen1)
##        print("++++++++", selected_set)
#        model = alg.modeling(gen1)
##        print("///// model ",i," ", model)
#        for j in range(population_size):
#            mod =np.copy(model)
#            gen1[i] = create_indiv(mod)
##            print(gen1[i])
#        print (min(map(evaluate,gen1)))
#    return gen1
#gen1 = umda(100)
#x = test(gen1)
            
        

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
