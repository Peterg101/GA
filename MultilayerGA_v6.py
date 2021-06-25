import numpy as np
import pandas as pd
import csv
import random

class MultilayerGA():

    def __init__(self):
        self.f = 8e9
        
        
        self.num_genes = 3 #Number of layers
        self.num_chromosones = 20
        self.num_parents = int(self.num_chromosones/2)
        self.num_materials = 16
        self.num_offspring = 10
        self.num_data_points = 5
        self.database_length = 16  #len(self.data)
        self.crossover_point = np.uint8(self.num_genes/2)
        self.population_size = (self.num_genes *self.num_chromosones)
        self.data_points = 20
        self.num_positions = 4
        self.penalty = -10000
        self.fitness_penalty = 1000
        self.num_frequencies = 41
        self.increment = 0.1e9
        self.GHz = 1e8 #GHz actually 1e9, this is used for division purposes
        self.max_f = self.f+((self.num_frequencies-1)*self.increment)
        self.c = 3e8
        self.wl = self.c/self.f
        self.pi = np.pi
        self.z_air = 377 #Ohm
        self.j = complex(0,1)

        #Loop Variables
        self.num_epochs = 5#
        self.num_generations = 100
        
        
    def generate_population(self):
        
        self.population = np.random.choice(self.num_materials, size = self.population_size).reshape(self.num_chromosones, self.num_genes)
    
    def calculate_fitness(self):
        #Constants
        self.data = pd.read_csv(self.filename, delimiter = ",", encoding = "ISO-8859-1").to_numpy()
        
        
        
        self.RL_fitness_array = np.zeros((self.num_chromosones, 1), dtype = float)
        self.fitness_penalty_array = np.zeros((self.num_chromosones, 1), dtype = float)
        
        for j in range(0, self.num_chromosones):
            self.first_layer = self.population[j][0]
            self.first_layer_e = self.data[self.first_layer][1]
            self.first_layer_u = self.data[self.first_layer][2]
            self.first_layer_d = self.data[self.first_layer][3]
            self.zm_first_layer = np.sqrt(self.first_layer_u/self.first_layer_e)
            z_first = self.zm_first_layer*(np.tanh((self.pi*np.sqrt(self.first_layer_e*self.first_layer_u))/self.wl)*self.first_layer_d)
            
            z_list = [z_first]
            zm_list = [self.zm_first_layer]
            
                
            #Calculation of RL   
            for i in range(1, self.num_genes):
                self.layer = self.population[j][i]
                self.layer_e = self.data[self.layer][1]
                self.layer_u = self.data[self.layer][2]
                self.layer_d = self.data[self.layer][3]
                self.zm_layer = np.sqrt(self.layer_u/self.layer_e)
                self.z_layer_minus1 = z_list[i-1]
                self.zm_layer_minus1 = zm_list[i-1]
                
                self.z_layer = self.zm_layer *((self.z_layer_minus1+(self.zm_layer*np.tanh((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/self.wl)*self.layer_d)))/(self.zm_layer+(self.z_layer_minus1*np.tanh(((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/self.wl)*self.layer_d)))
                                
                zm_list.append(self.zm_layer)
                z_list.append(self.z_layer)
                
            z_array= np.array(z_list)
            zm_array = np.array(zm_list)

            #Final RL Calculation
            RL = 20*np.log(abs((z_array[self.num_genes-1]-1)/(z_array[self.num_genes-1]+1)))
            
            #Impedance Matching
            zm_array_sorted = np.sort(zm_array)
        
            if (np.array_equal(zm_array, zm_array_sorted)):
                impedance_matcher = False
            else:
                impedance_matcher = True
        
            impedance_penalty = (int(impedance_matcher)*self.fitness_penalty)
            #Unique Materials 
            unique_materials = (len(np.unique(self.population[j])))
              
            unique_penalty=(int(unique_materials != self.num_genes)*self.fitness_penalty)

            self.RL_fitness_array[j] = RL + unique_penalty+impedance_penalty

        
        
    def select_mating_pool(self, fitness_array):
        
        self.parents = np.zeros((self.num_parents, self.num_genes), dtype = int)
        
        for i in range(self.num_parents):
            self.max_fitness_idx = np.where(fitness_array==np.amin(fitness_array))
            self.max_fitness_idx=self.max_fitness_idx[0][0]
            self.parents[i, :] = self.population[self.max_fitness_idx, :]
            fitness_array[self.max_fitness_idx] = self.fitness_penalty
        
    def crossover(self):
        self.offspring = np.zeros((self.num_offspring, self.num_genes), dtype = int)
        
        for i in range(self.num_offspring):
            self.parent_1_idx = i%self.num_parents
            self.parent_2_idx = (i+1)%self.num_parents
            self.offspring[i, 0:self.crossover_point] = self.parents[self.parent_1_idx, 0:self.crossover_point]
            self.offspring[i, self.crossover_point:] = self.parents[self.parent_2_idx, self.crossover_point:]   
        

        
    def mutation(self):
        
        for i in range(self.num_parents):
          self.random_gene = random.randint(0, self.num_genes-1)
          self.random_material = random.randint(0, self.num_materials-1)
          self.mutant_material = self.data[self.random_material]
          self.offspring[i][self.random_gene] = self.mutant_material[0]
          
    def stacker(self, population):
        self.population[0:self.num_parents, :] = self.parents
        self.population[self.num_parents:, :] = self.offspring

        
    def genetic_algorithm(self):
        
        
        best_data_array = np.zeros((self.num_frequencies, self.num_data_points), dtype = float)
        for x in range(0, self.num_frequencies):
            self.filename = "databases/multilayer_database_"+str(int(self.f+(self.increment*x)))+".csv"
            frequency = str(self.f+(self.increment*x))
            print("Frequency " + frequency)
            
            best_absorption_value_array = np.zeros((self.num_epochs, 1), dtype = float)
            best_combinations_array = np.zeros((self.num_epochs, self.num_genes), dtype = int)
            
            for j in range(self.num_epochs):
              #print("Epoch " +str(j+1))
              
              
              self.generate_population()
          
              for i in range(self.num_generations):
                #print("Frequency " +str(i+1))
                self.calculate_fitness()
                if i == self.num_generations -1:
                    best_absorption_value_array[j] = self.RL_fitness_array[0]
                    best_combinations_array[j] = self.population[0]
                
                self.select_mating_pool(self.RL_fitness_array)
                self.crossover()
                self.mutation()
                self.stacker(self.population)
            minimum_index = np.argmin(best_absorption_value_array)
            best_data_array[x][0] = frequency
            best_data_array[x][1:4] = best_combinations_array[minimum_index]
            best_data_array[x][4] = best_absorption_value_array[minimum_index]
            
        
        best_DataFrame = pd.DataFrame(best_data_array, columns = ["Frequency (GHz)", "Material 1", "Material 2", "Material 3", "Max Absorption (dB)"])
        best_DataFrame.to_csv("Best_Data_GA.csv", index = None)    

peter_season = MultilayerGA()


#peter_season.genetic_algorithm()
