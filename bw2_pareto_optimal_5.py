import numpy as np
import pandas as pd
import csv
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MultilayerGA_v6 import peter_season as ps

class BW_Optimiser():
    def __init__(self, file_name, file_name_2):
        
        #Imported values
        self.num_genes = ps.num_genes
        self.num_chromosones = ps.num_chromosones
        self.num_parents = ps.num_parents
        self.num_offspring = ps.num_offspring
        self.num_data_points = 4
        self.num_materials = 16 #len(self.data)
        self.crossover_point = ps.crossover_point
        self.population_size = ps.population_size
        self.data_points = ps.data_points
        self.num_generations = ps.num_generations
        self.fitness_penalty = -1000000000000000
        #self.d=ps.d

        #Penalties
        self.fitness_penalty = -1000
        self.max_number = self.fitness_penalty*10
        
        self.minimum_frequency = 8.4e9
        self.increment = 0.1e9
        self.c = 3e8
        self.f = 8e9
        self.wl = self.c/self.f
        self.pi = np.pi
        
        #Optimiser values
        self.best_data = pd.read_csv(file_name_2, delimiter = ',').to_numpy()
        self.freq_1 = 8
        self.freq_interval = 0.1
        self.freq_2 = self.freq_1+self.freq_interval
        self.num_freqs = 5
        self.max_freq = ((self.num_freqs)*self.freq_interval)+self.freq_1
        self.num_x_layers = ps.num_genes-1
        self.percentage_score = 10000
        self.unique_materials_penalty = 10000000000
        self.impedance_penalty = 10000000000
        self.x_pen_2 = 560000000000
        self.epoch_bandwidth = 1
        self.generation_bandwidth = 10
        self.number_of_frequencies = 41

        #GA Factors
        self.num_epochs_bw = 10
        self.num_generations_bw = 100
        self.restricted_mode = False

        #Data Acquisition
        self.full_data_array = pd.read_csv("Best_Data_GA.csv", delimiter = ",", encoding = "ISO-8859-1").to_numpy()
        self.optimal_data = np.concatenate((self.full_data_array[:,0].reshape(self.number_of_frequencies,1) , self.full_data_array[:,4].reshape(self.number_of_frequencies,1)), axis = 1)
        
        
    def generate_population(self):
        self.population = np.random.choice(self.num_materials-1, size = self.population_size).reshape(self.num_chromosones, self.num_genes)
        
        
    def select_optimal_values_and_relevant_data(self):
        self.optimal_values_array = (self.optimal_data[:,1][:self.num_freqs])
        
        self.relevant_data_array = np.zeros((self.num_freqs, self.num_materials, self.num_data_points), dtype = float)

        for i in range(0, self.num_freqs):
            import_data = pd.read_csv("databases/multilayer_database_"+str(int(self.minimum_frequency+(self.increment*i)))+".csv").to_numpy()
            self.relevant_data_array[i] = import_data
          
    def calculate_fitness_2(self):

        #print(self.relevant_data_array)
        #self.RL_values_array = np.zeros((self.num_freqs, self.num_chromosones, 1), dtype = float)
        self.RL_fitness_array = np.zeros((self.num_freqs, self.num_chromosones, 1), dtype = float)
        self.fitness_penalty_array = np.zeros((self.num_freqs, self.num_chromosones, 1), dtype = float)

        #print(RL_array)
        for f in range(0, self.num_freqs):
            

            for j in range(0, self.num_chromosones):
              #Top layer
              self.first_layer = self.population[j][0]
              self.first_layer_e = self.relevant_data_array[f][self.first_layer][1]
              self.first_layer_u = self.relevant_data_array[f][self.first_layer][2]
              self.first_layer_d = self.relevant_data_array[f][self.first_layer][3]
              self.zm_first_layer = np.sqrt(self.first_layer_u/self.first_layer_e)
              z_first = self.zm_first_layer*(np.tanh((np.pi*np.sqrt(self.first_layer_e*self.first_layer_u))/self.wl)*self.first_layer_d)
            
              z_list = [z_first]
              zm_list = [self.zm_first_layer]
            
              #Calculation of RL
              for i in range(1, self.num_genes):
                  
                  self.layer = self.population[j][i]
                  self.layer_e = self.relevant_data_array[f][self.layer][1]
                  self.layer_u = self.relevant_data_array[f][self.layer][2]
                  self.layer_d = self.relevant_data_array[f][self.layer][3]
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
              
              #Better than optimal solution penalty
              optimal_solution = self.optimal_values_array[f]
              
              better_than_optimal_penalty = (int(RL <= optimal_solution)*self.fitness_penalty)
              
              self.fitness_penalty_array[f][j] = unique_penalty+impedance_penalty+better_than_optimal_penalty
              
              self.RL_fitness_array[f][j] = RL/optimal_solution

        #self.RL_values_array = self.RL_values_array + self.fitness_penalty_array     
        self.RL_fitness_array = self.RL_fitness_array + self.fitness_penalty_array
        
        self.compiled_fitness_array = np.sum(self.RL_fitness_array, axis = 0)
        
        
    def select_mating_pool(self):
        
        self.parents = np.empty((self.num_parents, self.num_genes), dtype = np.object)
        for i in range(self.num_parents):
            self.max_fitness_idx = np.where(self.compiled_fitness_array==np.amax(self.compiled_fitness_array))
            self.max_fitness_idx=self.max_fitness_idx[0][0]
            self.parents[i, :] = self.population[self.max_fitness_idx, :]
            self.compiled_fitness_array[self.max_fitness_idx] = (self.fitness_penalty-1000000000)
            
        
    def crossover(self):
        self.offspring = np.empty((self.num_offspring, self.num_genes), dtype = np.object)

        for i in range(self.num_offspring):
            self.parent_1_idx = i%self.num_parents
            self.parent_2_idx = (i+1)%self.num_parents
            self.offspring[i, 0:self.crossover_point] = self.parents[self.parent_1_idx, 0:self.crossover_point]
            self.offspring[i, self.crossover_point:] = self.parents[self.parent_2_idx, self.crossover_point:]


    def mutation(self):
        
        for i in range(10):
          self.random_gene = random.randint(0, self.num_genes-1)
          self.random_material = random.randint(0, self.num_materials-1)
          self.mutant_material = random.randint(0, self.num_materials-1)
          self.offspring[i][self.random_gene] = self.mutant_material
          
    def stacker(self):
        self.population[0:self.num_parents, :] = self.parents
        self.population[self.num_parents:, :] =  self.offspring
        
    def BW_GA(self):
        Columns = ["Material", "e", "u"]
        
        self.complete_population = np.zeros((self.num_epochs_bw, self.num_generations_bw, self.num_chromosones, self.num_genes), dtype = int)
        self.complete_RL_values_array = np.zeros((self.num_epochs_bw, self.num_generations_bw, self.num_freqs, self.num_chromosones, 1), dtype = float)
        self.best_combination_array = np.zeros((self.num_epochs_bw, self.num_genes), dtype = int)
        self.best_RL_values = np.zeros((self.num_epochs_bw, self.num_freqs), dtype = float)
        for j in range(self.num_epochs_bw):
              print("Epoch " +str(j+1))
              best_combinations = []
              self.generate_population()
              self.select_optimal_values_and_relevant_data()
              for i in range(self.num_generations_bw):
                #print("Generation " +str(i+1))
                self.calculate_fitness_2()
                self.complete_RL_values_array[j][i] = self.RL_fitness_array
                self.complete_population[j][i] = self.population
                self.select_mating_pool()


                #print(self.parents)
                #print(self.RL_fitness_array[0:, 0:self.num_parents])
                #kjb,jlh
                self.crossover()
                self.mutation()
                self.stacker()
              #print(self.RL_fitness_array)
              self.best_combination_array[j] = self.population[0]
              self.best_RL_values[j][0] = (self.RL_fitness_array[0][0]*self.optimal_values_array[0])
              self.best_RL_values[j][1] = (self.RL_fitness_array[1][0]*self.optimal_values_array[1])
              
        
        
       
        x_values = []
        y_values = []
        z_values = []

        if self.num_freqs == 2:
            for a in range(0, self.num_epochs_bw):
                for b in range(0,self.num_generations_bw ):
                    
                      
                      for d in range(0, self.num_chromosones):
                          
                          x_values.append(self.complete_RL_values_array[a][b][0][d]*self.optimal_values_array[0])
                          
                          y_values.append(self.complete_RL_values_array[a][b][1][d]*self.optimal_values_array[1])
                          

            x = np.array(x_values)
            y = np.array(y_values)


            plt.scatter(x,y)
            plt.xlabel("Difference from optimum at " +str(self.freq_1) + " GHz" , fontsize=7)
            plt.ylabel("Difference from optimum at " +str(self.freq_1+self.freq_interval) + " GHz", fontsize =7)
            plt.title("Pareto Optimal Solutions for " +str(self.freq_1) + " GHz and " + str(self.freq_1+self.freq_interval) + " GHz")
            plt.xlim(-3,0)
            plt.ylim(-3,0)
            plt.show()
        
        
        elif self.num_freqs == 3:
            

          for a in range(0, self.num_epochs_bw):
                for b in range(0,self.num_generations_bw ):
                    
                      
                      for d in range(0, self.num_chromosones):
                          
                          x_values.append(self.complete_RL_values_array[a][b][0][d]*self.optimal_values_array[0])
                          
                          y_values.append(self.complete_RL_values_array[a][b][1][d]*self.optimal_values_array[1])

                          z_values.append(self.complete_RL_values_array[a][b][2][d]*self.optimal_values_array[2])
                          
          x = np.array(x_values).reshape(len(x_values))
          y = np.array(y_values).reshape(len(y_values))
          z = np.array(z_values).reshape(len(z_values))

          
          
          ax = plt.axes(projection='3d')
          ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
          ax.set_xlabel("Difference from optimum at " +str(self.freq_1) + " GHz" , fontsize=7, rotation = 0)
          ax.set_ylabel("Difference from optimum at " +str(self.freq_1+self.freq_interval) + " GHz", fontsize =7, rotation = 180)
          ax.set_zlabel("Difference from optimum at " +str(self.freq_1+self.freq_interval+self.freq_interval) + " GHz", fontsize =7, rotation = 180)
          ax.set_title("Pareto Optimal Solutions for " +str(self.freq_1) + " GHz, " + str(self.freq_1+self.freq_interval) +" and " + str(self.freq_1+self.freq_interval+self.freq_interval)  + " GHz")
          plt.xlim(-3,0)
          plt.ylim(-3,0)
          ax.set_zlim(-3,0)
          plt.show()
          
          
          
        elif self.num_freqs == 1 or self.num_freqs > 3:
            
            RL_values = []
            
            combinations = []
            
            for a in range(0, self.num_epochs_bw):
                
                for b in range(0, self.num_generations_bw):
                    
                    eval_array = (self.complete_RL_values_array[a][b])
                    
                    eval_array_sum = np.sum((eval_array), axis = 0)
                    
                    for c in range(0, self.num_chromosones):

                        if eval_array_sum[c] > 0:
                            
                            RL_values.append(eval_array[0:, c])
                            
                            combinations.append(self.complete_population[a][b][c])
                            
            RL_values_arr = np.array(RL_values)

            combinations_arr = np.array(combinations)

            uniqueValues, indicesList = (np.unique(combinations_arr, axis = 0, return_index = True))

            final_array = np.zeros((indicesList.shape[0], self.num_genes+self.num_freqs+1), dtype = float)
              
            
            for i in range(0, indicesList.shape[0]):
                indi = indicesList[i]
                final_array[i][0:self.num_genes] = combinations_arr[indicesList[i]]
                final_array[i][self.num_genes:(self.num_genes+self.num_freqs)] = RL_values_arr[indi].reshape(1,self.num_freqs)
                final_array[i][self.num_genes+self.num_freqs] = np.sum(final_array[i][self.num_genes:self.num_genes+self.num_freqs])
                
            
            final_array = final_array[np.argsort(final_array[:, self.num_genes+self.num_freqs])]
            print((final_array))
            
bandwidth_optimiser = BW_Optimiser("multilayer_database_complex.csv", "best_data.csv")

bandwidth_optimiser.BW_GA()


