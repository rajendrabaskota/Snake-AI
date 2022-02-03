import numpy as np
import random
from math import sqrt


class GeneticAlgorithm:
    def __init__(self, population, max_generation, num_of_layers, input_units, hidden_units, output_units, mutation_rate):
        self.population = population
        self.max_generation = max_generation
        self.num_of_layers = num_of_layers
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.weights1 = np.random.randn(population, hidden_units, input_units+1)
        self.weights2 = np.random.randn(population, output_units, hidden_units+1)
        # self.crossover_rate = None
        self.mutation_rate = mutation_rate

        self.initialize_weights()

    def initialize_weights(self):
        # He initialization

        std = sqrt(2.0 / (self.input_units+1))
        self.weights1 = self.weights1 * std

        std = sqrt(2.0 / (self.hidden_units+1))
        self.weights2 = self.weights2 * std

    def compute_fitness(self, datas, max_steps):
        fitness = [data[0] + data[1] * (1- data[1]/max_steps) / max_steps for data in datas]
        return fitness

    def selection(self, fitness):
        individuals = np.random.randint(self.population, size=2)
        parent_1 = individuals[0] if fitness[individuals[0]] > fitness[individuals[1]] else individuals[1]

        individuals = np.random.randint(self.population, size=2)
        parent_2 = individuals[0] if fitness[individuals[0]] > fitness[individuals[1]] else individuals[1]

        return [parent_1, parent_2]

    def crossover(self, parent_1, parent_2):
        weight_1_of_parent_1 = self.weights1[parent_1].flatten()
        weight_1_of_parent_2 = self.weights1[parent_2].flatten()
        
        weight_2_of_parent_1 = self.weights2[parent_1].flatten()
        weight_2_of_parent_2 = self.weights2[parent_2].flatten()

        index_1 = int((self.hidden_units*(self.input_units+1))/2)
        index_2 = int(self.hidden_units*(self.input_units+1))

        weight_1_of_children_1 = np.concatenate((weight_1_of_parent_1[0:index_1], weight_1_of_parent_2[index_1:index_2]))
        weight_1_of_children_2 = np.concatenate((weight_1_of_parent_2[0:index_1], weight_1_of_parent_1[index_1:index_2]))

        index_1 = int((self.output_units*(self.hidden_units+1))/2)
        index_2 = int(self.output_units*(self.hidden_units+1))

        weight_2_of_children_1 = np.concatenate((weight_2_of_parent_1[0:index_1], weight_2_of_parent_2[index_1:index_2]))
        weight_2_of_children_2 = np.concatenate((weight_2_of_parent_2[0:index_1], weight_2_of_parent_1[index_1:index_2]))

        print(len(weight_1_of_parent_1))
        print(len(weight_1_of_parent_2))

        weight_1_of_parent_1 = np.reshape(weight_1_of_parent_1, (self.hidden_units, self.input_units+1))
        weight_1_of_parent_2 = np.reshape(weight_1_of_parent_2, (self.hidden_units, self.input_units+1))
        weight_1_of_children_1 = np.reshape(weight_1_of_children_1, (self.hidden_units, self.input_units+1))
        weight_1_of_children_2 = np.reshape(weight_1_of_children_2, (self.hidden_units, self.input_units+1))

        weight_2_of_parent_1 = np.reshape(weight_2_of_parent_1, (self.output_units, self.hidden_units+1))        
        weight_2_of_parent_2 = np.reshape(weight_2_of_parent_2, (self.output_units, self.hidden_units+1))        
        weight_2_of_children_1 = np.reshape(weight_2_of_children_1, (self.output_units, self.hidden_units+1))        
        weight_2_of_children_2 = np.reshape(weight_2_of_children_2, (self.output_units, self.hidden_units+1))

        return [weight_1_of_parent_1, weight_1_of_parent_2, weight_2_of_parent_1, weight_2_of_parent_2, weight_1_of_children_1, weight_1_of_children_2, weight_2_of_children_1, weight_2_of_children_2]

    def mutation(self, weight_1_of_children_1, weight_1_of_children_2, weight_2_of_children_1, weight_2_of_children_2):
        if random.random() < self.mutation_rate:
            rand_weights = np.random.randn(1, 2)

            std = sqrt(2.0 / (self.input_units+1))
            rand_weights[0][0] = rand_weights[0][0] * std

            std = sqrt(2.0 / (self.hidden_units+1))
            rand_weights[0][1] = rand_weights[0][1] * std

            rand_position_1 = (int(random.random()*(self.hidden_units)), int(random.random()*(self.input_units)))
            rand_position_2 = (int(random.random()*(self.output_units)), int(random.random()*(self.hidden_units)))

            weight_1_of_children_1[rand_position_1] = rand_weights[0][0]
            weight_2_of_children_1[rand_position_2] = rand_weights[0][1]

        if random.random() < self.mutation_rate:
            rand_weights = np.random.randn(1, 2)

            std = sqrt(2.0 / (self.input_units+1))
            rand_weights[0][0] = rand_weights[0][0] * std

            std = sqrt(2.0 / (self.hidden_units+1))
            rand_weights[0][1] = rand_weights[0][1] * std

            rand_position_1 = (int(random.random()*(self.hidden_units)), int(random.random()*(self.input_units)))
            rand_position_2 = (int(random.random()*(self.output_units)), int(random.random()*(self.hidden_units)))

            weight_1_of_children_2[rand_position_1] = rand_weights[0][0]
            weight_2_of_children_2[rand_position_2] = rand_weights[0][1]

        return [weight_1_of_children_1, weight_1_of_children_2, weight_2_of_children_1, weight_2_of_children_2]

    def create_new_generation(self, datas, max_steps):
        weights1 = np.zeros((self.population, self.hidden_units, self.input_units+1))
        weights2 = np.zeros((self.population, self.output_units, self.hidden_units+1))

        fitness = self.compute_fitness(datas, max_steps)
        index_1 = 0
        index_2 = 0

        for i in range(int(self.population/4)):
            [parent_1, parent_2] = self.selection(fitness)

            [weight_1_of_parent_1,
            weight_1_of_parent_2,
            weight_2_of_parent_1,
            weight_2_of_parent_2,
            weight_1_of_children_1,
            weight_1_of_children_2,
            weight_2_of_children_1,
            weight_2_of_children_2] = self.crossover(parent_1, parent_2)

            [weight_1_of_children_1,
            weight_1_of_children_2,
            weight_2_of_children_1,
            weight_2_of_children_2] = self.mutation(weight_1_of_children_1, weight_1_of_children_2, weight_2_of_children_1, weight_2_of_children_2)

            weights1_many = [weight_1_of_parent_1, weight_1_of_parent_2, weight_1_of_children_1, weight_1_of_children_2]
            weights2_many = [weight_2_of_parent_1, weight_2_of_parent_2, weight_2_of_children_1, weight_2_of_children_2]

            for element in weights1_many:
                weights1[index_1] = element
                index_1 += 1

            for element in weights2_many:
                weights2[index_2] = element
                index_2 += 1

        self.weights1 = weights1
        self.weights2 = weights2
