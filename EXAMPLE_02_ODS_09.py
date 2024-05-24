# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:11:28 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 2 ODS 09
#==============================================================================
"""
Mohammadhossein Ghahramani (2020) develop a hybrid method that combines an 
evolutionary algorithm with a neural network to optimize control in smart 
manufacturing. This approach improves sensor data management in manufacturing, 
addressing challenges such as real-time control and maintenance optimization. 
By using advanced technologies and analytics, this method offers valuable 
insights to implement effective predictive technologies in smart 
manufacturing., 

Mohammadhossein Ghahramani. Data-driven Predictive Analysis 
for Smart Manufacturing Processes Based on a Decomposition Approach. TechRxiv. 
July 28, 2021. DOI: 10.36227/techrxiv.15045426.v1.

This Python code integrates an evolutionary algorithm with a neural network to 
optimize control in smart manufacturing by improving sensor data management. 
The evolutionary algorithm, implemented using the DEAP library, optimizes the 
hyperparameters of the neural network (number of epochs and batch size) to 
minimize the loss on the sensor data. The neural network is built using 
TensorFlow/Keras and trained on synthetic sensor data. The evolutionary 
algorithm evolves a population of individuals, each representing a set of 
hyperparameters, to find the best combination that optimizes the neural 
network's performance.
"""
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the problem to optimize: sensor data management in smart manufacturing
# This function represents the fitness function to optimize using the evolutionary algorithm
def fitness_function(individual, sensor_data):
    # Neural network model architecture
    model = Sequential()
    model.add(Dense(64, input_dim=sensor_data.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Split the sensor data into training and testing sets
    train_data, test_data = np.split(sensor_data, [int(0.8 * len(sensor_data))])
    train_labels = train_data[:, -1]
    test_labels = test_data[:, -1]

    # Train the model on the training data
    model.fit(train_data, train_labels, epochs=individual[0], batch_size=individual[1], verbose=0)

    # Evaluate the model on the test data
    loss = model.evaluate(test_data, test_labels, verbose=0)
    return loss,

# Initialize the DEAP framework for the evolutionary algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=100, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function, sensor_data=np.random.rand(100, 10)) # Placeholder for sensor data

# Main algorithm function
def main():
    population = toolbox.population(n=50)
    ngen = 40
    cxpb = 0.5
    mutpb = 0.2

    # Run the evolutionary algorithm
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                        stats=None, halloffame=None, verbose=True)

    # Extract the best individual from the final population
    best_individual = tools.selBest(population, 1)[0]
    print("Best individual is: %s\nwith fitness: %s" % (best_individual, best_individual.fitness.values))

if __name__ == "__main__":
    main()
