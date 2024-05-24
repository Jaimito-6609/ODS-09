# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:42:19 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 3 ODS 09
#==============================================================================
"""
Hsu, Jiang, and Lin (2023) examine how artificial intelligence and smart grid 
optimization improve smart manufacturing, highlighting advances in energy 
efficiency and sustainability. They analyze innovations and challenges in the 
integration of renewable energy, 5G/B5G technologies and the development of 
future manufacturing systems, providing guidelines for additional research. 
Hsu C-C, Jiang B-H, Lin C-C. A Survey on Recent Applications of Artificial 
Intelligence and Optimization for Smart Grids in Smart Manufacturing. Energies. 
2023; 16(22):7660. https://doi.org/10.3390/en16227660.

This Python code integrates an evolutionary algorithm with an LSTM neural 
network to optimize energy efficiency in smart manufacturing, considering 
smart grid data. The evolutionary algorithm, implemented using the DEAP 
library, optimizes the hyperparameters of the LSTM neural network (look_back 
window size, number of LSTM units, epochs, and batch size) to minimize the 
prediction error on the energy data. The neural network is built using 
TensorFlow/Keras and trained on synthetic energy data. The evolutionary 
algorithm evolves a population of individuals, each representing a set of 
hyperparameters, to find the best combination that optimizes the LSTM model's 
performance.
"""
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# Define the problem to optimize: smart grid energy efficiency in smart manufacturing
# This function represents the fitness function to optimize using the evolutionary algorithm
def fitness_function(individual, energy_data):
    # Normalize the energy data
    scaler = MinMaxScaler(feature_range=(0, 1))
    energy_data = scaler.fit_transform(energy_data)

    # Prepare the data for LSTM model
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    look_back = individual[0]
    train_data, test_data = np.split(energy_data, [int(0.8 * len(energy_data))])
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # LSTM model architecture
    model = Sequential()
    model.add(LSTM(individual[1], input_shape=(1, look_back)))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(trainX, trainY, epochs=individual[2], batch_size=individual[3], verbose=0)

    # Evaluate the model on the test data
    loss = model.evaluate(testX, testY, verbose=0)
    return loss,

# Initialize the DEAP framework for the evolutionary algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 100)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int, toolbox.attr_int, toolbox.attr_int, toolbox.attr_int), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=100, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function, energy_data=np.random.rand(1000, 1)) # Placeholder for energy data

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
