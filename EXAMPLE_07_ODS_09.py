# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:20:36 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 7 ODS 09
#==============================================================================

"""
Milošević and colleagues (2020) examine the use of AI in smart factories to 
optimize production preparation and prevent errors, aligning with Industry 4.0 
objectives of achieving flexible and adaptive production processes. They 
highlight how AI implementation maximizes resource utilization and continuous 
optimization, relying on cloud manufacturing and smart services, including 
natural metaheuristics, whose effectiveness is validated through experimental 
studies.

Milošević, M., Đurđev, M., Lukić, D., Antić, A., Ungureanu, N. (2020). 
Intelligent Process Planning for Smart Factory and Smart Manufacturing. In: 
Wang, L., Majstorovic, V., Mourtzis, D., Carpanzano, E., Moroni, G., 
Galantucci, L. (eds) Proceedings of 5th International Conference on the 
Industry 4.0 Model for Advanced Manufacturing. Lecture Notes in Mechanical 
Engineering. Springer, Cham. https://doi.org/10.1007/978-3-030-46212-3_14.

This Python code integrates LSTM neural networks with an evolutionary 
algorithm to optimize production preparation and prevent errors in smart 
factories, aligning with Industry 4.0 objectives. The evolutionary algorithm, 
implemented using the DEAP library, optimizes the hyperparameters of the LSTM 
neural network (look_back window size, number of LSTM units, epochs, and batch 
                size) to minimize the root mean square error (RMSE) on the 
production data. The LSTM neural network, built using TensorFlow/Keras, is 
trained on synthetic production data. The evolutionary algorithm evolves a 
population of individuals, each representing a set of hyperparameters, to 
find the best combination that optimizes the LSTM model's performance.
"""

import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Define the problem to optimize: production preparation and error prevention in smart factories
# This function represents the fitness function to optimize using the evolutionary algorithm
def fitness_function(individual, production_data):
    # Normalize the production data
    scaler = MinMaxScaler(feature_range=(0, 1))
    production_data_scaled = scaler.fit_transform(production_data)
    
    # Prepare the data for LSTM model
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    look_back = individual[0]
    train_data, test_data = np.split(production_data_scaled, [int(0.8 * len(production_data_scaled))])
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
    test_predict = model.predict(testX)
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform([testY])
    rmse = np.sqrt(mean_squared_error(testY[0], test_predict[:,0]))
    return rmse,

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
toolbox.register("evaluate", fitness_function, production_data=np.random.rand(1000, 1)) # Placeholder for production data

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
