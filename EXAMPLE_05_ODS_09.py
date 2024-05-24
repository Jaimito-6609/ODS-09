# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:57:50 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 5 ODS 09
#==============================================================================
"""
Lee (2023) examines how intelligent control theory, including fuzzy logic, 
neural networks, and genetic algorithms, drives optimization in smart 
manufacturing. The review highlights these technologies as crucial to 
improving equipment, operations and controls, highlighting their relevance to 
the advancement of Industry 4.0 focused on data, automation and AI. It 
emphasizes the growing importance of these methodologies, evidenced by an 
increase in publications and citations, and their transformative impact on 
advanced manufacturing.

Lee M-FR. A Review on Intelligent Control Theory and Applications in Process 
Optimization and Smart Manufacturing. Processes. 2023; 11(11):3171. 
https://doi.org/10.3390/pr11113171.

This Python code integrates fuzzy logic, neural networks, and genetic 
algorithms to optimize smart manufacturing processes. The evolutionary 
algorithm, implemented using the DEAP library, optimizes the hyperparameters 
of the neural network (epochs and batch size) to minimize the loss on the 
manufacturing data. The fuzzy logic control system is designed to evaluate 
the quality and speed of manufacturing processes. The neural network, built 
using TensorFlow/Keras, is trained on synthetic manufacturing data and the 
outputs of the fuzzy logic system. The evolutionary algorithm evolves a 
population of individuals, each representing a set of hyperparameters, to 
find the best combination that optimizes the overall intelligent control 
system for smart manufacturing.
"""

import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# Define the problem to optimize: smart manufacturing processes using intelligent control theory
# This function represents the fitness function to optimize using the evolutionary algorithm
def fitness_function(individual, manufacturing_data):
    # Normalize the manufacturing data
    scaler = MinMaxScaler(feature_range=(0, 1))
    manufacturing_data_scaled = scaler.fit_transform(manufacturing_data)
    
    # Fuzzy logic control system design
    quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
    speed = ctrl.Antecedent(np.arange(0, 11, 1), 'speed')
    control_output = ctrl.Consequent(np.arange(0, 11, 1), 'control_output')

    quality.automf(3)
    speed.automf(3)

    control_output['low'] = fuzz.trimf(control_output.universe, [0, 0, 5])
    control_output['medium'] = fuzz.trimf(control_output.universe, [0, 5, 10])
    control_output['high'] = fuzz.trimf(control_output.universe, [5, 10, 10])

    rule1 = ctrl.Rule(quality['poor'] | speed['poor'], control_output['low'])
    rule2 = ctrl.Rule(quality['average'], control_output['medium'])
    rule3 = ctrl.Rule(quality['good'] | speed['good'], control_output['high'])

    control_system = ctrl.ControlSystem([rule1, rule2, rule3])
    control_simulation = ctrl.ControlSystemSimulation(control_system)

    # Evaluate fuzzy control system on scaled data
    def evaluate_fuzzy_system(data):
        control_values = []
        for i in range(len(data)):
            control_simulation.input['quality'] = data[i][0]
            control_simulation.input['speed'] = data[i][1]
            control_simulation.compute()
            control_values.append(control_simulation.output['control_output'])
        return np.array(control_values)

    fuzzy_output = evaluate_fuzzy_system(manufacturing_data_scaled)

    # Neural network model architecture for optimization
    model = Sequential()
    model.add(Dense(64, input_dim=manufacturing_data_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Split the manufacturing data into training and testing sets
    train_data, test_data = np.split(manufacturing_data_scaled, [int(0.8 * len(manufacturing_data_scaled))])
    train_labels, test_labels = np.split(fuzzy_output, [int(0.8 * len(fuzzy_output))])

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
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int, toolbox.attr_int), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=100, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function, manufacturing_data=np.random.rand(1000, 2)) # Placeholder for manufacturing data

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
