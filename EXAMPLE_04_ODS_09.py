# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:52:15 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 4 ODS 09
#==============================================================================
"""
Ghoreishi and Happonen (2020) discuss how integrating AI techniques into 
product design in the circular economy improves sustainability and optimizes 
manufacturing production. They maintain that designing circular products, 
supported by academic guides and strategies and using AI for data analysis and 
real-time optimization, increases productivity. Their study, based on a 
qualitative methodology, proposes a framework for the essential role of AI in 
this design, highlighting its positive impact. 

Ghoreishi, M., & Happonen, A. (2020). Key enablers for deploying artificial 
intelligence for circular economy embracing sustainable product design: Three 
case studies. AIP Conference Proceedings. https://doi.org/10.1063/5.0001339.

This Python code integrates an evolutionary algorithm with a combination of 
PCA, K-Means clustering, and a neural network to optimize sustainable product 
design in the circular economy. The evolutionary algorithm, implemented using 
the DEAP library, optimizes the hyperparameters of the PCA (number of 
components), K-Means clustering (number of clusters), and the neural network 
(epochs and batch size) to minimize the loss on the product data. The neural 
network is built using TensorFlow/Keras and trained on synthetic product data. 
The evolutionary algorithm evolves a population of individuals, each 
representing a set of hyperparameters, to find the best combination that 
optimizes the overall framework for sustainable product design.
"""

import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the problem to optimize: sustainable product design in the circular economy
# This function represents the fitness function to optimize using the evolutionary algorithm
def fitness_function(individual, product_data):
    # Standardize the product data
    scaler = StandardScaler()
    product_data_scaled = scaler.fit_transform(product_data)

    # Principal Component Analysis (PCA) for dimensionality reduction
    pca = PCA(n_components=individual[0])
    product_data_pca = pca.fit_transform(product_data_scaled)

    # K-Means clustering to identify sustainable design clusters
    kmeans = KMeans(n_clusters=individual[1])
    kmeans.fit(product_data_pca)
    labels = kmeans.labels_

    # Neural network model architecture for real-time optimization
    model = Sequential()
    model.add(Dense(64, input_dim=product_data_pca.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Split the product data into training and testing sets
    train_data, test_data = np.split(product_data_pca, [int(0.8 * len(product_data_pca))])
    train_labels, test_labels = np.split(labels, [int(0.8 * len(labels))])

    # Train the model on the training data
    model.fit(train_data, train_labels, epochs=individual[2], batch_size=individual[3], verbose=0)

    # Evaluate the model on the test data
    loss = model.evaluate(test_data, test_labels, verbose=0)
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
toolbox.register("evaluate", fitness_function, product_data=np.random.rand(1000, 20)) # Placeholder for product data

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
