import random
import numpy as np
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score




class NeuralNetworkOptimizer(IntegerProblem):
    def __init__(self, X_train, X_test, y_train, y_test, number_of_variables: int = 7):
        super(NeuralNetworkOptimizer, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        """

        estructura:
        x1 = número de capas: [0, 5]
        x2 = número de nueronas por capa: [12, 64]
        x3 = funcion de activación: '0: Relu', '1: sigmoid', '2: tanh'

        hiperparametros: 
        x4 = learning rate [1e-4, 0.1]
        x5 = Optimización: ' 0: Adam, 1: SGD, 2: RMSprop'
        x6 = Regularización: 0: L1, 1: L2, 2:L1L2, 3: None
        x7 = Inicialización de pesos: 0: 'He', 1: 'Glorot', 2: 'uniform'
        x8 = Dropout: 0: Yes, 1: No
    
        """
        self.lower_bound = [1, 12, 0, 1, 0, 0, 0, 0]
        self.upper_bound = [5, 64, 2, 100, 2, 3, 2, 1] 

        
        self.obj_directions = [self.MAXIMIZE]
        self.obj_labels = ["Accuracy"]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def number_of_variables(self) -> int:
        return len(self.lower_bound)

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:

        n_layers = solution.variables[0]
        n_neurons = solution.variables[1]
        activation_index = solution.variables[2]
        learning_rate = solution.variables[3]/1000
        optimization = solution.variables[4]
        regularization_index = solution.variables[5]
        weight_initialization = solution.variables[6]
        dropout = solution.variables[7]

        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test


        # Inicialización de pesos
        if weight_initialization == 0:
            initializer = HeNormal()
        elif weight_initialization == 1:
            initializer = GlorotUniform()
        else:
            initializer = 'uniform' 

        # Función de activación
        activation_functions = {
            0: 'relu',
            1: 'sigmoid',
            2: 'tanh'
        }
        activation_func = activation_functions[activation_index]

        # Regularización
        regularization_functions = {
            0: l1(0.01),  # L1 regularization with a regularization factor of 0.01
            1: l2(0.01),  # L2 regularization with a regularization factor of 0.01,
            2: l1_l2(l1=0.01, l2=0.01),
            3: 'None'
        }
        regularization = regularization_functions[regularization_index]

        #Optimizador

        if optimization == 0:
            optimizer = Adam(learning_rate=learning_rate)
        elif optimization == 1:
            optimizer = SGD(learning_rate=learning_rate)
        elif optimization == 2:
            optimizer = RMSprop(learning_rate=learning_rate)
        
        print("Optimizer", optimizer)
        print("regularization", regularization)
        print("activation_func", activation_func)
        print("initializer", initializer)
        print("n_layer", n_layers)
        print("n_neurons", n_neurons)
        print("learning_rate", learning_rate)
        print("Vbujkfdvgnlxdñ")

        
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
    
        for _ in range(n_layers):
            if weight_initialization == 0:
                initializer = HeNormal()
            elif weight_initialization == 1:
                initializer = GlorotUniform()
            else:
                initializer = 'uniform' 
            if regularization == 'None':
                model.add(Dense(n_neurons, activation=activation_func, kernel_initializer=initializer))
            else:
                model.add(Dense(n_neurons, activation=activation_func, kernel_initializer=initializer, kernel_regularizer=regularization))
            if dropout == 0:
                model.add(Dropout(0.2)) 

       
        # Capa de salida
        model.add(Dense(9, activation='softmax'))
        
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1,)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        mapping = {
            "builtUp": 1,
            "herbaceousVegetation": 2,
            "shrubland": 3,
            "water": 4,
            "wetland": 5,
            "cropland": 6,
            "closedForest": 7,
            "openForest": 8,
            "bareSoil": 9
            }


        y_pred_encoded = model.predict(X_test)
        
        y_pred = np.array([np.argmax(pred) + 1 for pred in y_pred_encoded])
        y_pred = [list(mapping.keys())[list(mapping.values()).index(idx)] for idx in y_pred]
        
        accuracy = accuracy_score(y_test, y_pred)
        print("metrica", accuracy)
        print("SDFGHJKILO")
        print("vhjklñ")
        
        solution.objectives[0] = -1*accuracy

        return solution


    def name(self):
        return "NeuralNetworkOptimizer"


