import random
import numpy as np
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from landcoverpy.minio import MinioConnection
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.metrics import Precision, Recall

from sklearn.model_selection import train_test_split




class NeuralNetworkOptimizer(IntegerProblem):
    def __init__(self, X_train, X_test, y_train, y_test):
        super(NeuralNetworkOptimizer, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        """

        estructura:
        x1 = número de capas: [0, 10]
        x2 = número de nueronas por capa: [3, 64]
        x3 = funcion de activación: '0: Relu', '1: sigmoid', '2: tanh'

        hiperparametros: 
        x4 = learning rate [1e-4, 0.1]
        x5 = Optimización: ' 0: Adam, 1: SGD, 2: RMSprop'
        x6 = Regularización: 0: L1, 1: L2 y 2: dropout
        x7 = Inicialización de pesos: 0: 'He', 1: 'Glorot', 2: 'uniform'
        """
        self.lower_bound = [0, 3, 0, 1, 0, 0, 0]
        self.upper_bound = [10, 64, 2, 1000, 2, 2, 2] 

        
        self.obj_directions = [self.MAXIMIZE, self.MAXIMIZE, self.MAXIMIZE]
        self.obj_labels = ["Accuracy", "Precision", "Recall"]

    def number_of_objectives(self) -> int:
        return len(self.obj_directions)

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        n_layers = solution.variables[0]
        n_neurons = solution.variables[1]
        activation_index = solution.variables[2]
        learning_rate = solution.variables[3]/1000
        optimization = solution.variables[4]
        regularization_index = solution.variables[5]
        weight_initialization = solution.variables[6]


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
            1: l2(0.01),  # L2 regularization with a regularization factor of 0.01
            2: 'dropout'  # Dropout will be handled separately
        }
        regularization = regularization_functions[regularization_index]

        #Optimizador

        if optimization == 0:
            optimizer = Adam(learning_rate=learning_rate)
        elif optimization == 1:
            optimizer = SGD(learning_rate=learning_rate)
        elif optimization == 2:
            optimizer = RMSprop(learning_rate=learning_rate)

        
        model = Sequential()
        if regularization == 'dropout':
            model.add(Dense(n_neurons, activation=activation_func, kernel_initializer=initializer, input_shape=(input_shape,)))
        else:
            model.add(Dense(n_neurons, activation=activation_func, kernel_initializer=initializer, kernel_regularizer=regularization, input_shape=(input_shape,)))

        
        for _ in range(n_layers - 1):
            model.add(Dense(n_neurons, activation=activation_func, kernel_initializer=initializer))
            if regularization == 'dropout':
                model.add(Dropout(0.5)) 
        
        # Capa de salida
        model.add(Dense(9, activation='softmax'))
        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
        

        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        y_pred_encoded = model.predict(X_test)
        y_pred_indices = np.argmax(y_pred_encoded, axis=1)

        # Invertir el proceso de codificación utilizando el codificador de etiquetas inverso
        y_pred = label_encoder_train.inverse_transform(y_pred_indices)
        # Get y_pred label names
        y_true = [list(mapping.keys())[list(mapping.values()).index(idx)] for idx in y_pred]

        
        
        
        solution.objectives[0] = acc
        solution.objectives[1] = precision
        solution.objectives[2] = recall

        return solution


    def name(self):
        return "NeuralNetworkOptimizer"

 
    def train_and_evaluate(self, model):
        x_train, y_train, x_val, y_val = self.dataset.get_data()
        model.fit(x_train, y_train, epochs=10, verbose=0)
        loss, _ = model.evaluate(x_val, y_val, verbose=0)
        return loss
