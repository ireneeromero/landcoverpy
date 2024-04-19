import random
import numpy as np
from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class NeuralNetworkOptimizer(BinaryProblem):
    def __init__(self, dataset, max_layers=5, max_neurons_per_layer=100, max_learning_rate=0.1):
        super(NeuralNetworkOptimizer, self).__init__()
        self.dataset = dataset
        self.max_layers = max_layers
        self.max_neurons_per_layer = max_neurons_per_layer
        self.max_learning_rate = max_learning_rate
        self.number_of_bits = max_layers * int(np.ceil(np.log2(max_neurons_per_layer))) + int(np.ceil(np.log2(int(1 / max_learning_rate * 100))))
        self.obj_directions = [self.MINIMIZE]

    def number_of_variables(self) -> int:
        return 1

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        structure = solution.variables[0]
        # Aquí interpretamos la estructura binaria para configurar la red
        model = self.build_model(structure)
        loss = self.train_and_evaluate(model)
        solution.objectives[0] = loss
        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(
            number_of_variables=self.number_of_variables(),
            number_of_objectives=self.number_of_objectives()
        )
        new_solution.variables[0] = [random.randint(0, 1) for _ in range(self.number_of_bits)]
        return new_solution

    def name(self):
        return "NeuralNetworkOptimizer"

    def build_model(self, structure):
        model = Sequential()
        layer_bits = int(np.ceil(np.log2(self.max_neurons_per_layer)))
        lr_bits = int(np.ceil(np.log2(int(1 / self.max_learning_rate * 100))))
        
        neurons_per_layer = [int(''.join(map(str, structure[i * layer_bits:(i + 1) * layer_bits])), 2) for i in range(self.max_layers)]
        learning_rate = int(''.join(map(str, structure[-lr_bits:])), 2) / 100.0
        
        for neurons in neurons_per_layer:
            if neurons > 0:
                model.add(Dense(neurons, activation='relu'))
        model.add(Dense(self.dataset.num_classes, activation='softmax'))  # Assuming classification problem
        model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate(self, model):
        x_train, y_train, x_val, y_val = self.dataset.get_data()
        model.fit(x_train, y_train, epochs=10, verbose=0)
        loss, _ = model.evaluate(x_val, y_val, verbose=0)
        return loss
