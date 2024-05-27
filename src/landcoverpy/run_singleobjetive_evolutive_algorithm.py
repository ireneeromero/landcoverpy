from os.path import join
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.evaluator import MultiprocessEvaluator
from sklearn.preprocessing import LabelEncoder 
from jmetal.util.observer import PrintObjectivesObserver, PlotFrontToFileObserver, WriteFrontToFileObserver
from jmetal.lab.visualization import InteractivePlot, Plot
from landcoverpy.so_evolutive_algorithm import NeuralNetworkOptimizer
from landcoverpy.minio_func import MinioConnection
from landcoverpy.config import settings
from landcoverpy.utilities.confusion_matrix import compute_confusion_matrix
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from jmetal.util.observer import Observer


class EvaluationObserver(Observer):
    def __init__(self, output_directory: str, step: int = 1) -> None:
        """Observer that saves the best evaluation for each generation to a file.

        :param output_directory: Output directory.
        :param step: Frequency of recording the evaluations.
        """
        self.directory = output_directory
        self.step = step
        self.counter = 0
        self.best_evaluations = []
        self.all_evaluations = []

        # Create directory if it does not exist
        if Path(self.directory).is_dir():
            print(f"Directory {self.directory} exists. Removing contents.")
            for file in os.listdir(self.directory):
                os.remove(f"{self.directory}/{file}")
        else:
            print(f"Directory {self.directory} does not exist. Creating it.")
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        solutions = kwargs["SOLUTIONS"]

        if solutions is not None:
            for solution in solutions:
                self.all_evaluations.append(solution.objectives[0])
            self.counter += 1

            print(f"Step {self.counter}: {solutions}")

            # Save evaluations to file at specified intervals
            if self.counter % self.step == 0:
                self.save_to_file()

    def save_to_file(self) -> None:
        file_path = os.path.join(self.directory, f"evaluations_{self.counter}.txt")
        with open(file_path, 'w') as f:
            for eval in self.best_evaluations:
                f.write(f"{-1*eval}\n")
        print(f"Saved evaluations to {file_path}")

    def plot_evaluations(self) -> None:
        abs_evaluations = [abs(eval) for eval in self.all_evaluations]
        x_labels = list(range(len(abs_evaluations)))
        plt.plot(x_labels, abs_evaluations)
        plt.xlabel('Evaluation')
        plt.ylabel('Fitness')
        plt.title('Evaluation per Generation')
        plt.savefig(os.path.join(self.directory, 'evaluation_plot.png'))
        plt.show()

if __name__ == "__main__":

    X_train_dataset = "x_train.csv"
    X_train_dataset_path = join(settings.TMP_DIR, X_train_dataset)

    X_test_dataset = "x_test.csv"
    X_test_dataset_path = join(settings.TMP_DIR, X_test_dataset)

    y_train_dataset = "y_train.csv"
    y_train_dataset_path = join(settings.TMP_DIR, y_train_dataset)

    y_test_dataset = "y_test.csv"
    y_test_dataset_path = join(settings.TMP_DIR, y_test_dataset)

    minio_client = MinioConnection()


    minio_client.fget_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME +'/train-test', X_train_dataset),
        file_path=X_train_dataset_path,
    )

    minio_client.fget_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME +'/train-test', X_test_dataset),
        file_path=X_test_dataset_path,
    )

    minio_client.fget_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME +'/train-test', y_train_dataset),
        file_path=y_train_dataset_path,
    )

    minio_client.fget_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME +'/train-test', y_test_dataset),
        file_path=y_test_dataset_path,
    )

    X_train = pd.read_csv(X_train_dataset_path)
    X_test = pd.read_csv(X_test_dataset_path)
    y_train = pd.read_csv(y_train_dataset_path)
    y_test = pd.read_csv(y_test_dataset_path)

    y_train = y_train['class']
    y_test = y_test['class']
   


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

    y_train_mapped = [mapping[label] for label in y_train]

    label_encoder_train = LabelEncoder()
    y_train = label_encoder_train.fit_transform(y_train_mapped)


    problem = NeuralNetworkOptimizer(X_train, X_test, y_train, y_test)
    print("problem.number_of_variables", problem.number_of_variables())

    max_evaluations = 100
    
    output_directory = "output_directory_path"  # Define la ruta de tu directorio de salida
    evaluation_observer = EvaluationObserver(output_directory=output_directory, step=1)



    algorithm = GeneticAlgorithm(
        population_evaluator=MultiprocessEvaluator(8),
        problem=problem,
        population_size=10,
        offspring_population_size=10,
        mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
         crossover=IntegerSBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    

    
    algorithm.observable.register(evaluation_observer)
    
    algorithm.run()
    result = algorithm.get_result()


    # Save results to file
    print_function_values_to_file(result, "FUN." + algorithm.label)
    print_variables_to_file(result, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print("Solution: " + str(result.variables[0]))
    print("Fitness:  " + str(result.objectives[0]))
    print("Computing time: " + str(algorithm.total_computing_time))

    evaluation_observer.save_to_file()
    evaluation_observer.plot_evaluations()


  