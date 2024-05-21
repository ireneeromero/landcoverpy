from os.path import join
from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy
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



if __name__ == "__main__":

    land_cover_dataset = "dataset_postprocessed.csv"
    training_dataset_path = join(settings.TMP_DIR, land_cover_dataset)

    minio_client = MinioConnection()

    minio_client.fget_object(
        bucket_name=settings.MINIO_BUCKET_DATASETS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, land_cover_dataset),
        file_path=training_dataset_path,
    )

    df = pd.read_csv(training_dataset_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(np.nan)
    df = df.dropna()

    y_train_data = df["class"]
    x_train_data = df.drop(
        [
            "class",
            "latitude",
            "longitude",
            "spring_product_name",
            "autumn_product_name",
            "summer_product_name",
        ],
        axis=1,
    )

    used_columns = x_train_data.columns.tolist()
    
    unique_locations_with_class = df[['latitude', 'longitude', 'class']].drop_duplicates()
    train_dfs = []
    test_dfs = []
    for class_label in unique_locations_with_class['class'].unique():

        class_locations = unique_locations_with_class[unique_locations_with_class['class'] == class_label]
        class_locations = class_locations.sample(frac=1).reset_index(drop=True)
        
        split_point = int(len(class_locations) * 0.85)
        
        train_locations = class_locations.iloc[:split_point]
        test_locations = class_locations.iloc[split_point:]
        
        train_dfs.append(train_locations)
        test_dfs.append(test_locations)

    train_coordinates = pd.concat(train_dfs).reset_index(drop=True)
    test_coordinates = pd.concat(test_dfs).reset_index(drop=True)

    # Filter the coordinates for Andalusia.
    filtered_test_coordinates = test_coordinates[(test_coordinates['latitude'] >= 36.000192) & (test_coordinates['latitude'] <= 38.738181)]

    train_df = pd.merge(df, train_coordinates, on=['latitude', 'longitude', 'class'])
    test_df = pd.merge(df, test_coordinates, on=['latitude', 'longitude', 'class'])

    X_train = train_df[used_columns]
    X_test = test_df[used_columns]
    y_train = train_df['class']
    y_test = test_df['class']


    print(y_train.unique())
    print(y_test.unique())

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

    max_evaluations = 20000

    algorithm = EvolutionStrategy(
        population_evaluator=MultiprocessEvaluator(8),
        problem=problem,
        mu=100, #population_size
        lambda_=100, #offspring_population_size
        elitist=True,
        mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables()),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=PlotFrontToFileObserver(output_directory='single_directory', step=100))
    
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

  