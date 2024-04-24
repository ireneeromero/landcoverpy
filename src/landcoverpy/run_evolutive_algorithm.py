from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import IntegerPolynomialMutation, IntegerSBXCrossover
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

from sklearn.preprocessing import LabelEncoder 

from landcoverpy.evolutive_algorithm import NeuralNetworkOptimizer
from landcoverpy.config import settings
from landcoverpy.minio import MinioConnection
from landcoverpy.utilities.confusion_matrix import compute_confusion_matrix


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

    used_columns = _feature_reduction(x_train_data, y_train_data)
    
    unique_locations = df.drop_duplicates(subset=["latitude","longitude"])
    unique_locations = unique_locations[['latitude', 'longitude']]

    unique_locations = unique_locations.sample(frac=1).reset_index(drop=True)

    train_size = 0.85

    split_index = int(len(unique_locations) * train_size)

    train_coordinates = unique_locations[:split_index]
    test_coordinates = unique_locations[split_index:]
    print("test_coordinates", test_coordinates)

    # Filter the coordinates for Andalusia.
    filtered_test_coordinates = test_coordinates[(test_coordinates['latitude'] >= 36.000192) & (test_coordinates['latitude'] <= 38.738181)]
    
    train_df = pd.merge(df, train_coordinates, on=['latitude', 'longitude'])
    test_df = pd.merge(df, filtered_test_coordinates, on=['latitude', 'longitude'])

    X_train = train_df[used_columns]
    X_test = test_df[used_columns]
    y_train = train_df['class']
    y_test = test_df['class']



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

    max_evaluations = 100

    algorithm = NSGAII(
        problem=problem,
        population_size=10,
        offspring_population_size=10,
        mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=0.20),
        crossover=IntegerSBXCrossover(probability=1.0, distribution_index=0.20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()
    front = algorithm.result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.get_name()}")
    print(f"Computing time: {algorithm.total_computing_time}")