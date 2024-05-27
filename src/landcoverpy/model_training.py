import json
from os.path import join

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from landcoverpy.config import settings
from landcoverpy.minio_func import MinioConnection
from landcoverpy.utilities.confusion_matrix import compute_confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder


def _feature_reduction(
    df_x: pd.DataFrame, df_y: pd.DataFrame, percentage_columns: int = 100
):
    """Feature reduction method. Receives the training dataset and returns a set of variables."""

    if percentage_columns < 100:
        n_columns = len(df_x.columns.tolist())
        n_features = int(n_columns * percentage_columns / 100)
        model = LogisticRegression(
            penalty="elasticnet", max_iter=10000, solver="saga", n_jobs=-1, l1_ratio=0.5
        )
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        fit = rfe.fit(df_x, df_y)
        used_columns = df_x.columns[fit.support_].tolist()
    else:
        used_columns = df_x.columns.tolist()

    return used_columns


def train_model_land_cover(land_cover_dataset: str, n_jobs: int = 2):
    """Trains a Random Forest model using a land cover dataset."""

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

    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    X_train = X_train.reindex(columns=used_columns)
    print(X_train)
    clf.fit(X_train, y_train)
    y_true = clf.predict(X_test)
    print("y_true",y_true)

    labels = y_train_data.unique()

    confusion_image_filename = "confusion_matrix_RF.png"
    out_image_path = join(settings.TMP_DIR, confusion_image_filename)
    compute_confusion_matrix(y_true, y_test, labels, out_image_path=out_image_path)

    minio_folder = settings.LAND_COVER_MODEL_FOLDER

    # Save confusion matrix image to minio
    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, minio_folder, confusion_image_filename),
        file_path=out_image_path,
        content_type="image/png",
    )

    model_name = "model_RF.joblib"
    model_path = join(settings.TMP_DIR, model_name)
    joblib.dump(clf, model_path)

    # Save model to minio
    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_name}",
        file_path=model_path,
        content_type="mlmodel/randomforest",
    )

    model_metadata = {
        "model": str(type(clf)),
        "n_jobs": n_jobs,
        "used_columns": list(used_columns),
        "classes": list(labels)
    }

    model_metadata_name = "metadata_RF.json"
    model_metadata_path = join(settings.TMP_DIR, model_metadata_name)

    with open(model_metadata_path, "w") as f:
        json.dump(model_metadata, f)

    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_metadata_name}",
        file_path=model_metadata_path,
        content_type="text/json",
    )

def train_dnn_model_land_cover(land_cover_dataset: str, n_jobs: int = 2):
    """Trains a Random Forest model using a land cover dataset."""

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

    y_combined = pd.concat([y_train, y_test], ignore_index=True)
   

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
 
  #2 23 0 21 1 3 0 1 
  # 3 49 2 2 2 3 0 0 
  # 2, 61, 2, 76, 1, 3, 0, 1
  # 1 54 2 71 2 0 0 0 
  # 3 53 1 6 0 3 0 1 

  # 2 60 2 15 1 3 1 1 
    n_layers = 2
    n_neurons = 60
    activation_index = 2
    learning_rate = 15/1000
    optimization = 1
    regularization_index = 3
    weight_initialization = 1
    dropout = 1


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

    # Train model
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
        
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1,)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    y_pred_encoded = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_encoded, axis=1)

    # Reverse the encoding process using the reverse label encoder
    y_pred = label_encoder_train.inverse_transform(y_pred_indices)
    # Get y_pred label names
    y_true = [list(mapping.keys())[list(mapping.values()).index(idx)] for idx in y_pred]
   
    accuracy = accuracy_score(y_test, y_true)
    print("accuracy", accuracy) 
    labels = y_combined.unique()

    confusion_image_filename = "confusion_matrix.png"
    out_image_path = join(settings.TMP_DIR, confusion_image_filename)
    compute_confusion_matrix(y_true, y_test, labels, out_image_path=out_image_path)

    minio_folder = settings.LAND_COVER_MODEL_FOLDER

    # Save confusion matrix image to minio
    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=join(settings.MINIO_DATA_FOLDER_NAME, minio_folder, confusion_image_filename),
        file_path=out_image_path,
        content_type="image/png",
    )

    # model_name = "model.h5"
    # model_path = join(settings.TMP_DIR, model_name)
    # model.save(model_path)

    # # Save model to minio
    # minio_client.fput_object(
    #     bucket_name=settings.MINIO_BUCKET_MODELS,
    #     object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_name}",
    #     file_path=model_path,
    #     content_type="mlmodel/dnn",
    # )



    # model_metadata = {
    #     "model": str(type(model)),
    #     "n_jobs": n_jobs,
    #     "used_columns": list(used_columns),
    #     "classes": list(labels)
    # }

    # model_metadata_name = "metadata.json"
    # model_metadata_path = join(settings.TMP_DIR, model_metadata_name)

    # with open(model_metadata_path, "w") as f:
    #     json.dump(model_metadata, f)

    # minio_client.fput_object(
    #     bucket_name=settings.MINIO_BUCKET_MODELS,
    #     object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_metadata_name}",
    #     file_path=model_metadata_path,
    #     content_type="text/json",
    # )

