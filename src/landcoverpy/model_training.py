
from os.path import join

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from landcoverpy.config import settings
from landcoverpy.minio_func import MinioConnection
from landcoverpy.utilities.confusion_matrix import compute_confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l1, l2, l1_l2
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
   
    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    print(X_train)
    clf.fit(X_train, y_train)
    y_true = clf.predict(X_test)
    print("y_true",y_true)

    labels = y_combined.unique()

    confusion_image_filename = "confusion_matrix_RF_new_v2.png"
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

    model_name = "model_RF_new_v2.joblib"
    model_path = join(settings.TMP_DIR, model_name)
    joblib.dump(clf, model_path)

    # Save model to minio
    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
        object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_name}",
        file_path=model_path,
        content_type="mlmodel/randomforest",
    )

    

def train_dnn_model_land_cover(model_to_use: str = "base", n_jobs: int = 2):
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
 
    #Set this parameters to the individual values

    n_layers = 2
    n_neurons = 122
    activation_index = 1
    learning_rate = 4/1000
    optimization = 0
    regularization_index = 3
    weight_initialization = 0
    dropout = 0
    dropout_value = 1/10


    initializer_functions = {
            0: 'he_normal',
            1: 'glorot_uniform',
            2: 'random_uniform'
        }
    initializer = initializer_functions[weight_initialization]



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



    if model_to_use == "base":
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(9, activation='softmax'))

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',  # Usar 'sparse_categorical_crossentropy' si las etiquetas son enteros
                metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.15)

    else:

        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        
        for _ in range(n_layers):
                
            if regularization == 'None':
                model.add(Dense(n_neurons, activation=activation_func, kernel_initializer=initializer))
            else:
                model.add(Dense(n_neurons, activation=activation_func, kernel_initializer=initializer, kernel_regularizer=regularization))
            if dropout == 0:
                model.add(Dropout(dropout_value)) 

                    
        #Capa de salida
        model.add(Dense(9, activation='softmax'))
            
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, mode='min', verbose=1,)
        model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.15, callbacks=[early_stopping])
    

    y_pred_encoded = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_encoded, axis=1)

    # Reverse the encoding process using the reverse label encoder
    y_pred = label_encoder_train.inverse_transform(y_pred_indices)
    # Get y_pred label names
    y_true = [list(mapping.keys())[list(mapping.values()).index(idx)] for idx in y_pred]
   
    accuracy = accuracy_score(y_test, y_true)
    print("accuracy", accuracy) 
    labels = y_combined.unique()

    confusion_image_filename = "confusion_matrix_test.png"
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

    model_name = "model_base_test.h5"
    model_path = join(settings.TMP_DIR, model_name)
    model.save(model_path)

    # # Save model to minio
    minio_client.fput_object(
        bucket_name=settings.MINIO_BUCKET_MODELS,
         object_name=f"{settings.MINIO_DATA_FOLDER_NAME}/{minio_folder}/{model_name}",
         file_path=model_path,
         content_type="mlmodel/dnn",
     )



    
