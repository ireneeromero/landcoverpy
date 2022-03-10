import pandas as pd
import sys
from config import settings
from utils import get_minio, safe_minio_execute

df = pd.read_csv(sys.argv[1])
print("Histogram of classes in input dataset\n",df["class"].value_counts())

# Sustituir clases
df['class'].replace(to_replace=["mixedBosque", "plantacion", "bosqueRibera"], value="bosque", inplace=True)
df['class'].replace(to_replace="pastos", value="dehesas", inplace=True)

# Reducir tamaño datos bosque
df_bosque = df[df["class"]=="bosque"]
df_not_bosque = df[df["class"]!="bosque"]
df_bosque = df_bosque.drop_duplicates(subset=["longitude","latitude"]).reset_index(drop=True)
df = pd.concat([df_bosque, df_not_bosque])

#Shuffle dataframe
df = df.sample(frac=1).reset_index(drop=True)
print("Histogram of classes in output dataset\n",df["class"].value_counts())


df.to_csv(sys.argv[2], index=False)
minio_client = get_minio()
safe_minio_execute(
        func = minio_client.fput_object,
        bucket_name = settings.MINIO_BUCKET_DATASETS,
        object_name =  f"{settings.MINIO_DATA_FOLDER_NAME}/{sys.argv[2]}",
        file_path=sys.argv[2],
        content_type="text/csv"
    )