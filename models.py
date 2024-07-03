import os
import pandas as pd
from keras._tf_keras.keras.applications import VGG16


def load_data(csv_path, required_columns):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File '{csv_path}' not found.")

    data = pd.read_csv(csv_path)
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the CSV file.")

    return data


# Load VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
