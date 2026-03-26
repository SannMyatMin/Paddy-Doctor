import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image

class PaddyDiseaseDataProcessor:
    def __init__(self, dataset_dir="paddy_disease_classification", img_size=224):
        self.dataset_dir = dataset_dir
        self.csv         = os.path.join(self.dataset_dir, "train.csv")
        self.train_ds    = os.path.join(self.dataset_dir, "dataset/train")
        self.val_ds      = os.path.join(self.dataset_dir, "dataset/val")
        self.img_size    = img_size

    def get_trainDs_valDs(self):
        df                   = pd.read_csv(self.csv)
        disease_encoder      = LabelEncoder()
        df["disease_enc"]    = disease_encoder.fit_transform(df["label"])

        variety_encoder      = LabelEncoder()
        df["variety_enc"]    = variety_encoder.fit_transform(df["variety"])

        max_age              = df["age"].max()
        df["age_normalized"] = df["age"] / max_age

        x_image     = []
        y_disease   = []
        y_variety   = []
        y_age       = []

        for _,row in df.itterrows():
            img_path = os.path.join(self.dataset_dir, row["label"], row["image_id"])
            if not os.path.exists(img_path):
                continue
            img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
            img = image.image_to_array(img) / 255.0

            x_image.append(img)
            y_disease.append(row["disease_enc"])
            y_variety.append(row["variety_enc"])
            y_age.append(row["age_normalized"])

        x_image   = np.array(x_image)
        y_disease = np.array(y_disease)
        y_variety = np.array(y_variety)
        y_age     = np.array(y_age)

        return x_image, y_disease, y_variety, y_age, disease_encoder, variety_encoder, max_age
