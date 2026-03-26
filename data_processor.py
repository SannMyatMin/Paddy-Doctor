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

    def prepare_labels(self):
        df = pd.read_csv(self.csv)

        self.disease_encoder = LabelEncoder()
        df["disease_enc"]    = self.disease_encoder.fit_transform(df["label"])

        self.variety_encoder = LabelEncoder()
        df["variety_enc"]    = self.variety_encoder.fit_transform(df["variety"])

        self.max_age         = df["age"].max()
        df["age_normalized"] = df["age"] / self.max_age

        self.df = df

    def get_Ds(self, ds_dir):
        x_image, y_disease, y_variety, y_age = [], [], [], []

        for _, row in self.df.iterrows():
            img_path = os.path.join(ds_dir, row["label"], row["image_id"])

            if not os.path.exists(img_path):
                continue

            img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
            img = image.img_to_array(img) / 255.0

            x_image.append(img)
            y_disease.append(row["disease_enc"])
            y_variety.append(row["variety_enc"])
            y_age.append(row["age_normalized"])

        return (
            np.array(x_image),
            np.array(y_disease),
            np.array(y_variety),
            np.array(y_age) )
    
    def get_train_val_ds(self):
        self.prepare_labels()

        x_train, y_d_train, y_v_train, y_a_train = self.get_Ds(self.train_ds)
        x_val, y_d_val, y_v_val, y_a_val = self.get_Ds(self.val_ds)

        return (
            x_train, y_d_train, y_v_train, y_a_train,
            x_val, y_d_val, y_v_val, y_a_val,
            self.disease_encoder,
            self.variety_encoder,
            self.max_age )