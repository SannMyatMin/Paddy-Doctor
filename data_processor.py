import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class PaddyDiseaseDataProcessor:

    def __init__(self, dataset_dir="paddy_disease_classification", img_size=224, batch_size=16):
        self.dataset_dir = dataset_dir
        self.csv         = os.path.join(dataset_dir, "train.csv")
        self.train_ds    = os.path.join(dataset_dir, "dataset/train")
        self.val_ds      = os.path.join(dataset_dir, "dataset/val")
        self.img_size    = img_size
        self.batch_size  = batch_size

    def prepare_labels(self):
        df = pd.read_csv(self.csv)

        self.disease_encoder = LabelEncoder()
        df["disease_enc"]    = self.disease_encoder.fit_transform(df["label"])

        self.variety_encoder = LabelEncoder()
        df["variety_enc"]    = self.variety_encoder.fit_transform(df["variety"])

        self.max_age         = df["age"].max()
        df["age_normalized"] = df["age"] / self.max_age

        self.df = df

    def parse_image(self, img_path, disease, variety, age):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (self.img_size, self.img_size))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        return img, {
            "disease": disease,
            "variety": variety,
            "age"    : age }

    def build_dataset(self, ds_dir):
        image_paths = []
        diseases    = []
        varieties   = []
        ages        = []

        for _, row in self.df.iterrows():
            img_path = os.path.join(ds_dir, row["label"], row["image_id"])
            if os.path.exists(img_path):
                image_paths.append(img_path)
                diseases.append(row["disease_enc"])
                varieties.append(row["variety_enc"])
                ages.append(row["age_normalized"])

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, diseases, varieties, ages))
        dataset = dataset.map(lambda p, d, v, a: self.parse_image(p, d, v, a),
                              num_parallel_calls=tf.data.AUTOTUNE )
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def get_train_val_ds(self):
        self.prepare_labels()
        train_dataset = self.build_dataset(self.train_ds)
        val_dataset   = self.build_dataset(self.val_ds)

        return (train_dataset, val_dataset, self.disease_encoder, self.variety_encoder, self.max_age)