import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping


class PaddyDiseaseClassifierModel:

    def __init__(self, disease_encoder, variety_encoder, max_age, img_size=224):
        self.img_size        = img_size
        self.batch_size      = 32
        self.model           = None
        self.disease_encoder = disease_encoder
        self.variety_encoder = variety_encoder
        self.max_age         = max_age


    def build_model(self):
        base_model = MobileNetV2(input_shape=(self.img_size, self.img_size, 3), include_top=False,
            weights='imagenet')

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1)
        ])

        x = data_augmentation(base_model.input)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        disease_output = layers.Dense(len(self.disease_encoder.classes_), activation='softmax',
            name='disease')(x)

        variety_output = layers.Dense(len(self.variety_encoder.classes_),activation='softmax',
            name='variety')(x)

        age_output = layers.Dense(1, activation='linear', name='age')(x)

        self.model = models.Model(inputs=base_model.input,
                                  outputs=[disease_output, variety_output, age_output] )

        self.model.compile(
            optimizer='adam',
            loss={
                'disease': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                'variety': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                'age': 'mse'
            },
            metrics={
                'disease': 'accuracy',
                'variety': 'accuracy',
                'age': 'mae'
            }
        )
        print("Model built!")

    def train(self, x_t, y_d_t, y_v_t, y_a_t, x_v, y_d_v, y_v_v, y_a_v, epochs=60):
        self.build_model()
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            x_t,
            {
                'disease': y_d_t,
                'variety': y_v_t,
                'age'    : y_a_t
            },
            validation_data=(
                x_v,
                {
                    'disease': y_d_v,
                    'variety': y_v_v,
                    'age'    : y_a_v
                }
            ),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop]
        )

        print("Training completed!")
        return history

    def save(self, path="trained_model/multi_output_rice_model.keras"):
        self.model.save(path)
        print(f"Model saved at {path}")

    def predict(self, img_path):
        if self.model is None:
            raise Exception("Model not loaded or built!")

        img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred_disease, pred_variety, pred_age = self.model.predict(img)

        disease = self.disease_encoder.inverse_transform([np.argmax(pred_disease)])
        variety = self.variety_encoder.inverse_transform([np.argmax(pred_variety)])
        age = pred_age[0][0] * self.max_age

        print("🌾 Prediction Result:")
        print("Disease:", disease[0])
        print("Variety:", variety[0])
        print("Age (approx):", int(age))

        return {
            "disease": disease[0],
            "variety": variety[0],
            "age": int(age)
        }