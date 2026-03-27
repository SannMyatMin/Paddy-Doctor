import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class PaddyDiseaseClassifierModel:
    def __init__(self, disease_encoder, variety_encoder, max_age, img_size=224):
        self.img_size = img_size
        self.model = None
        self.disease_encoder = disease_encoder
        self.variety_encoder = variety_encoder
        self.max_age = max_age

    def build_model(self, fine_tune_at=100):
        # Base Model Load လုပ်ခြင်း
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )

        # Fine-tuning: Layer အားလုံးကို အရင်ဖွင့်ပါ
        base_model.trainable = True
        
        # သတ်မှတ်ထားတဲ့ layer အထိပဲ ပြန်ပိတ်ပါ (အစောပိုင်း layer တွေက general features တွေမို့ပါ)
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1)
        ])

        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
        
        x = data_augmentation(inputs)
        x = base_model(x, training=True) # Fine-tuning အတွက် True ထားပါ
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x) # Capacity နည်းနည်းတိုးလိုက်ပါတယ်
        x = layers.Dropout(0.4)(x) # Overfitting ကာကွယ်ရန်

        # Outputs
        disease_out = layers.Dense(len(self.disease_encoder.classes_), activation='softmax', name='disease')(x)
        variety_out = layers.Dense(len(self.variety_encoder.classes_), activation='softmax', name='variety')(x)
        age_out = layers.Dense(1, activation='linear', name='age')(x)

        self.model = models.Model(inputs=inputs, outputs=[disease_out, variety_out, age_out])

        # Learning Rate ကို Fine-tuning အတွက် အနိမ့်ထားပါ (1e-5)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss={
                'disease': 'sparse_categorical_crossentropy',
                'variety': 'sparse_categorical_crossentropy',
                'age': 'mse'
            },
            metrics={
                'disease': 'accuracy',
                'variety': 'accuracy',
                'age': 'mae'
            }
        )
        self.model.summary()

    def train(self, train_ds, val_ds, epochs=100):
        # Model မရှိသေးရင် အသစ်ဆောက်ပါ
        if self.model is None:
            print("Building new model...")
            self.build_model()

        # အကောင်းဆုံး Weight တွေကို အမြဲသိမ်းထားဖို့ Checkpoint ထည့်ပါမယ်
        checkpoint = ModelCheckpoint(
            "best_rice_model.keras", 
            monitor='val_disease_accuracy', 
            save_best_only=True, 
            mode='max'
        )

        # Patience ကို ၁၅ အထိ တိုးလိုက်ပါတယ်
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True
        )

        history = self.model.fit(
            train_ds, 
            validation_data=val_ds, 
            epochs=epochs, 
            callbacks=[early_stop, checkpoint]
        )
        return history
    
    def predict(self, img_path):
        if self.model is None:
            raise Exception("Model not loaded!")

        img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
        img = image.img_to_array(img)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        pred_disease, pred_variety, pred_age = self.model.predict(img)

        disease = self.disease_encoder.inverse_transform([np.argmax(pred_disease[0])])[0]
        variety = self.variety_encoder.inverse_transform([np.argmax(pred_variety[0])])[0]
        age = int(pred_age[0][0] * self.max_age)

        print("🌾 Prediction Result:")
        print("Disease:", disease)
        print("Variety:", variety)
        print("Age:", age)

        return {
            "disease": disease,
            "variety": variety,
            "age"    : age }