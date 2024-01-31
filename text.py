import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imread
import os
import matplotlib.pyplot as plt
import datetime

# Load and preprocess data
labels_csv = pd.read_csv('./dog-breed-identification/labels.csv')
filenames = ["./dog-breed-identification/train/" + fname + ".jpg" for fname in labels_csv['id']]
labels = labels_csv['breed'].to_numpy()
unique_breeds = np.unique(labels)
boolean_labels = [label == unique_breeds for label in labels]

# Create data batches
NUM_IMAGES = 1000
x_train, x_val, y_train, y_val = train_test_split(filenames[:NUM_IMAGES], boolean_labels[:NUM_IMAGES], test_size=0.2, random_state=99)
IMG_SIZE = 224
BATCH_SIZE = 32

def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label

train_data = tf.data.Dataset.from_tensor_slices((tf.constant(x_train), tf.constant(y_train)))
train_data = train_data.shuffle(buffer_size=len(x_train)).map(get_image_label).batch(BATCH_SIZE)

val_data = tf.data.Dataset.from_tensor_slices((tf.constant(x_val), tf.constant(y_val)))
val_data = val_data.map(get_image_label).batch(BATCH_SIZE)

# Build and train the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = len(unique_breeds)
MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5'

def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url),
        tf.keras.layers.Dense(units=output_shape, activation='softmax')
    ])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.build(INPUT_SHAPE)
    return model

model = create_model()

# Create callbacks
def create_tensorboard_callback():
    logdir = os.path.join('./dog-breed-identification/logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return tf.keras.callbacks.TensorBoard(logdir)

tensorboard = create_tensorboard_callback()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

# Train the model
NUM_EPOCHS = 100
model.fit(x=train_data, epochs=NUM_EPOCHS, validation_data=val_data, validation_freq=1, callbacks=[tensorboard, early_stopping])

# Save and load the model
def save_model(model, suffix=None):
    model_dir = os.path.join('./dog-breed-identification/models', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_path = model_dir + '-' + suffix + '.h5'
    model.save(model_path)
    return model_path

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})

save_model(model, suffix='1000-images-mobilenetv2-Adam')
load_1000_images_model = load_model('./dog-breed-identification/models/20230127-201731-1000-images-mobilenetv2-Adam.h5')

# Evaluate the model
model.evaluate(val_data)
load_1000_images_model.evaluate(val_data)

# Make predictions on test data
test_path = './dog-breed-identification/test/'
test_filenames = [test_path + fname for fname in os.listdir(test_path)]
test_data = create_data_batches(test_filenames, test_data=True)
test_predictions = model.predict(test_data)
test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]

# Create submission file
preds_df = pd.DataFrame(columns=['id'] + list(unique_breeds))
preds_df['id'] = test_ids
preds_df[list(unique_breeds)] = test_predictions
preds_df.to_csv('./dog-breed-identification/full_model_submission1.csv', index=False)
