import cProfile
import pstats

def run_dog_breed_classification():
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import tensorflow_hub as hub
    from sklearn.model_selection import train_test_split
    import os
    import datetime

    # Define constants
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
    MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5'

    # File paths
    train_path = "./dog-breed-identification/train/"
    labels_csv_path = "./labels.csv"

    # Read labels CSV
    labels_csv = pd.read_csv(labels_csv_path)
    print(labels_csv.head())

    # Get filenames and labels
    filenames = [train_path + fname + ".jpg" for fname in labels_csv['id']]
    labels = labels_csv['breed'].to_numpy()
    unique_breeds = np.unique(labels)

    # Create boolean labels
    unique_breeds = np.unique(labels)
    boolean_labels = [label == unique_breeds for label in labels]

    # Create train and validation sets
    NUM_IMAGES = 1000
    OUTPUT_SHAPE = len(unique_breeds)

    x_train, x_val, y_train, y_val = train_test_split(filenames[:NUM_IMAGES], boolean_labels[:NUM_IMAGES], test_size=0.2, random_state=19)

    # Function to preprocess images
    def process_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
        return image

    def get_image_label(image_path, label):
        """ Takes the image path,label and converts into a tuple with image as a tensor and its label. Returns a tensor tuple."""
        image = process_image(image_path)
        return image, label

    # Create data batches
    def create_data_batches(x, y=None, batch_size=BATCH_SIZE, test_data=False, valid_data=False):
        if test_data:
            data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
            data_batch = data.map(process_image).batch(BATCH_SIZE)
            return data_batch
        elif valid_data:
            data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
            data_batch = data.map(get_image_label).batch(BATCH_SIZE)
            return data_batch
        else:
            data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))
            data = data.shuffle(buffer_size=len(x))
            data_batch = data.map(get_image_label).batch(BATCH_SIZE)
            return data_batch

    # Function to create model
    def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
        model = tf.keras.Sequential([
            hub.KerasLayer(model_url),
            tf.keras.layers.Dense(units=output_shape, activation='softmax')
        ])
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )
        model.build(input_shape)
        return model

    def create_tensorboard_callback():
        logdir = os.path.join('./dog-breed-identification/logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        return tf.keras.callbacks.TensorBoard(logdir)

    train_data = create_data_batches(x_train, y_train)
    val_data = create_data_batches(x_val, y_val, valid_data=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

    # Function to train the model
    def train_model():
        model = create_model()
        tensorboard = create_tensorboard_callback()
        model.fit(x=train_data, epochs=NUM_EPOCHS, validation_data=val_data, validation_freq=1,
                  callbacks=[tensorboard, early_stopping])
        return model

    # Train the model
    model = train_model()

    # Create TensorBoard callback

    # Function to unbatchify data
    def unbatchify(data):
        labels_ = []
        images_ = []

        for image, label in data.unbatch().as_numpy_iterator():
            images_.append(image)
            labels_.append(label)
        return images_, labels_

    # Evaluate the model
    model.evaluate(val_data)

    # Save and load the model
    def save_model(model, suffix=None):
        model_dir = os.path.join('./dog-breed-identification/models', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        model_path = model_dir + '-' + suffix + '.h5'
        print(f"Model is saving to {model_path}")
        model.save(model_path)
        return model_path

    def load_model(model_path):
        print(f"Loading a saved model from the path {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
        return model

    # Save and load the model
    save_model(model, suffix="1000-images-mobilenetv2-Adam")
    load_1000_images_model = load_model("./dog-breed-identification/models/20240129-145518-1000-images-mobilenetv2-Adam.h5")

    # Evaluate the loaded model
    load_1000_images_model.evaluate(val_data)

    # Full model training
    full_data = create_data_batches(filenames, boolean_labels)
    full_model = create_model()
    full_model_tensorboard = create_tensorboard_callback()
    full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
    full_model.fit(x=full_data, epochs=NUM_EPOCHS, callbacks=[full_model_tensorboard, full_model_early_stopping])

    # Predictions on test data
    test_path = "./dog-breed-identification/test/"
    test_filenames = [test_path + fname for fname in os.listdir(test_path)]
    test_data = create_data_batches(test_filenames, test_data=True)
    test_predictions = full_model.predict(test_data)

    print("Evaluate loaded model on validation data:")
    load_1000_images_model.evaluate(val_data)

    # Evaluate the full model on validation data
    print("Evaluate full model on validation data:")
    full_model.evaluate(val_data)

    # Print additional metrics or information
    # For example, you can use the following code to get predictions and accuracy
    def get_predictions_and_accuracy(model, data):
        predictions = model.predict(data)
        accuracy = model.evaluate(data)[-1]
        return predictions, accuracy

    # Get predictions and accuracy for the loaded model
    loaded_model_predictions, loaded_model_accuracy = get_predictions_and_accuracy(load_1000_images_model, val_data)
    print("Loaded model accuracy:", loaded_model_accuracy)

    # Get predictions and accuracy for the full model
    full_model_predictions, full_model_accuracy = get_predictions_and_accuracy(full_model, val_data)
    print("Full model accuracy:", full_model_accuracy)

    # Save predictions to CSV
    test_ids = [os.path.splitext(path)[0] for path in os.listdir(test_path)]
    preds_df = pd.DataFrame(columns=['id'] + list(unique_breeds))
    preds_df['id'] = test_ids
    preds_df[list(unique_breeds)] = test_predictions
    preds_df.to_csv("./dog-breed-identification/full_model_submission1.csv", index=False)

# Use cProfile to profile the function
profile_filename = 'dog_breed_classification_profile.prof'
cProfile.run('run_dog_breed_classification()', profile_filename)

# Analyze the profiling results using pstats
stats = pstats.Stats(profile_filename)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats(10)
