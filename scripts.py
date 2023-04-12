import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import tensorflow as tf
import sys
from typing import Optional
import pandas as pd
import cv2


def process_dataset(input_folder: str,
                    output_folder: str) -> None:

    # Define the subfolder names for codes 1 to 11
    code_subfolder_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "ten thousand", "hundred million"]

    # Define the number of images to sample for each code in each dataset
    validation_sample_size = 70
    small_sample_size = 70
    medium_sample_size = 140
    large_sample_size = 210

    # Set the random seed
    random.seed(413)

    # Create the output folders if they don't exist
    for folder_name in ["small_training_dataset", "medium_training_dataset", "large_training_dataset", "validation_dataset"]:
        folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        # Create the subfolders for each code
        for code_name in code_subfolder_names:
            code_folder_path = os.path.join(folder_path, code_name)
            os.makedirs(code_folder_path, exist_ok=True)

    # Get a list of all the input image file paths
    image_file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".jpg")]

    # Shuffle the list of image file paths
    random.shuffle(image_file_paths)

    # Split the image file paths by code
    code_image_paths = {}
    for image_file_path in image_file_paths:
        code = os.path.splitext(os.path.basename(image_file_path))[0].split("_")[-1]
        code = int(code)
        if code not in code_image_paths:
            code_image_paths[code] = []
        code_image_paths[code].append(image_file_path)

    # Sample images for the validation dataset
    validation_image_paths = []
    for code, image_paths in code_image_paths.items():
        validation_image_paths.extend(random.sample(image_paths, validation_sample_size))

    # Move the sampled images to the validation dataset folder
    for image_path in validation_image_paths:
        code = os.path.splitext(os.path.basename(image_path))[0].split("_")[-1]
        code_folder_path = os.path.join(output_folder, "validation_dataset", code_subfolder_names[int(code)-1])
        shutil.copy(image_path, code_folder_path)

    # Sample images for the training datasets
    for dataset_name, sample_size in [("small_training_dataset", small_sample_size), ("medium_training_dataset", medium_sample_size), ("large_training_dataset", large_sample_size)]:
        for code, image_paths in code_image_paths.items():
            remaining_image_paths = [p for p in image_paths if p not in validation_image_paths]
            sampled_image_paths = random.sample(remaining_image_paths, sample_size)
            for image_path in sampled_image_paths:
                code = os.path.splitext(os.path.basename(image_path))[0].split("_")[-1]
                code_folder_path = os.path.join(output_folder, dataset_name, code_subfolder_names[int(code)-1])
                shutil.copy(image_path, code_folder_path)


"""
Add a classifier to a base model.
"""
def add_classifier(base_model: tf.keras.Model,
                   image_shape: tuple[int, int, int],
                   number_of_classes: int) -> tf.keras.Model:

    inputs = tf.keras.Input(shape=image_shape)
    return tf.keras.Model(inputs=inputs, outputs=tf.keras.layers.Dense(units=number_of_classes)(
                                                 tf.keras.layers.Dropout(0.5)(
                                                 tf.keras.layers.Dense(1024, activation="relu")(
                                                 tf.keras.layers.Flatten()(
                                                 base_model(inputs, training=False))))))


"""
Plot training and validation accuracy.
"""
def accuracies(history: tf.keras.callbacks.History,
               title: str,
               figsize: Optional[tuple[int, int]] = None) -> None:
    
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(training_accuracy, label='Training Accuracy')
    axis.plot(validation_accuracy, label='Validation Accuracy')
    axis.legend()
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Accuracy')
    axis.set_ylim(min(axis.get_ylim()), 1)
    figure.suptitle(title)


"""
Plot Training and validation loss.
"""
def losses(history: tf.keras.callbacks.History,
           title: str,
           figsize: Optional[tuple[int, int]] = None) -> None:

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(training_loss, label='Training Loss')
    axis.plot(validation_loss, label='Validation Loss')
    axis.legend()
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Cross Entropy')
    axis.set_ylim(0, max(axis.get_ylim()))
    figure.suptitle(title)


"""
Freeze a number of layers in the model.
"""
def freeze(model: tf.keras.Model,
           number_of_layers_to_freeze: int) -> None:
    
    for layer in model.layers[:number_of_layers_to_freeze]:
        layer.trainable = False


"""
Plot accuracies after fine-tuning.
"""
def accuracies_after_fine_tuning(classifier_history: tf.keras.callbacks.History,
                                 fine_tuning_history: tf.keras.callbacks.History,
                                 classifier_number_of_epoches: int,
                                 title: str,
                                 figsize: Optional[tuple[int, int]] = None) -> None:

    trianing_accuracy = classifier_history.history['accuracy'] + fine_tuning_history.history['accuracy']
    validation_accuracy = classifier_history.history['val_accuracy'] + fine_tuning_history.history['val_accuracy']

    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(trianing_accuracy, label='Training Accuracy')
    axis.plot(validation_accuracy, label='Validation Accuracy')
    axis.set_ylim(0.8, 1)
    axis.plot([classifier_number_of_epoches - 1, classifier_number_of_epoches - 1],
              axis.get_ylim(), label='Start Fine Tuning')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Accuracy')
    axis.legend()
    figure.suptitle(title)


"""
Plot losses after fine-tuning.
"""
def losses_after_fine_tuning(classifier_history: tf.keras.callbacks.History,
                             fine_tuning_history: tf.keras.callbacks.History,
                             classifier_number_of_epoches: int,
                             title: str,
                             figsize: Optional[tuple[int, int]] = None) -> None:

    trianing_loss = classifier_history.history['loss'] + fine_tuning_history.history['loss']
    validation_loss = classifier_history.history['val_loss'] + fine_tuning_history.history['val_loss']

    figure, axis = plt.subplots(figsize=figsize)

    axis.plot(trianing_loss, label='Training Loss')
    axis.plot(validation_loss, label='Validation Loss')
    axis.set_ylim(0, 1.0)
    axis.plot([classifier_number_of_epoches - 1, classifier_number_of_epoches - 1],
              axis.get_ylim(), label='Start Fine Tuning')
    axis.legend()
    figure.suptitle(title)
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Cross Entropy')


def peek_into_dataloader(dataloader: tf.data.Dataset) -> None:

    class_names = dataloader.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in dataloader.take(1):
        for i in range(9):
            axis = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    image_batch, label_batch = next(iter(dataloader))
    print("Image batch shape = {}".format(image_batch.shape))
    print("Label batch shape = {}".format(label_batch.shape))
    

def train_classifier(models: tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model],
                     training_dataloaders: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                     validation_dataloader: tf.data.Dataset,
                     number_of_epochs: int,
                     checkpoint_names: list[list[str]]) -> list[list[tf.keras.callbacks.History]]:

    assert len(checkpoint_names) == len(models) * len(training_dataloaders)

    histories = []

    for i in range(len(models)):
        model_histories = []

        initial_checkpoint_path = os.path.join("./checkpoints/initial_models", "model{}".format(i))
        models[i].save_weights(initial_checkpoint_path)
        models[i].layers[1].trainable = False

        for j in range(len(training_dataloaders)):
            model_histories.append(models[i].fit(training_dataloaders[j],
                                                 epochs=number_of_epochs,
                                                 validation_data=validation_dataloader))
            
            models[i].save_weights(os.path.join("./checkpoints") + checkpoint_names[i][j])
            models[i].load_weights(initial_checkpoint_path)

        models[i].layers[1].trainable = True
        histories.append(model_histories)

    return histories


