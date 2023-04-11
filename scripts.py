import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import sys
from typing import Optional


"""
Add a classifier to a base model.
"""
def add_classifier(base_model: tf.keras.Model,
                   number_of_classes: int) -> tf.keras.Model:

    output = base_model.output
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(1024, activation="relu")(output)
    output = tf.keras.layers.Dropout(0.5)(output)
    model = tf.keras.Model(input=base_model.input, output=tf.keras.layers.Dense(units=number_of_classes, activation="softmax")(output))

    return model


"""
Create a model from the base model.
"""
def model(base_model: tf.keras.Model,
          number_of_classes: int,
          learning_rate) -> tf.keras.Model:

    model = add_classifier(base_model, number_of_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


"""
Plot training and validation accuracy.
"""
def accuracies(history: tf.keras.callbacks.History,
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
    figure.suptitle('Training and Validation Accuracy')


"""
Plot Training and validation loss.
"""
def losses(history: tf.keras.callbacks.History,
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
    figure.suptitle('Training and Validation Loss')


# """
# Return a pre-trained model without the last classification layer.
# """
# def create_base_model(model_name: str, input_shape: tuple[int, int]) -> tf.keras.Model:

#     if model_name == "efficientnet":
#         return 
#     else:
#         print("The model {} is an invalid base model.".format(model_name))
#         sys.exit(1)

