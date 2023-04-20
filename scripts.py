import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import random
import tensorflow as tf
from typing import Type, Optional
import multiprocessing

# Disclaimer: contain directly copied or modified code from:
#   1. ChatGPT (23 Mar)
#   2. https://www.tensorflow.org/tutorials/images/transfer_learning
#   3. https://www.kaggle.com/code/ateplyuk/mnist-efficientnet/notebook


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


def validation_accuracies_of_all_models_after_fine_tuning(classifier_histories: list[tf.keras.callbacks.History],
                                                          fine_tuning_histories: list[tf.keras.callbacks.History],
                                                          curve_labels: list[str],
                                                          classifier_number_of_epoches: int,
                                                          title: str,
                                                          figsize: Optional[tuple[int, int]] = None) -> None:
    figure, axis = plt.subplots(figsize=figsize)

    for i in range(len(curve_labels)):
        axis.plot(classifier_histories[i].history['val_accuracy'] + fine_tuning_histories[i].history['val_accuracy'], label=curve_labels[i])

    axis.set_ylim(0.8, 1)
    axis.plot([classifier_number_of_epoches - 1, classifier_number_of_epoches - 1],
              axis.get_ylim(), label='Start Fine Tuning')
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Accuracy')
    axis.legend()
    # figure.suptitle(title)


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


def validation_losses_of_all_models_after_fine_tuning(classifier_histories: list[tf.keras.callbacks.History],
                                                      fine_tuning_histories: list[tf.keras.callbacks.History],
                                                      curve_names: list[str],
                                                      classifier_number_of_epoches: int,
                                                      title: str,
                                                      figsize: Optional[tuple[int, int]] = None) -> None:
    figure, axis = plt.subplots(figsize=figsize)

    for i in range(len(curve_names)):
        axis.plot(classifier_histories[i].history['val_loss'] + fine_tuning_histories[i].history['val_loss'], label=curve_names[i])

    axis.set_ylim(0, 1.0)

    axis.plot([classifier_number_of_epoches - 1, classifier_number_of_epoches - 1],
              axis.get_ylim(), label='Start Fine Tuning')
    
    axis.legend()
    # figure.suptitle(title)
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
    

def train(model: tf.keras.Model,
          training_dataloader: tf.data.Dataset,
          validation_dataloader: tf.data.Dataset,
          optimizer: Type[tf.keras.optimizers.Optimizer],
          learning_rate: float,
          loss: tf.keras.losses.Loss,
          number_of_epochs: int) -> tf.keras.callbacks.History:
    
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
    return model.fit(training_dataloader, epochs=number_of_epochs, validation_data=validation_dataloader)


def train_one_classifier(shared_dictionary) -> None:

    shared_dictionary["validation_accuracy"] = shared_dictionary["model"].fit(shared_dictionary["training_dataloaders"],
                                                                              epochs=shared_dictionary["number_of_epochs"],
                                                                              validation_data=shared_dictionary["validation_dataloader"]).history['val_accuracy'][-1]


"""
Train each of the models using each of the dataloaders.

Return the final accuracies after each training.
"""
def train_classifiers(models: list[tf.keras.Model],
                      optimizers: list[Type[tf.keras.optimizers.Optimizer]],
                      training_dataloaders: list[tf.data.Dataset],
                      validation_dataloader: tf.data.Dataset,
                      learning_rate: float,
                      loss: tf.keras.losses.Loss,
                      number_of_epochs: int,
                      checkpoint_names: np.ndarray) -> np.ndarray:

    assert checkpoint_names.shape[0] == len(models)
    assert checkpoint_names.shape[1] == len(training_dataloaders)

    accuracies = np.zeros((len(models), len(training_dataloaders)))
    manager = multiprocessing.Manager()
    shared_dictionary = manager.dict()

    for i in range(len(models)):
        models[i].layers[1].trainable = False
        models[i].compile(optimizer=optimizers[i](learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
        assert(int(sum(tf.keras.backend.count_params(p) for p in models[i].trainable_variables)) == 5259279)

        shared_dictionary["model"] = models[i]
        shared_dictionary["validation_dataloader"] = validation_dataloader
        shared_dictionary["number_of_epochs"] = number_of_epochs

        for j in range(len(training_dataloaders)):
            shared_dictionary["training_dataloaders"] = training_dataloaders[j]

            process = multiprocessing.Process(target=train_one_classifier, args=(shared_dictionary))
            process.start()
            process.join()

            accuracies[i, j] = shared_dictionary["validation_accuracy"]
            shared_dictionary["model"].save(os.path.join("./saved_models", checkpoint_names[i, j]))

    return accuracies


def fine_tune_one_model(optimizer: Type[tf.keras.optimizers.Optimizer],
                        training_dataloader: tf.data.Dataset,
                        validation_dataloader: tf.data.Dataset,
                        learning_rate: float,
                        loss: tf.keras.losses.Loss,
                        number_of_epochs: int,
                        checkpoint_name: str,
                        percentage_of_fine_tune_layers: float,
                        accuracies: np.ndarray,
                        i: int, j: int, k: int) -> None:
    
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(os.path.join("./saved_models", checkpoint_name))

    model.layers[1].trainable = True
    freeze(model.layers[1], int(np.floor(len(model.layers[1].layers) * percentage_of_fine_tune_layers)))
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
    
    accuracies[i, j, k] = model.fit(training_dataloader, epochs=number_of_epochs, validation_data=validation_dataloader).history['val_accuracy'][-1]


def fine_tune_models(optimizers: list[Type[tf.keras.optimizers.Optimizer]],
                     training_dataloaders: list[tf.data.Dataset],
                     validation_dataloader: tf.data.Dataset,
                     learning_rate: float,
                     loss: tf.keras.losses.Loss,
                     number_of_epochs: int,
                     checkpoint_names: np.ndarray,
                     percentage_of_fine_tune_layers: list[float]) -> np.ndarray:
    
    assert checkpoint_names.shape[1] == len(training_dataloaders)
    
    accuracies = np.empty((checkpoint_names.shape[0], len(training_dataloaders), len(percentage_of_fine_tune_layers)))

    for i in range(checkpoint_names.shape[0]):
        for j in range(checkpoint_names.shape[1]):
            for k in range(len(percentage_of_fine_tune_layers)):

                process = Process(target=fine_tune_one_model, args=(optimizers[i],
                                                                    training_dataloaders[j],
                                                                    validation_dataloader,
                                                                    learning_rate,
                                                                    loss,
                                                                    number_of_epochs,
                                                                    checkpoint_names[i, j],
                                                                    percentage_of_fine_tune_layers[k],
                                                                    accuracies,
                                                                    i, j, k))
                process.start()
                process.join()    

    return accuracies


def train_classifier(model: tf.keras.Model,
                    training_dataloader: tf.data.Dataset,
                    validation_dataloader: tf.data.Dataset,
                    number_of_epochs: int,
                    optimizer: Type[tf.keras.optimizers.Optimizer],
                    learning_rate: float,
                    loss: Type[tf.keras.losses.Loss],
                    from_logits: bool,
                    metrics: list[str]) -> tf.keras.callbacks.History:

    print("Number of trainable parameters before freezing the base model: {}".format(int(sum(tf.keras.backend.count_params(p) for p in model.trainable_variables))))

    model.layers[1].trainable = False
    assert(int(sum(tf.keras.backend.count_params(p) for p in model.trainable_variables)) == 5259279)

    print("Number of trainable parameters after freezing the base model: {}".format(int(sum(tf.keras.backend.count_params(p) for p in model.trainable_variables))))

    model.compile(optimizer=optimizer(learning_rate=learning_rate), 
                  loss=loss(from_logits=from_logits),
                  metrics=metrics)

    return model.fit(training_dataloader,
                     epochs=number_of_epochs,
                     validation_data=validation_dataloader)


def fine_tune(model: tf.keras.Model,
              number_of_layers_to_freeze: int,
              classifier_training_history: tf.keras.callbacks.History,
              training_dataloader: tf.data.Dataset,
              validation_dataloader: tf.data.Dataset,
              total_number_of_epochs: int,
              optimizer: Type[tf.keras.optimizers.Optimizer],
              learning_rate: float,
              loss: Type[tf.keras.losses.Loss],
              from_logits: bool,
              metrics: list[str]) -> tf.keras.callbacks.History:

    assert(number_of_layers_to_freeze >= 0)

    model.layers[1].trainable = True

    print("Number of trainable parameters after unfreezing the entire base model: {}".format(int(sum(tf.keras.backend.count_params(p) for p in model.trainable_variables))))

    for layer in model.layers[1].layers[:number_of_layers_to_freeze]:
        layer.trainable = False

    print("Number of trainable parameters after freezing the first {} layers of the base model: {}".format(number_of_layers_to_freeze, int(sum(tf.keras.backend.count_params(p) for p in model.trainable_variables))))

    model.compile(optimizer=optimizer(learning_rate=learning_rate), 
                  loss=loss(from_logits=from_logits),
                  metrics=metrics)
    
    return model.fit(training_dataloader,
                     epochs=total_number_of_epochs,
                     initial_epoch=classifier_training_history.epoch[-1],
                     validation_data=validation_dataloader)
