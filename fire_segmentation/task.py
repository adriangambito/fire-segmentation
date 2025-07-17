"""Fire-Segmentation: A Flower / TensorFlow app."""

# TODO: import libraries
# import model
# import dataset with flwr-datasets
# import fit
# verificare steps_per_epoch, batch_size, epochs

import os
import random
import itertools
import numpy as np
import cv2

import keras
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from .keras_segmentation.models.unet import shufflnet_unet
from .keras_segmentation.data_utils.augmentation import augment_seg
from .keras_segmentation.data_utils.data_loader import get_pairs_from_paths, verify_segmentation_dataset, get_image_array, get_segmentation_array
from .keras_segmentation.models.config import IMAGE_ORDERING


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = shufflnet_unet(n_classes=2,  input_height=416, input_width=608)
    #model.summary()
    model.compile(loss='categorical_crossentropy',
                optimizer='SGD',
                metrics=['accuracy'])
    
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):

    train_images = '/Users/magambito/Development/Projects/fire-segmentation/fire_segmentation/Fire Dataset/images_prepped_train'
    train_annotations = '/Users/magambito/Development/Projects/fire-segmentation/fire_segmentation/Fire Dataset/annotations_binary_train'
    n_classes = 2  # Binary segmentation: fire vs. no fire
    batch_size = 2
    do_augment = False
    augmentation_name= "aug_all"
    input_height =  416
    input_width =  608
    output_height =  104
    output_width =  152

    print("Verifying training dataset")
    verified = verify_segmentation_dataset(train_images,
                                            train_annotations,
                                            n_classes, 
                                            show_all_errors=True)
    
    # Download and partition dataset
    # Ottieni tutte le coppie immagini/annotazioni
    img_seg_pairs = get_pairs_from_paths(train_images, train_annotations)
    total = len(img_seg_pairs)

    # Shuffle globale per garantire randomicità
    random.shuffle(img_seg_pairs)

    # Calcola l'intervallo per questa partizione
    samples_per_partition = total // num_partitions
    start_idx = partition_id * samples_per_partition
    end_idx = (partition_id + 1) * samples_per_partition if partition_id < num_partitions - 1 else total
    partition_pairs = img_seg_pairs[start_idx:end_idx]

    # Crea un generatore per questa partizione
    def train_generator():
        zipped = itertools.cycle(partition_pairs)

        while True:
            X = []
            Y = []
            for _ in range(batch_size):
                im_path, seg_path = next(zipped)

                im = cv2.imread(im_path, 1)
                seg = cv2.imread(seg_path, 1)

                if do_augment:
                    im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0], augmentation_name)

                X.append(get_image_array(im, input_width, input_height, ordering=IMAGE_ORDERING))
                Y.append(get_segmentation_array(seg, n_classes, output_width, output_height))

            return np.array(X), np.array(Y)

    return train_generator()
