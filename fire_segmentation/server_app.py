"""Fire-Segmentation: A Flower / TensorFlow app."""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import cv2
import numpy
import six
import random

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fire_segmentation.task import load_model
from .keras_segmentation.models.unet import shufflnet_unet
from .keras_segmentation.data_utils.data_loader import get_pairs_from_paths, verify_segmentation_dataset, get_image_array, get_segmentation_array, image_segmentation_generator
from .keras_segmentation.predict import visualize_segmentation
from .keras_segmentation.data_utils.data_loader import class_colors
from keras.losses import CategoricalCrossentropy

from .keras_segmentation.models.config import IMAGE_ORDERING

SEED = 42  # scegli un numero fisso

# Imposta il seed per Python, NumPy e TensorFlow
#os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
numpy.random.seed(SEED)
tf.random.set_seed(SEED)

def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None):

    # if model is None and (checkpoints_path is not None):
    #     model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height,
                        ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr_softmax = pr.reshape((output_height, output_width, n_classes))  # probabilità per ogni classe
    pr_argmax = pr_softmax.argmax(axis=2)  # classi finali (per IoU, ecc.)
    

    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height)

    im_bw = cv2.threshold(seg_img[:,:,2], 147.0, 255.0, cv2.THRESH_BINARY)[1]

    smallest = numpy.amin(im_bw)
    biggest = numpy.amax(im_bw)

    
    ## extracting fire specific regions
    fire_specific_image = inp
    print(fire_specific_image.shape)

    for i in range(0,fire_specific_image.shape[0]):
        for j in range(0,fire_specific_image.shape[1]):
            if(im_bw[i][j] == 0):
                fire_specific_image[i][j] = 0
            else:
                fire_specific_image[i][j] = fire_specific_image[i][j]

    if out_fname is not None:
        cv2.imwrite(out_fname, im_bw)
        cv2.imwrite("D:/fire_segmentation/shufflenet_unet_results/shufflenet_unet8/shufflenet_unet8_fire057.png",fire_specific_image)

    return pr_argmax, pr_softmax



def get_evaluate_fn():
    """Return a callback that evaluates the global model"""
    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using provided centralised testset."""

        print("\033[34mCENTRALIZED EVALUATE\033[0m")
        model = shufflnet_unet(n_classes=2,  input_height=416, input_width=608)
        model.set_weights(parameters_ndarrays)

        # Load global test set
        inp_images_dir = '/Users/magambito/Development/Projects/fire-segmentation/fire_segmentation/Fire Dataset/images_prepped_test'
        annotations_dir = '/Users/magambito/Development/Projects/fire-segmentation/fire_segmentation/Fire Dataset/annotations_binary_test'
        loss_fn = CategoricalCrossentropy()
        paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

        assert type(inp_images) is list
        assert type(annotations) is list

        tp = np.zeros(model.n_classes)
        fp = np.zeros(model.n_classes)
        fn = np.zeros(model.n_classes)
        n_pixels = np.zeros(model.n_classes)

        loss = 0.0

        for inp, ann in tqdm(zip(inp_images, annotations)):
            pr_argmax, pr_softmax = predict(model, inp)

            gt = get_segmentation_array(ann, model.n_classes,
                                        model.output_width, model.output_height,
                                        no_reshape=True)

            # Calcolo della loss
            gt_tensor = tf.convert_to_tensor([gt], dtype=tf.float32)  # batch=1
            pr_tensor = tf.convert_to_tensor([pr_softmax], dtype=tf.float32)  # batch=1

            sample_loss = loss_fn(gt_tensor, pr_tensor).numpy()
            loss += sample_loss  # accumula la loss media per immagine

            # Continua con argmax per IoU, ecc.
            gt = gt.argmax(-1)
            pr = pr_argmax.flatten()
            gt = gt.flatten()

            for cl_i in range(model.n_classes):

                tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
                fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
                fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
                n_pixels[cl_i] += np.sum(gt == cl_i)

        cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
        n_pixels_norm = n_pixels / np.sum(n_pixels)
        frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
        mean_IU = np.mean(cl_wise_score)

        loss /= len(inp_images)


        return loss, {
            "frequency_weighted_IU": frequency_weighted_IU,
            "mean_IU": mean_IU,
            "class_wise_IU": cl_wise_score
        }
    
    return evaluate


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Load global test set
    test_images = '/Users/magambito/Development/Projects/fire-segmentation/fire_segmentation/Fire Dataset/images_prepped_test'
    test_annotations = '/Users/magambito/Development/Projects/fire-segmentation/fire_segmentation/Fire Dataset/annotations_binary_test'


    n_classes = 2  # Binary segmentation: fire vs. no fire
    # val_batch_size = 2
    # do_augment = False
    # augmentation_name= "aug_all"
    # input_height =  416
    # input_width =  608
    # output_height =  104
    # output_width =  152

    print("Verifying test dataset")
    verified = verify_segmentation_dataset(test_images,
                                            test_annotations,
                                            n_classes, 
                                            show_all_errors=True)
    
    # test_gen = image_segmentation_generator(
    #     test_images,
    #     test_annotations,
    #     batch_size=val_batch_size,
    #     n_classes=n_classes,
    #     input_height=input_height,
    #     input_width=input_width,
    #     output_height=output_height,
    #     output_width=output_width,
    #     do_augment=do_augment,
    #     augmentation_name=augmentation_name,
    # )

    # Define strategy
    strategy = strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
