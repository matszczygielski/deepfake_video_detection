import os
import click
from itertools import product

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks

from tensorflow.keras.backend import clear_session

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import module from parallel directory
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir) 
from utils.video_frame_generator import VideoFrameGenerator

from tensorboard.plugins.hparams import api as hp


@click.command()
@click.option("--in-width", type=click.INT, default=224, help="Input image width in pixels.")
@click.option("--in-height", type=click.INT, default=224, help="Input image height in pixels.")
@click.option("--sequence-len", type=click.INT, default=10, help="Video sequence length to process.")
@click.option("--cnn-model", type=click.Choice(["vgg16", "resnet50"]), default="vgg16", help="CNN model type.")
@click.option("--do-freeze-backbone", type=click.BOOL, default=False, help="Whether or not freeze CNN model.")
@click.option("--early-stop-patience", type=click.INT, default=5, help="Number of non-improved results (epochs) to stop training.")
@click.option("--batch", type=click.INT, default=8, help="Batch size while training.")
@click.option("--epochs", type=click.INT, default=100, help="Max epochs to train.")
@click.option("--log-dir", type=click.STRING, default="log-cnnconvlstm/", help="Directory to log results.")
@click.argument("dataset_path", type=click.STRING)
def train_hparams(dataset_path,
                  in_width, in_height, sequence_len,
                  cnn_model, do_freeze_backbone,
                  early_stop_patience, batch, epochs,
                  log_dir):

    hparams_space = [
        hp.HParam('num_dense_units', hp.Discrete([2048, 1024, 512])),
        hp.HParam('num_convlstm_filters', hp.Discrete([1024, 512])),
        hp.HParam('convlstm_kernel_size', hp.Discrete([5, 3])),
        hp.HParam('lr', hp.Discrete([1e-2, 1e-3, 1e-4, 1e-5]))
    ]

    # validate dataset_path
    paths_to_validate = [
        os.path.join(dataset_path, "train", "real"),
        os.path.join(dataset_path, "train", "fake"),
        os.path.join(dataset_path, "val", "fake"),
        os.path.join(dataset_path, "val", "fake")
    ]
    for path in paths_to_validate:
        if not os.path.isdir(path):
            raise OSError("Path '{}' doesn't exist".format(path))

    def create_model(hparams):
        if cnn_model == "vgg16":
            model_base = VGG16(weights="imagenet", include_top=False,
                               input_shape=(in_width, in_height, 3))
            preprocess_input_fn = vgg16_preprocess_input
        elif cnn_model == "resnet50":
            model_base = ResNet50(weights="imagenet", include_top=False,
                                  input_shape=(in_width, in_height, 3))
            preprocess_input_fn = resnet_preprocess_input

        if do_freeze_backbone:
            for layer in model_base.layers:
                layer.trainable = False

        input = layers.Input([sequence_len, in_width, in_height, 3])
        x = preprocess_input_fn(input)
        x = layers.TimeDistributed(model_base, input_shape=(sequence_len, in_width, in_height, 3))(x)
        x = layers.ConvLSTM2D(filters=hparams["num_convlstm_filters"],
                              kernel_size=(hparams["convlstm_kernel_size"], hparams["convlstm_kernel_size"]))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(hparams["num_dense_units"], activation="relu")(x)
        x = layers.Dense(hparams["num_dense_units"], activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=[input], outputs=[x])
        model.summary()

        return model

    def train(run_id, model, train_generator, valid_generator, hparams):
        # callbacks
        es = callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=early_stop_patience)
        tb_metrics = callbacks.TensorBoard("{}/run-{}".format(log_dir, run_id))
        tb_hparams = hp.KerasCallback("{}/run-{}".format(log_dir, run_id), hparams)

        # compile model
        opt = optimizers.SGD(learning_rate=hparams["lr"], momentum=0.9, nesterov=True)
        loss_fn = losses.BinaryCrossentropy()
        metrics_fn = ["accuracy"]
        model.compile(loss=loss_fn, optimizer=opt, metrics=metrics_fn)

        # fit
        history = model.fit(train_generator,
                            epochs = epochs,
                            validation_data = valid_generator,
                            verbose = 1,
                            shuffle=False,
                            callbacks = [es, tb_metrics, tb_hparams])

        return max(history.history["val_accuracy"])

    # collect hparams values possibilities
    # first, get cartesian product of max values count in hparam
    hparam_values_max_count = max([len(p.domain.values) for p in hparams_space])
    hparams_values_idxs_product_overloaded = product([x for x in range(hparam_values_max_count)],
                                                      repeat=len(hparams_space))

    # second, get idxs collection that are in range to all hparams
    hparams_values_idxs_collection = []
    for values_idxs in hparams_values_idxs_product_overloaded:
        is_value_idx_out_of_range = False
        for hparam_idx, value_idx in enumerate(values_idxs):
            if value_idx >= len(hparams_space[hparam_idx].domain.values):
                is_value_idx_out_of_range = True

        if not is_value_idx_out_of_range:
            hparams_values_idxs_collection.append(values_idxs)

    # reverse to run the largest model and make sure OOM doesn't appear
    hparams_values_idxs_collection.reverse()

    # iterate through all hparams values collection
    for run_id, hparams_values_idxs in enumerate(hparams_values_idxs_collection):
        hparams = dict()
        print("RUN {}: Creating hparams set:".format(run_id))
        for hparam_idx, value_idx in enumerate(hparams_values_idxs):
            param = hparams_space[hparam_idx]
            hparams[param.name] = param.domain.values[value_idx]
            print("{}: {}".format(param.name, hparams[param.name]))

        clear_session()
        model = create_model(hparams)

        # create data generators
        # some augumentation
        train_datagen = ImageDataGenerator(vertical_flip=True, rotation_range=30)
        valid_datagen = ImageDataGenerator(vertical_flip=True, rotation_range=30)

        train_generator = VideoFrameGenerator(classes=['real', 'fake'], 
                                              glob_pattern= dataset_path + 'train/{classname}/*.mp4',
                                              nb_frames=sequence_len,
                                              shuffle=True,
                                              batch_size=batch,
                                              target_shape=(in_width, in_height),
                                              nb_channel=3,
                                              transformation=train_datagen,
                                              use_frame_cache=True,
                                              class_mode="binary",
                                              rescale=1)

        valid_generator = VideoFrameGenerator(classes=['real', 'fake'], 
                                              glob_pattern= dataset_path + 'val/{classname}/*.mp4',
                                              nb_frames=sequence_len,
                                              shuffle=True,
                                              batch_size=batch,
                                              target_shape=(in_width, in_height),
                                              nb_channel=3,
                                              transformation=valid_datagen,
                                              use_frame_cache=True,
                                              class_mode="binary",
                                              rescale=1)

        train(run_id, model, train_generator, valid_generator, hparams)


if __name__ == "__main__":
    train_hparams()
