import os
import click

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import module from parallel directory
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir) 
from utils.video_frame_generator import VideoFrameGenerator


@click.command()
@click.option("--in-width", type=click.INT, default=224, help="Input image width in pixels.")
@click.option("--in-height", type=click.INT, default=224, help="Input image height in pixels.")
@click.option("--sequence-len", type=click.INT, default=10, help="Video sequence length to process.")
@click.option("--cnn-model", type=click.Choice(["vgg16", "resnet50"]), default="vgg16", help="CNN model type.")
@click.option("--do-freeze-backbone", type=click.BOOL, default=False, help="Whether or not freeze CNN model.")
@click.option("--num-gru-units", type=click.INT, default=256, help="Number of GRU units in model.")
@click.option("--num-dense-units", type=click.INT, default=1024, help="Number of dense units in model.")
@click.option("--early-stop-patience", type=click.INT, default=20, help="Number of non-improved results (epochs) to stop training.")
@click.option("--batch", type=click.INT, default=8, help="Batch size while training.")
@click.option("--epochs", type=click.INT, default=100, help="Max epochs to train.")
@click.option("--lr", type=click.INT, default=1e-3, help="Starting learning rate.")
@click.option("--reduce-lr-patience", type=click.INT, default=10, help="Number of non-improved results (epochs) to reduce lr.")
@click.option("--min-lr", type=click.INT, default=1e-8, help="Min learning rate.")
@click.option("--log-dir", type=click.STRING, default="log-cnngru/", help="Directory to log results.")
@click.option("--out-model-path", type=click.STRING, default="../models/cnngru.h5", help="Path of output model.")
@click.argument("dataset_path", type=click.STRING)
def train(dataset_path,
          in_width, in_height, sequence_len,
          cnn_model, do_freeze_backbone,
          num_gru_units, num_dense_units,
          early_stop_patience, batch, epochs, lr,
          reduce_lr_patience, min_lr,
          log_dir, out_model_path):

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

    def create_model():
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
        x = layers.TimeDistributed(layers.Flatten())(x)
        x = layers.GRU(num_gru_units, activation="relu")(x)
        x = layers.Dense(num_dense_units, activation="relu")(x)
        x = layers.Dense(num_dense_units, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=[input], outputs=[x])
        model.summary()

        return model

    model = create_model()

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

    # callbacks
    es = callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=early_stop_patience)
    tb_metrics = callbacks.TensorBoard(log_dir)
    mdl_chckpnt = callbacks.ModelCheckpoint(filepath=out_model_path, monitor='val_loss',
                                            save_best_only=True, save_weights_only=False)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                            patience=reduce_lr_patience, min_lr=min_lr)

    # compile model
    opt = optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    loss_fn = losses.BinaryCrossentropy()
    metrics_fn = ["accuracy", metrics.AUC()]
    model.compile(loss=loss_fn, optimizer=opt, metrics=metrics_fn)

    # fit
    model.fit(train_generator,
              epochs = epochs,
              validation_data = valid_generator,
              verbose = 1,
              shuffle=False,
              callbacks = [es, tb_metrics, mdl_chckpnt, reduce_lr])


if __name__ == "__main__":
    train()
