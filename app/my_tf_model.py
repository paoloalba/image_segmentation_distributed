import os
import math
import json
import logging

from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow_examples.models.pix2pix import pix2pix

from custom_callbacks import OverrideProgbarLogger

class MYTFMODEL:
    def __init__(self, persistent_storage_path, strategy):
        self.get_batch_size()
        self.get_epochs()

        self.output_channels = 3

        self.get_data_sample_folder(input_data_sample=persistent_storage_path) 

        self.date_format = "%Y%m%d_%H%M%S"
        self.key_dir = "training_"
        self.chck_dir = os.path.join(persistent_storage_path, "training_checkpoints")
        os.makedirs(self.chck_dir, exist_ok=True)

        self.trained_model_key = "trained_model_"
        self.trained_model_dir = os.path.join(persistent_storage_path, "trained_models")
        os.makedirs(self.trained_model_dir, exist_ok=True)

        self.main_tb_log_dir = os.path.join(persistent_storage_path, "tensorboard_track")
        os.makedirs(self.main_tb_log_dir, exist_ok=True)

        self.title_for_display = ['Input Image', 'True Mask', 'Predicted Mask']

        self.train_source_set = None
        self.test_source_set = None
        self.model_history = None

        self.ref_time = datetime.now().strftime(self.date_format)

        self.callback_list = []

        self.strategy = strategy

    ##### Distributed training helpers #####
    def get_temp_dir(self, dirpath):
        task_id = self.strategy.cluster_resolver.task_id
        base_dirpath = 'workertemp_' + str(task_id)
        temp_dir = os.path.join(dirpath, base_dirpath)
        tf.io.gfile.makedirs(temp_dir)
        return temp_dir
    def write_filepath(self, filepath):
        dirpath = os.path.dirname(filepath)
        base = os.path.basename(filepath)
        if not self.is_chief_worker():
            dirpath = self.get_temp_dir(dirpath)
        return os.path.join(dirpath, base)

    ##### Parameter getters #####
    def get_batch_size(self):
        batch_size = os.getenv("BATCH_SIZE", "64")
        per_worker_batch_size = int(batch_size)

        per_worker_batch_size = 64
        tf_config = json.loads(os.environ['TF_CONFIG'])
        num_workers = len(tf_config['cluster']['worker'])
        self.batch_size = per_worker_batch_size * num_workers

        buffer_size = os.getenv("BUFFER_SIZE", "1000")
        self.buffer_size = int(buffer_size)
    def get_epochs(self):
        epochs = os.getenv("TRAINING_EPOCHS", "20")
        self.epochs = int(epochs)

        epoch_save_frq = os.getenv("TRAINING_SAVE_FREQ")
        epoch_save_num = os.getenv("TRAINING_SAVE_NUM")

        if epoch_save_num:
            epoch_save_num = int(epoch_save_num)
            self.epoch_save_frq = int(math.ceil(1.0 * self.epochs / epoch_save_num))
        elif epoch_save_frq:
            self.epoch_save_frq = int(epoch_save_frq)
        else:
            self.epoch_save_frq = self.epochs

        logging.info("Number of epochs set to {0}".format(self.epochs))
        logging.info("Save frq set to {0} (env save frq: {1}; env save num: {2})".format(self.epoch_save_frq, epoch_save_frq, epoch_save_num))
    def get_data_sample_folder(self, input_data_sample=None):
        data_folder = os.getenv("DATA_DIR_FOLDER", None)

        if data_folder is None:
            if input_data_sample is None:
                raise Exception("No folder to save data samples is given.")
            else:
                self.data_folder = os.path.join(input_data_sample, "datasets")
        else:
            self.data_folder = data_folder

        os.makedirs(self.data_folder, exist_ok=True)
    def is_chief_worker(self):
        task_type, task_id = (self.strategy.cluster_resolver.task_type,
                            self.strategy.cluster_resolver.task_id)

        # If `task_type` is None, this may be operating as single worker, which works
        # effectively as chief.
        return task_type is None or task_type == 'chief' or (task_type == 'worker' and task_id == 0)

    ##### Datasets loaders #####
    @staticmethod
    def normalize(input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask
    @tf.function
    def load_image_train(self, datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = self.normalize(input_image, input_mask)

        return input_image, input_mask
    def load_image_test(self, datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        input_image, input_mask = self.normalize(input_image, input_mask)

        return input_image, input_mask
    def load_samples(self):
        logging.info("Importing data samples: {0}".format(self.data_folder))
        dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir=self.data_folder, download=False)

        val_subsplits = os.getenv("VAL_SUBSPLITS", "5")
        self.val_subsplits = int(val_subsplits)

        self.train_length = info.splits['train'].num_examples
        self.validation_steps = info.splits['test'].num_examples//self.batch_size//self.val_subsplits

        train = dataset['train'].map(self.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = dataset['test'].map(self.load_image_test)

        train_dataset = train.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
        self.train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.test_dataset = test.batch(self.batch_size)

        self.train_source_set = train
        self.test_source_set = test
        logging.info("Data samples correctly imported.")

    ##### Model creator #####
    def unet_model(self, output_channels):
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = inputs

        # Downsampling through the model
        skips = self.down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same')  #64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
    def create_model(self):
        logging.info("Creating model")
        base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

        self.down_stack.trainable = False

        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]

        self.model = self.unet_model(self.output_channels)
        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        logging.info("Model created and compiled.")
    ##### Model trainer #####
    def get_callbacks(self):
        main_cp_dir = os.path.join(self.chck_dir, "{0}{1}".format(self.key_dir, self.ref_time))
        main_cp_dir = self.write_filepath(main_cp_dir)
        checkpoint_path = main_cp_dir + "/cp-{epoch:04d}.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                save_weights_only=True,
                                                                period=self.epoch_save_frq,
                                                                verbose=1)

        self.callback_list.append(cp_callback)
        self.model.save_weights(checkpoint_path.format(epoch=0))

        tb_log_dir = os.path.join(self.main_tb_log_dir, self.ref_time)
        checkpoint_path = self.write_filepath(checkpoint_path)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir,
                                                            histogram_freq=1)

        self.callback_list.append(tb_callback)

        self.callback_list.append(OverrideProgbarLogger(count_mode='steps'))
    def train_model(self):
        logging.info("Model will be trained.")

        self.get_callbacks()
        self.model_history = self.model.fit(self.train_dataset,
                                            epochs=self.epochs,
                                            steps_per_epoch=self.train_length // self.batch_size,
                                            validation_steps=self.validation_steps,
                                            validation_data=self.test_dataset,
                                            callbacks=self.callback_list)

        logging.info("Model has been trained.")

    ##### Model loader #####
    def check_for_old_training(self):
        all_elems = os.listdir(self.chck_dir)

        ref_date = None
        for eee in all_elems:
            elem_path = os.path.join(self.chck_dir, eee)
            if os.path.isdir(elem_path) and eee.startswith(self.key_dir):
                dir_date = eee.replace(self.key_dir, "")
                current_date = datetime.strptime(dir_date, self.date_format)

                if ref_date is None:
                    ref_date = current_date
                elif current_date > ref_date:
                    ref_date = current_date

        if ref_date is None:
            return None
        else:
            return "{0}{1}".format(self.key_dir, ref_date.strftime(self.date_format))
    def load_and_evaluate(self, check_point):
        latest_training = tf.train.latest_checkpoint(check_point)
        logging.info("Latest available training: {0}".format(latest_training))
        self.model.load_weights(latest_training)
        loss, acc = self.model.evaluate(self.test_dataset, verbose=2)
        logging.info("Restored model, accuracy: {:5.2f}%".format(100*acc))
        logging.info("Restored model, loss: {:5.2f}".format(loss))
    def try_load_previous_training(self):
        exist_training = self.check_for_old_training()

        if exist_training is None:
            logging.info("No pregress training")
        else:
            checko_path = os.path.join(self.chck_dir, exist_training)
            self.load_and_evaluate(checko_path)

    ##### save Model #####
    def save_model(self):
        file_path = os.path.join(self.trained_model_dir, "{0}{1}".format(self.trained_model_key, self.ref_time))
        file_path = self.write_filepath(file_path)
        self.model.save_weights(file_path)
        logging.info("Trained model has been saved: {0}".format(file_path))

    ##### use Model #####
    @staticmethod
    def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]
    def display(self, display_list):
        plt.figure(figsize=(15, 15))

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(self.title_for_display[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()
    def show_predictions(self, dataset=None, num=1):
        if dataset is None:
            dataset = self.train_source_set

        for image, mask in dataset.take(num):
            pred_mask = self.model.predict(image[tf.newaxis, ...])
            self.display([image, mask, self.create_mask(pred_mask)])

            # self.display(
            #     [image[0],
            #     mask[0],
            #     self.create_mask(self.model.predict(image[0][tf.newaxis, ...]))
            #     ])

    def plot_summary(self):
        if self.model_history:
            loss = self.model_history.history['loss']
            val_loss = self.model_history.history['val_loss']

            epochs = range(self.epochs)

            plt.figure()
            plt.plot(epochs, loss, 'r', label='Training loss')
            plt.plot(epochs, val_loss, 'bo', label='Validation loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.ylim([0, 1])
            plt.legend()
            plt.show()

