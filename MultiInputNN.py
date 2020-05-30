import os

import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as kbck

from PIL import Image

import itertools
import ImgGenerator
from ImageConsumer import ImageConsumer


class MultiInputNN:
    Xnor = 127.  # input normalization
    Ynor = 255.  # output de-normalization
    Epochs = 4  # number of fit epoch
    TestProbes = 20  # number of test probes

    def __init__(self):
        self._generator = ImgGenerator.gen_numpy_2chanel_sample()

    def define_NN(self):
        tf.keras.backend.clear_session()  # For easy reset of notebook state.

        in1 = keras.Input(shape=(ImgGenerator.H, ImgGenerator.W // 2, 2), name='inp1')
        in_r1 = layers.Reshape((ImgGenerator.H, ImgGenerator.W), name="reshaped_input")(in1)

        lstm = layers.LSTM(units=256 * 4, name="lstm")(in_r1)
        lstm = layers.Reshape((128, 8), name="reshaped_lstm")(lstm)
        print(lstm)

        in_conv = layers.DepthwiseConv2D((128, 1), padding="same", data_format='channels_last', name="depth-conv")(in1)
        print(in_conv)
        in_conv = layers.Reshape((128, 128), name="reshape_conv")(in_conv)

        rnn = layers.SimpleRNN(256, name="rnn")(in_r1)
        rnn = layers.Reshape((128, 2), name="reshaped_rnn")(rnn)
        print(rnn)

        # Rotated
        rot_layer = layers.Lambda(lambda x: kbck.reverse(x, axes=0), output_shape=(64, 128, 2))(in1)
        rot_layer = layers.Reshape((ImgGenerator.H, ImgGenerator.W), name="reshaped_input2")(rot_layer)
        in_conv2 = layers.SimpleRNN(128, name="rnn2")(rot_layer)
        print(in_conv2)
        in_conv2 = layers.Reshape((128, 1), name="reshape_conv2")(in_conv2)

        d0 = layers.Dense(1024, activation="tanh", name="dense-inp")(in_r1)  #
        print(d0)

        d0 = layers.Concatenate(axis=2)([d0, lstm, in_conv, rnn, in_conv2])
        print(d0.shape)
        rnn = layers.Flatten()(d0)
        # rnn = layers.BatchNormalization(momentum=0.8)(rnn)
        # rnn = layers.LeakyReLU()(rnn)
        print(rnn)
        dense_1 = layers.Dense(2048, activation="relu")(rnn)  # , activation="relu"
        # dense_1 = layers.BatchNormalization(momentum=0.8)(dense_1)
        # dense_1 = layers.LeakyReLU()(dense_1)
        print(dense_1)
        # for layer_idx in range(0, 5):
        #    dense_1 = layers.BatchNormalization(momentum=0.8)(dense_1)
        #    dense_1 = layers.Dense(1024, activation="tanh", name=f"muldence{layer_idx}")(dense_1)#, activation="relu"
        dense_2 = layers.Dense(4096, activation="relu")(dense_1)  # , activation="relu"
        # dense_2 = layers.BatchNormalization(momentum=0.8)(dense_2)
        print(dense_2)

        output = layers.Dense(128 * 128)(dense_2)  # ,, activation="softplus"
        # output = layers.Softmax()(output)

        print(f"Last dense:{output}")

        output = layers.Reshape((128, 128))(output)
        print(f"Out layer:{output}")
        return [in1], [output]

    def create_model(self, inputs, outputs):
        model = keras.Model(inputs=inputs, outputs=outputs)
        # model = keras.Model(inputs=x, outputs=output)
        model.summary()
        model.compile(optimizer="Adam", loss="mse", metrics=["acc"])
        return model

    def fit_model(self, model):
        train_x, train_y = ([], [])
        for x, y in itertools.islice(self._generator, 0, MultiInputNN.Epochs + 30):
            train_x.append(x / MultiInputNN.Xnor)  # train_x.append(np.expand_dims(x, 0))
            train_y.append(y / MultiInputNN.Ynor)  #
        train_x = np.stack(train_x)
        train_y = np.stack(train_y)

        print(f"Shape of train_x:{train_x.shape}, train_y:{train_y.shape}")
        # print(f"Shape of train_x:{train_x[0].shape}, train_y:{train_y[0].shape}")

        history = model.fit(train_x, train_y, epochs=MultiInputNN.Epochs)
        ## Evaluate model
        test_x = []
        test_y = []
        T = 20
        # make test set
        for x, y in itertools.islice(self._generator, 0, T):
            test_x.append(x / MultiInputNN.Xnor)
            test_y.append(y / MultiInputNN.Ynor)

        test_x = np.stack(test_x)
        test_y = np.stack(test_y)  # convert from list to np.array

        print(f'Test: x: shape:{test_x.shape}, max:{np.amax(test_x)} ; y: shape:{test_y.shape} max:{np.amax(test_y)}')

        test_scores = model.evaluate(test_x, test_y, verbose=2)
        print('Test loss:', test_scores[0])
        print('Test accuracy:', test_scores[1])

    def use_model(self, model, image_consumer):
        test_x = []
        test_y = []
        # make test set
        for x, y in itertools.islice(self._generator, 0, MultiInputNN.TestProbes):
            test_x.append(x / MultiInputNN.Xnor)
            test_y.append(y / MultiInputNN.Ynor)

        test_x = np.stack(test_x)
        test_y = np.stack(test_y)  # convert from list to np.array

        print(f'Test: x: shape:{test_x.shape}, max:{np.amax(test_x)} ; y: shape:{test_y.shape} max:{np.amax(test_y)}')

        test_scores = model.evaluate(test_x, test_y, verbose=2)
        print('Test loss:', test_scores[0])
        print('Test accuracy:', test_scores[1])
        prediction = model.predict(test_x)

        print(f"Source Max item={np.amax(test_y)}, source-shape:{test_y.shape}, predictions-shape:{prediction.shape}")
        # ix = Image.fromarray((target_y[0]*255).astype(np.int8), 'L')

        for p, y in zip((prediction * MultiInputNN.Ynor), (test_y * MultiInputNN.Ynor).astype(np.uint8)):
            with image_consumer.step() as display:
                p_mean = p.mean()

                display.annotate(
                    f"Source item-max={np.amax(y)}, source-shape:{y.shape}, predictions: shape:{p.shape} :mean{p_mean}")
                iy = Image.fromarray(y, 'L')
                p_grey = p
                p_grey[p_grey > 255.] = 255.
                p_grey[p_grey < 0] = 0.
                p_grey = p_grey.astype(np.uint8)
                ip = Image.fromarray(p_grey, 'L')
                ip_bw = Image.fromarray(((p > p_mean).astype(np.int) * 255).astype(np.uint8), 'L')
                display.himage_list([iy, ip, ip_bw])


def main():
    example = MultiInputNN()
    nn = example.define_NN()
    model = example.create_model(*nn)
    example.fit_model(model)
    out_html = ImageConsumer()
    example.use_model(model, out_html)
    target = os.path.join(ImgGenerator.DefaultRenderDir, "multi_input_nn.html")
    out_html.as_html(target)
    print(f"Open '{target}' file to display results...")


if __name__ == "__main__":
    main()
