import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.layers import LSTM, Bidirectional, TimeDistributed, Dense, Masking, Input, Lambda, Reshape


class brnn_ctc_lfbank():
    def __init__(self, nfeature, nclass, lr, momentum, is_training, is_SGD):
        self.nfeature = nfeature
        self.nclass = nclass
        self.lr = lr
        self.momentum = momentum
        self.is_training = is_training
        self.is_SGD = is_SGD
        self._model_init()
        if self.is_SGD:
            self.opt = keras.optimizers.SGD(learning_rate=self.lr, momentum = self.momentum)
        else:
            self.opt = keras.optimizers.Adam(learning_rate=self.lr)
        if self.is_training:
            self._ctc_init()
            self.compile_ctc()
        else:
            self.compile_model()
        
    def _model_init(self):
        self.input_data = Input(shape=(None, self.nfeature), dtype='float32',name='input')
        mask = Masking(mask_value= -1,input_shape=(None, self.nfeature), name = "mask1")(self.input_data)
        lstm = Bidirectional(LSTM(500, return_sequences=True, activity_regularizer = l2(5e-2), dropout = 0.4), name = "LSTM1")(mask)
        lstm = Bidirectional(LSTM(500, return_sequences=True, activity_regularizer = l2(5e-2), dropout = 0.4), name = "LSTM2")(lstm)
        lstm = Bidirectional(LSTM(500, return_sequences=True, activity_regularizer = l2(5e-2), dropout = 0.4), name = "LSTM3")(lstm)
        self.output = TimeDistributed(Dense(self.nclass+1, activation='softmax'), name = "output")(lstm)
        self.model = Model(inputs=self.input_data, outputs=self.output)
        
    def _ctc_init(self):
        y_true = Input(name='y_true', shape=[None], dtype='float32')
        input_length = Input(name='input_length_loss', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        y_true_mask = Masking(mask_value= -1, name = "mask2")(y_true)
        self.loss_out = Lambda(ctc_lambda_func, name='ctc')([y_true_mask, self.output, input_length, label_length])
        self.ctc_model = Model(inputs=[self.input_data, y_true, input_length, label_length], outputs=[self.loss_out])

    def compile_ctc(self):
        self.ctc_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=self.opt)
        tf.keras.utils.plot_model(self.ctc_model, to_file="ctc_model.png")

    def compile_model(self):
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=self.opt)
        tf.keras.utils.plot_model(self.model, to_file="decoder_model.png")


# model supplement
def ctc_lambda_func(args):   
    y_true, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

