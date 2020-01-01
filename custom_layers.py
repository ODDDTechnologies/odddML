from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Multiply, Activation
from tensorflow.keras.initializers import glorot_uniform
import tensorflow as tf
import numpy as np
import cv2
import os


"""
Gated implementations
    GatedConv2D: Introduce a Conv2D layer (same number of filters) to multiply with its sigmoid activation.
    FullGatedConv2D: Introduce a Conv2D to extract features (linear and sigmoid), making a full gated process.
                     This process will double number of filters to make one convolutional process.
"""

class GatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, **kwargs):
        super(GatedConv2D, self).__init__(**kwargs)

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(GatedConv2D, self).call(inputs)
        linear = Activation("linear")(inputs)
        sigmoid = Activation("sigmoid")(output)

        return Multiply()([linear, sigmoid])

    def get_config(self):
        """Return the config of the layer"""

        config = super(GatedConv2D, self).get_config()
        return config



class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size, dropout):
        """
        Args:
            vocab_size: input language vocabulary
            embedding_dim: embeddig dimension 
            units: units 
            batch_size: batch size
        Raises:
        """
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.lstm_1 = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=dropout,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_2 = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=dropout,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_3 = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=dropout,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_4 = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=dropout,
                                         recurrent_initializer='glorot_uniform')

    def call(self, x, pre_state):
        """
        Args:
            x: input tensor
            pre_state: initial state in LSTM
        Returns:
            output: 
            state: hidden state and cell state used for next steps
        Raises:
        """
        x = self.embedding(x)
        x, state_h_1, state_c_1 = self.lstm_1(x, initial_state=pre_state[0])
        x, state_h_2, state_c_2 = self.lstm_2(x, initial_state=pre_state[1])
        x, state_h_3, state_c_3 = self.lstm_3(x, initial_state=pre_state[2])
        output, state_h_4, state_c_4 = self.lstm_4(x, initial_state=pre_state[3])
        state = [[state_h_1, state_c_1], [state_h_2, state_c_2], [state_h_3, state_c_3], [state_h_4, state_c_4]]
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, method, batch_size, dropout):
        """
        Args:
            vocab_size: target language vocabulary
            embedding_dim: embeddig dimension 
            units: units 
            batch_size: batch size
        """
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.lstm_1 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=dropout,
                                         recurrent_initializer='glorot_uniform')
        
        self.lstm_2 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=dropout,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_3 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=dropout,
                                         recurrent_initializer='glorot_uniform')

        self.lstm_4 = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         dropout=dropout,
                                         recurrent_initializer='glorot_uniform')

        self.attention_layer = AttentionLayer(units, method)

        self.W_c = tf.keras.layers.Dense(embedding_dim, activation='tanh')

        self.W_s = tf.keras.layers.Dense(vocab_size)



    def call(self, x, pre_state, enc_output, pre_h_t):
        x = self.embedding(x)

        # input_feeding shape == (batch_size, 1, word_embedding_dim + pre_h_t_embedding_dim)
        x = tf.concat([x, pre_h_t], axis=-1)
        x, state_h_1, state_c_1 = self.lstm_1(x, initial_state=pre_state[0])
        x, state_h_2, state_c_2 = self.lstm_2(x, initial_state=pre_state[1])
        x, state_h_3, state_c_3 = self.lstm_3(x, initial_state=pre_state[2])

        # dec_output shape == (batch_size, 1, units)
        dec_output, state_h_4, state_c_4 = self.lstm_4(x, initial_state=pre_state[3])
        
        state = [[state_h_1, state_c_1], [state_h_2, state_c_2], [state_h_3, state_c_3], [state_h_4, state_c_4]]

        context_vector = self.attention_layer(dec_output, enc_output)

        # h_t shape == (batch_size, 1, embedding_dim)
        h_t = self.W_c(tf.concat([tf.expand_dims(context_vector, 1), dec_output], axis=-1))
        #h_t = self.W_c(tf.concat([context_vector, tf.squeeze(dec_output)], axis=-1))

        # y_t shape == (batch_size, vocab_size)
        y_t = tf.squeeze(self.W_s(h_t), axis=1)

        return y_t, state, h_t

class AttentionLayer(tf.keras.Model):
    def __init__(self, units, method='concat'):
        super(AttentionLayer, self).__init__()
        # TODO: Three types of score function
        self.method = method
        self.W_a = tf.keras.layers.Dense(units)
        self.v_a = tf.keras.layers.Dense(1)

    def call(self, dec_h_t, enc_h_s):
        """
        Args:
            dec_h_t: current target state (batch_size, 1, units)
            enc_h_s: all source states (batch_size, seq_len, units)
        
        Returns:
            context_vector: (batch_size, units)
        """

        # concat_h = tf.concat([dec_h_t, enc_h_s], axis=1)
        # concat_h = tf.reshape(concat_h, [concat_h.shape[0] * concat_h.shape[1], concat_h.shape[2]])
        # print('concat_h shape:', concat_h.shape)

        # score shape == (batch_size, seq_len, 1)
        if self.method == 'concat':
            score = self.v_a(tf.nn.tanh(self.W_a(dec_h_t + enc_h_s)))
        elif self.method == 'general':
            score = tf.matmul(self.W_a(enc_h_s), dec_h_t, transpose_b=True)
        elif self.method == 'dot':
            score = tf.matmul(enc_h_s, dec_h_t, transpose_b=True) 

        # a_t shape == (batch_size, seq_len, 1)
        a_t = tf.nn.softmax(score, axis=1)

        # TODO: replace matmul operator with multiply operator
        # tf.matmul(a_t, enc_h_s, transpose_a=True) -> a_t * enc_h_s
        # result shape after * operation: (batch_size, seq_len, units)

        # (batch_size, 1, units)
        # context_vector shape == (batch_size, units)
        context_vector = tf.reduce_sum(a_t * enc_h_s, axis=1)

        return context_vector
"""
Tensorflow Keras layer implementation of the gated convolution.
    Args:
        filters (int): Number of output filters.
        kwargs: Other Conv2D keyword arguments.
    Reference (based):
        Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier,
        Language modeling with gated convolutional networks, in
        Proc. 34th Int. Conf. Mach. Learn. (ICML), vol. 70,
        Sydney, Australia, pp. 933–941, 2017.

        A. van den Oord and N. Kalchbrenner and O. Vinyals and L. Espeholt and A. Graves and K. Kavukcuoglu
        Conditional Image Generation with PixelCNN Decoders, 2016
        NIPS'16 Proceedings of the 30th International Conference on Neural Information Processing Systems
"""


class FullGatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(FullGatedConv2D, self).call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""

        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer"""

        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config

if __name__ == "__main__":
    pass