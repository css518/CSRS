from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.engine import Input
from keras.layers import Concatenate, Dot, Embedding, Dropout, Lambda, Activation, Dense, Reshape, Conv1D, MaxPooling1D, \
    Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import logging
from attention_layer import AttentionLayer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CSRS:
    def __init__(self, config):
        self.config = config
        self.model_params = config.get('model_params', dict())
        self.data_params = config.get('data_params', dict())
        self.methname = Input(shape=(self.data_params['methname_len'],), dtype='int32', name='i_methname')
        self.apiseq = Input(shape=(self.data_params['apiseq_len'],), dtype='int32', name='i_apiseq')
        self.tokens = Input(shape=(self.data_params['tokens_len'],), dtype='int32', name='i_tokens')
        self.desc = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_good')

        self._training_model = None

        if not os.path.exists(self.config['workdir'] + 'models/' + self.model_params['model_name'] + '/'):
            os.makedirs(self.config['workdir'] + 'models/' + self.model_params['model_name'] + '/')

    def build(self):
        logger.debug('Building Code Representation Model')
        self.methname = Input(shape=(self.data_params['methname_len'],), dtype='int32', name='methname')
        self.apiseq = Input(shape=(self.data_params['apiseq_len'],), dtype='int32', name='apiseq')
        self.tokens = Input(shape=(self.data_params['tokens_len'],), dtype='int32', name='tokens')

        ## method name representation ##
        init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_methname']) if \
            self.model_params['init_embed_weights_methname'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=init_emb_weights,
                              mask_zero=False,
                              name='embedding_methname')
        methname_embedding = embedding(self.methname)
        dropout = Dropout(0.25, name='dropout_methname_embed')
        methname_dropout = dropout(methname_embedding)

        methname_conv1 = Conv1D(100, 1, padding='valid', activation='relu', strides=1, name='methname_conv1')
        methname_conv2 = Conv1D(100, 2, padding='valid', activation='relu', strides=1, name='methname_conv2')
        methname_conv3 = Conv1D(100, 3, padding='valid', activation='relu', strides=1, name='methname_conv3')
        methname_conv1_out = methname_conv1(methname_dropout)
        methname_conv2_out = methname_conv2(methname_dropout)
        methname_conv3_out = methname_conv3(methname_dropout)
        dropout = Dropout(0.25, name='dropout_methname_conv')
        methname_conv1_dropout = dropout(methname_conv1_out)
        methname_conv2_dropout = dropout(methname_conv2_out)
        methname_conv3_dropout = dropout(methname_conv3_out)
        merged_methname = Concatenate(name='methname_merge', axis=1)(
            [methname_conv1_dropout, methname_conv2_dropout, methname_conv3_dropout])

        ## API Sequence Representation ##
        init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_api']) if \
            self.model_params['init_embed_weights_api'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              mask_zero=False,
                              name='embedding_apiseq')
        apiseq_embedding = embedding(self.apiseq)
        dropout = Dropout(0.25, name='dropout_apiseq_embed')
        apiseq_dropout = dropout(apiseq_embedding)

        api_conv1 = Conv1D(100, 1, padding='valid', activation='relu', strides=1, name='api_conv1')
        api_conv2 = Conv1D(100, 2, padding='valid', activation='relu', strides=1, name='api_conv2')
        api_conv3 = Conv1D(100, 3, padding='valid', activation='relu', strides=1, name='api_conv3')
        api_conv1_out = api_conv1(apiseq_dropout)
        api_conv2_out = api_conv2(apiseq_dropout)
        api_conv3_out = api_conv3(apiseq_dropout)
        dropout = Dropout(0.25, name='dropout_api_conv')
        api_conv1_dropout = dropout(api_conv1_out)
        api_conv2_dropout = dropout(api_conv2_out)
        api_conv3_dropout = dropout(api_conv3_out)
        merged_api = Concatenate(name='api_merge', axis=1)([api_conv1_dropout, api_conv2_dropout, api_conv3_dropout])

        ## Tokens Representation ##
        init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_tokens']) if \
            self.model_params['init_embed_weights_tokens'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=init_emb_weights,
                              name='embedding_tokens')
        tokens_embedding = embedding(self.tokens)
        dropout = Dropout(0.25, name='dropout_tokens_embed')
        tokens_dropout = dropout(tokens_embedding)

        tokens_conv1 = Conv1D(100, 1, padding='valid', activation='relu', strides=1, name='tokens_conv1')
        tokens_conv2 = Conv1D(100, 2, padding='valid', activation='relu', strides=1, name='tokens_conv2')
        tokens_conv3 = Conv1D(100, 3, padding='valid', activation='relu', strides=1, name='tokens_conv3')
        tokens_conv1_out = tokens_conv1(tokens_dropout)
        tokens_conv2_out = tokens_conv2(tokens_dropout)
        tokens_conv3_out = tokens_conv3(tokens_dropout)
        dropout = Dropout(0.25, name='dropout_tokens_conv')
        tokens_conv1_dropout = dropout(tokens_conv1_out)
        tokens_conv2_dropout = dropout(tokens_conv2_out)
        tokens_conv3_dropout = dropout(tokens_conv3_out)
        merged_tokens = Concatenate(name='tokens_merge', axis=1)(
            [tokens_conv1_dropout, tokens_conv2_dropout, tokens_conv3_dropout])

        merged_code = Concatenate(name='code_merge', axis=1)([merged_methname, merged_api, merged_tokens])  # (122,200)

        ## Desc Representation ##
        logger.debug('Building Desc Representation Model')
        self.desc = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='desc')
        init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_desc']) if \
            self.model_params['init_embed_weights_desc'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        embedding = Embedding(input_dim=self.data_params['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=init_emb_weights,
                              mask_zero=False,
                              name='embedding_desc')
        desc_embedding = embedding(self.desc)
        dropout = Dropout(0.25, name='dropout_desc_embed')
        desc_dropout = dropout(desc_embedding)

        desc_conv1 = Conv1D(100, 1, padding='valid', activation='relu', strides=1, name='desc_conv1')
        desc_conv2 = Conv1D(100, 2, padding='valid', activation='relu', strides=1, name='desc_conv2')
        desc_conv3 = Conv1D(100, 3, padding='valid', activation='relu', strides=1, name='desc_conv3')
        desc_conv1_out = desc_conv1(desc_dropout)
        desc_conv2_out = desc_conv2(desc_dropout)
        desc_conv3_out = desc_conv3(desc_dropout)
        dropout = Dropout(0.25, name='dropout_desc_conv')
        desc_conv1_dropout = dropout(desc_conv1_out)
        desc_conv2_dropout = dropout(desc_conv2_out)
        desc_conv3_dropout = dropout(desc_conv3_out)
        merged_desc = Concatenate(name='desc_merge', axis=1)(
            [desc_conv1_dropout, desc_conv2_dropout, desc_conv3_dropout])

        # relevance matching
        relevance_matrix = Dot(axes=-1, name='rel_mat')([merged_code, desc_conv1_dropout])
        norm = Activation('softmax', name='rel_mat_softmax')(relevance_matrix)
        rel_maxp = GlobalMaxPooling1D(name='rel_mat_maxp')(norm)
        rel_meanp = GlobalAveragePooling1D(name='rel_mat_meanp')(norm)
        rele_out = Concatenate(axis=-1, name='rel_merge')([rel_maxp, rel_meanp])

        # semantic matching
        attention = AttentionLayer(name='attention_layer')
        attention_out = attention([merged_code, desc_conv1_dropout])

        gmp_1 = GlobalMaxPooling1D(name='blobalmaxpool_colum')
        att_1 = gmp_1(attention_out)
        activ1 = Activation('softmax', name='AP_active_colum')
        att_1_next = activ1(att_1)
        dot1 = Dot(axes=1, normalize=False, name='column_dot')
        desc_out = dot1([att_1_next, desc_conv1_dropout])

        attention_trans_layer = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)), name='trans_attention')
        attention_transposed = attention_trans_layer(attention_out)
        gmp_2 = GlobalMaxPooling1D(name='blobalmaxpool_row')
        att_2 = gmp_2(attention_transposed)
        activ2 = Activation('softmax', name='AP_active_row')
        att_2_next = activ2(att_2)
        dot2 = Dot(axes=1, normalize=False, name='row_dot')
        code_out = dot2([att_2_next, merged_code])

        logger.debug('Building training model')
        feature_vector = Concatenate(axis=-1, name='final_feature_vec')([rele_out, desc_out, code_out])
        # feature_vector = Concatenate(axis=-1, name='final_feature_vec')([desc_out, code_out])

        feature_vector1 = Dense(256, activation='relu', name="feature_vector1")(feature_vector)
        feature_vector1 = BatchNormalization(name='bn1')(feature_vector1)
        feature_vector2 = Dense(128, activation='relu', name="feature_vector2")(feature_vector1)
        feature_vector2 = BatchNormalization(name='bn2')(feature_vector2)
        prediction = Dense(1, activation='sigmoid', name="prediction")(feature_vector2)
        self._training_model = Model(inputs=[self.methname, self.apiseq, self.tokens, self.desc], outputs=[prediction],
                                     name='training_model')
        print('\nsummary of training model')
        self._training_model.summary()

    def compile(self, optimizer, **kwargs):
        logger.info('compiling models')
        self._training_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'], **kwargs)

    def fit(self, x, y, **kwargs):
        assert self._training_model is not None, 'Must compile the model before fitting data'
        return self._training_model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self._training_model.predict(x, **kwargs)

    def save(self, train_model_file, **kwargs):
        assert self._training_model is not None, 'Must compile the model before saving weights'
        self._training_model.save_weights(train_model_file, **kwargs)

    def load(self, train_model_file, **kwargs):
        assert self._training_model is not None, 'Must compile the model before saving weights'
        self._training_model.load_weights(train_model_file, **kwargs)
