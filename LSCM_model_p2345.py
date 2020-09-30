import numpy as np
import tensorflow as tf
import sys
from deeplab_resnet import model as deeplab101
from util.cell import ConvLSTMCell

from util import data_reader
from util.processing_tools import *
from util import loss


class LSCM_model(object):

    def __init__(self, batch_size=1,
                 num_steps=30,
                 vf_h=40,
                 vf_w=40,
                 H=320,
                 W=320,
                 vf_dim=2048,
                 vocab_size=12112,
                 w_emb_dim=1000,
                 v_emb_dim=1000,
                 mlp_dim=500,
                 start_lr=0.00025,
                 lr_decay_step=800000,
                 lr_decay_rate=1.0,
                 rnn_size=1000,
                 keep_prob_rnn=1.0,
                 keep_prob_emb=1.0,
                 keep_prob_mlp=1.0,
                 num_rnn_layers=1,
                 optimizer='adam',
                 weight_decay=0.0005,
                 mode='eval',
                 conv5=False,
                 glove_dim=300,
                 emb_name='Gref'):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.H = H
        self.W = W
        self.vf_dim = vf_dim
        self.start_lr = start_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.vocab_size = vocab_size
        self.w_emb_dim = w_emb_dim
        self.v_emb_dim = v_emb_dim
        self.glove_dim = glove_dim
        self.emb_name = emb_name
        self.mlp_dim = mlp_dim
        self.rnn_size = rnn_size
        self.keep_prob_rnn = keep_prob_rnn
        self.keep_prob_emb = keep_prob_emb
        self.keep_prob_mlp = keep_prob_mlp
        self.num_rnn_layers = num_rnn_layers
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.mode = mode
        self.conv5 = conv5

        self.words = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.im = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])
        self.target_fine = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 1])
        self.valid_idx = tf.placeholder(tf.int32, [self.batch_size, 1])
        self.graph_adj = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, self.num_steps])
        self.tree_height = tf.placeholder(tf.int32, [self.batch_size, 1])

        resmodel = deeplab101.DeepLabResNetModel({'data': self.im}, is_training=False)
        self.visual_feat_c5 = resmodel.layers['res5c_relu']
        self.visual_feat_c4 = resmodel.layers['res4b22_relu']
        self.visual_feat_c3 = resmodel.layers['res3b3_relu']
        self.visual_feat_c2 = resmodel.layers['res2c_relu']  # [2H, 2W, 256]

        # GloVe Embedding
        glove_np = np.load('data/{}_spacy_emb.npy'.format(self.emb_name))
        print("Size of {} GloVe embedding: {}".format(self.emb_name, glove_np.shape))
        self.glove = tf.convert_to_tensor(glove_np, tf.float32)  # [vocab_size, 300]

        # graph_adj trunc, [b, T, T]
        self.adj_weight = tf.slice(self.graph_adj, [0, 0, 0],
                                   [-1, self.num_steps - self.valid_idx[0, 0], self.num_steps - self.valid_idx[0, 0]])

        with tf.variable_scope("text_objseg"):
            self.build_graph()
            if self.mode == 'eval':
                return
            self.train_op()

    def build_graph(self):
        print("#" * 30)
        print("LSCM_model_p2345, function version")
        print("#" * 30)

        embedding_mat = tf.Variable(self.glove)
        embedded_seq = tf.nn.embedding_lookup(embedding_mat,
                                              tf.transpose(self.words))  # [num_step, batch_size, glove_emb]
        print("Build Glove Embedding.")

        rnn_cell_basic = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=False)
        if self.mode == 'train' and self.keep_prob_rnn < 1:
            rnn_cell_basic = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_basic, output_keep_prob=self.keep_prob_rnn)
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_basic] * self.num_rnn_layers, state_is_tuple=False)

        state = cell.zero_state(self.batch_size, tf.float32)
        state_shape = state.get_shape().as_list()
        state_shape[0] = self.batch_size
        state.set_shape(state_shape)

        words_feat_list = []

        def f1():
            # return tf.constant(0.), state
            return tf.zeros([self.batch_size, self.rnn_size]), state

        def f2():
            # Word input to embedding layer
            w_emb = embedded_seq[n, :, :]
            if self.mode == 'train' and self.keep_prob_emb < 1:
                w_emb = tf.nn.dropout(w_emb, self.keep_prob_emb)
            return cell(w_emb, state)

        with tf.variable_scope("RNN"):
            for n in range(self.num_steps):
                if n > 0:
                    tf.get_variable_scope().reuse_variables()

                # rnn_output, state = cell(w_emb, state)
                rnn_output, state = tf.cond(tf.equal(self.words[0, n], tf.constant(0)), f1, f2)
                word_feat = tf.reshape(rnn_output, [self.batch_size, 1, self.rnn_size])
                words_feat_list.append(word_feat)

        # words_feat: [B, num_steps, rnn_size]
        words_feat = tf.concat(words_feat_list, 1)
        words_feat = tf.slice(words_feat, [0, self.valid_idx[0, 0], 0],
                              [-1, self.num_steps - self.valid_idx[0, 0], -1])  # [B, T, C]

        lang_feat = tf.reduce_max(words_feat, 1)  # [rnn_dim]
        lang_feat = tf.reshape(lang_feat, [self.batch_size, 1, 1, self.rnn_size])
        lang_feat = tf.nn.l2_normalize(lang_feat, 3)  # [B, 1, 1, C]

        words_feat = tf.nn.l2_normalize(words_feat, 2)
        # words_feat: [B, 1, num_words, rnn_size]
        words_feat = tf.expand_dims(words_feat, 1)

        visual_feat_c5 = self._conv("c5_lateral", self.visual_feat_c5, 1, self.vf_dim, self.v_emb_dim, [1, 1, 1, 1])
        visual_feat_c5 = tf.nn.l2_normalize(visual_feat_c5, 3)
        visual_feat_c4 = self._conv("c4_lateral", self.visual_feat_c4, 1, 1024, self.v_emb_dim, [1, 1, 1, 1])
        visual_feat_c4 = tf.nn.l2_normalize(visual_feat_c4, 3)
        visual_feat_c3 = self._conv("c3_lateral", self.visual_feat_c3, 1, 512, self.v_emb_dim, [1, 1, 1, 1])
        visual_feat_c3 = tf.nn.l2_normalize(visual_feat_c3, 3)
        visual_feat_c2 = self._conv("c2_lateral", self.visual_feat_c2, 3, 256, self.v_emb_dim, [1, 2, 2, 1])
        visual_feat_c2 = tf.nn.l2_normalize(visual_feat_c2, 3)

        # Generate spatial grid
        spatial = tf.convert_to_tensor(generate_spatial_batch(self.batch_size, self.vf_h, self.vf_w))

        fusion_c5 = self.build_full_module(visual_feat_c5, words_feat, lang_feat, spatial, level="c5")
        fusion_c4 = self.build_full_module(visual_feat_c4, words_feat, lang_feat, spatial, level="c4")
        fusion_c3 = self.build_full_module(visual_feat_c3, words_feat, lang_feat, spatial, level="c3")
        fusion_c2 = self.build_full_module(visual_feat_c2, words_feat, lang_feat, spatial, level="c2")

        score_c5 = self._conv("score_c5", fusion_c5, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c5 = tf.image.resize_bilinear(score_c5, [self.H, self.W])
        score_c4 = self._conv("score_c4", fusion_c4, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c4 = tf.image.resize_bilinear(score_c4, [self.H, self.W])
        score_c3 = self._conv("score_c3", fusion_c3, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c3 = tf.image.resize_bilinear(score_c3, [self.H, self.W])
        score_c2 = self._conv("score_c2", fusion_c2, 3, self.mlp_dim, 1, [1, 1, 1, 1])
        self.up_c2 = tf.image.resize_bilinear(score_c2, [self.H, self.W])

        # Convolutional LSTM
        convlstm_cell = ConvLSTMCell([self.vf_h, self.vf_w], self.mlp_dim, [1, 1])
        convlstm_outputs, states = tf.nn.dynamic_rnn(convlstm_cell, tf.convert_to_tensor(
            [[fusion_c5[0], fusion_c4[0], fusion_c3[0], fusion_c2[0], fusion_c3[0], fusion_c4[0], fusion_c5[0]]]),
                                                     dtype=tf.float32)

        score = self._conv("score", convlstm_outputs[:, -1], 3, self.mlp_dim, 1, [1, 1, 1, 1])

        self.pred = score
        self.up = tf.image.resize_bilinear(self.pred, [self.H, self.W])
        self.sigm = tf.sigmoid(self.up)

    def mutan_head(self, lang_feat, spatial_feat, visual_feat, level=''):
        # visual feature transform
        vis_trans = tf.concat([visual_feat, spatial_feat], 3)   # [B, H, W, C+8]
        vis_trans = self._conv("vis_trans_{}".format(level), vis_trans, 1,
                               self.v_emb_dim+8, self.v_emb_dim, [1, 1, 1, 1])
        vis_trans = tf.nn.tanh(vis_trans)  # [B, H, W, C]

        # lang feature transform
        lang_trans = self._conv("lang_trans_{}".format(level), lang_feat,
                                1, self.rnn_size, self.v_emb_dim, [1, 1, 1, 1])

        lang_trans = tf.nn.tanh(lang_trans)  # [B, 1, 1, C]

        mutan_feat = vis_trans * lang_trans  # [B, H, W, C]
        return mutan_feat

    def mutan_fusion(self, lang_feat, spatial_feat, visual_feat, level=''):
        # fuse language feature and visual feature
        # lang_feat: [B, 1, 1, C], visual_feat: [B, H, W, C], spatial_feat: [B, H, W, 8]
        # output: [B, H, W, C']
        head1 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head1'.format(level))
        head2 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head2'.format(level))
        head3 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head3'.format(level))
        head4 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head4'.format(level))
        head5 = self.mutan_head(lang_feat, spatial_feat, visual_feat, '{}_head5'.format(level))

        fused_feats = tf.stack([head1, head2, head3, head4, head5], axis=4)  # [B, H, W, C, 5]
        fused_feats = tf.reduce_sum(fused_feats, 4)  # [B, H, W, C]
        fused_feats = tf.nn.tanh(fused_feats)
        fused_feats = tf.nn.l2_normalize(fused_feats, 3)

        print("Build Mutan Fusion Module.")

        return fused_feats

    def build_full_module(self, visual_feat, words_feat, lang_feat, spatial, level=""):
        mm_feat = self.mutan_fusion(lang_feat, spatial, visual_feat, level=level)
        print("Build Mutan Fusion Module")
        vagr_feat = self.build_lscm(mm_feat, words_feat, spatial, level=level)
        print("Build Visual Attention Module")

        tiled_lang_feat = tf.tile(lang_feat, [1, self.vf_w, self.vf_h, 1])  # [b, h, w, C]
        feat_all = tf.concat([mm_feat, vagr_feat, tiled_lang_feat, spatial], 3)
        # Feature fusion
        fusion = self._conv("fusion_{}".format(level), feat_all, 1, self.v_emb_dim * 2 + self.rnn_size + 8,
                            self.mlp_dim, [1, 1, 1, 1])
        fusion = tf.nn.relu(fusion)
        return fusion

    def graph_conv(self, graph_feat, nodes_num, nodes_dim, adj_mat, graph_name="", level=""):
        # Node message passing
        graph_feat_reshaped = tf.reshape(graph_feat, [self.batch_size, nodes_num, nodes_dim])
        gconv_feat = tf.matmul(adj_mat, graph_feat_reshaped)  # [B, nodes_num, nodes_dim]
        gconv_feat = tf.reshape(gconv_feat, [self.batch_size, 1, nodes_num, nodes_dim])
        gconv_feat = tf.contrib.layers.layer_norm(gconv_feat,
                                                  scope="gconv_feat_ln_{}_{}".format(graph_name, level))
        gconv_feat = graph_feat + gconv_feat
        gconv_feat = tf.nn.relu(gconv_feat)  # [B, 1, nodes_num, nodes_dim]
        gconv_update = self._conv("gconv_update_{}_{}".format(graph_name, level),
                                  gconv_feat, 1, nodes_dim, nodes_dim, [1, 1, 1, 1])
        gconv_update = tf.contrib.layers.layer_norm(gconv_update,
                                                    scope="gconv_update_ln_{}_{}".format(graph_name, level))
        gconv_update = tf.nn.relu(gconv_update)
        return gconv_update

    def build_lscm(self, vis_la_sp, words_feat, spatial, level=""):
        # Visual Attention
        vis_key = self._conv("vis_key_{}".format(level), vis_la_sp, 1, self.v_emb_dim, self.v_emb_dim, [1, 1, 1, 1])
        vis_key = tf.reshape(vis_key, [self.batch_size, self.vf_h * self.vf_w, self.v_emb_dim])
        words_query = self._conv("words_query_{}".format(level), words_feat, 1, self.rnn_size, self.rnn_size,
                                 [1, 1, 1, 1])
        words_query = tf.reshape(words_query, [self.batch_size, self.num_steps - self.valid_idx[0, 0], self.rnn_size])
        vis_la_sp = tf.reshape(vis_la_sp, [self.batch_size, self.vf_h * self.vf_w, self.v_emb_dim])
        vis_attn_map = tf.matmul(vis_key, words_query, transpose_b=True)  # [B, vf_h * vf_w, num_words]
        # Normalization for affinity matrix
        vis_attn_map = tf.divide(vis_attn_map, self.rnn_size ** 0.5)
        vis_attn_map = tf.nn.softmax(vis_attn_map, axis=1)
        # Adjacent matrix node message passing
        vis_attn_feat = tf.matmul(vis_attn_map, vis_la_sp, transpose_a=True)  # [B, num_words, v_emb_dim]
        vis_attn_feat = tf.reshape(vis_attn_feat,
                                   [self.batch_size, 1, self.num_steps - self.valid_idx[0, 0], self.v_emb_dim])
        vis_attn_feat = tf.nn.l2_normalize(vis_attn_feat, 3)  # [B, 1, num_words, v_emb_dim]

        vagr_query = self._conv("vagr_query_{}".format(level), vis_attn_feat, 1, self.v_emb_dim, self.v_emb_dim,
                                [1, 1, 1, 1])
        vagr_query = tf.reshape(vagr_query, [self.batch_size, self.num_steps - self.valid_idx[0, 0], self.v_emb_dim])
        vagr_key = self._conv("vagr_key_{}".format(level), vis_attn_feat, 1, self.v_emb_dim, self.v_emb_dim,
                              [1, 1, 1, 1])
        vagr_key = tf.reshape(vagr_key, [self.batch_size, self.num_steps - self.valid_idx[0, 0], self.v_emb_dim])
        vagr_mat = tf.matmul(vagr_query, vagr_key, transpose_b=True)  # [B, num_words, num_words]
        # Normalization for affinity matrix
        vagr_mat = tf.divide(vagr_mat, self.v_emb_dim ** 0.5)
        vagr_mat = tf.nn.softmax(vagr_mat, axis=2)

        # translate to tree adj mat
        vagr_mat = vagr_mat * self.adj_weight

        # Adjacent matrix node message passing
        vagr_feat = self.graph_conv(vis_attn_feat, self.num_steps - self.valid_idx[0, 0],
                                    self.v_emb_dim, vagr_mat, "g1", level)

        # [B, num_words, v_emb_dim]
        vagr_feat = tf.reshape(vagr_feat, [self.batch_size, self.num_steps - self.valid_idx[0, 0], self.v_emb_dim])
        vagr_feat = tf.matmul(vis_attn_map, vagr_feat)  # [B, vf_h * vf_w, v_emb_dim]
        vagr_feat = tf.reshape(vagr_feat, [self.batch_size, self.vf_h, self.vf_w, self.v_emb_dim])
        vagr_feat = tf.nn.l2_normalize(vagr_feat, 3)
        return vagr_feat

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.conv2d(x, w, strides, padding='SAME') + b

    def _atrous_conv(self, name, x, filter_size, in_filters, out_filters, rate):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                                initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.atrous_conv2d(x, w, rate=rate, padding='SAME') + b

    def train_op(self):
        if self.conv5:
            tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('text_objseg')
                     or var.name.startswith('res5') or var.name.startswith('res4')
                     or var.name.startswith('res3')]
        else:
            tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('text_objseg')]
        reg_var_list = [var for var in tvars if var.op.name.find(r'DW') > 0 or var.name[-9:-2] == 'weights']
        print('Collecting variables for regularization:')
        for var in reg_var_list: print('\t%s' % var.name)
        print('Done.')

        # define loss
        self.target = tf.image.resize_bilinear(self.target_fine, [self.vf_h, self.vf_w])
        self.cls_loss_c5 = loss.weighed_logistic_loss(self.up_c5, self.target_fine, 1, 1)
        self.cls_loss_c4 = loss.weighed_logistic_loss(self.up_c4, self.target_fine, 1, 1)
        self.cls_loss_c3 = loss.weighed_logistic_loss(self.up_c3, self.target_fine, 1, 1)
        self.cls_loss_c2 = loss.weighed_logistic_loss(self.up_c2, self.target_fine, 1, 1)
        self.cls_loss = loss.weighed_logistic_loss(self.up, self.target_fine, 1, 1)
        self.cls_loss_all = 0.6 * self.cls_loss + 0.1 * self.cls_loss_c5 + 0.1 * self.cls_loss_c4 \
                            + 0.1 * self.cls_loss_c3 + 0.1 * self.cls_loss_c2
        self.reg_loss = loss.l2_regularization_loss(reg_var_list, self.weight_decay)
        self.cost = self.cls_loss_all + self.reg_loss

        # learning rate
        lr = tf.Variable(0.0, trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.start_lr, lr, self.lr_decay_step, end_learning_rate=0.00001,
                                                       power=0.9)

        # optimizer
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unknown optimizer type %s!" % self.optimizer)

        # learning rate multiplier
        grads_and_vars = optimizer.compute_gradients(self.cost, var_list=tvars)
        var_lr_mult = {}
        for var in tvars:
            if var.op.name.find(r'biases') > 0:
                var_lr_mult[var] = 2.0
            elif var.name.startswith('res5') or var.name.startswith('res4') or var.name.startswith('res3'):
                var_lr_mult[var] = 1.0
            else:
                var_lr_mult[var] = 1.0
        print('Variable learning rate multiplication:')
        for var in tvars:
            print('\t%s: %f' % (var.name, var_lr_mult[var]))
        print('Done.')
        grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v) for g, v in
                          grads_and_vars]

        # training step
        self.train_step = optimizer.apply_gradients(grads_and_vars, global_step=lr)
