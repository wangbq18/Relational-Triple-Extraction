#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/19 22:59
# @Author : Erin
# ===================
# 转化为GPU调用
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import modeling
import optimization

import os
# ==============================
# 抽取bert层特征
class bert_m(object):
    # bert层相当于提取上层特征

    def __init__(self, bert_config, init_checkpoint, input_ids, input_mask, segment_ids,is_training):
        model = modeling.BertModel(config=bert_config,
                                   is_training=is_training,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=segment_ids,
                                   use_one_hot_embeddings=False)

        self.tvars = tf.trainable_variables()
        (self.assignment_map, _) = modeling.get_assignment_map_from_checkpoint(self.tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, self.assignment_map)
        # ===================
        # 分类采用model.get_pooled_output()
        self.output_layer = model.get_sequence_output()
class model_fn(object):
    def __init__(self, bert_config, init_checkpoint, num_labels, max_len,
                 relation_hidden_size, relation_vocab,
                 learning_rate, num_train_steps,
                 num_warmup_steps,batch_size):
        self.num_labels = num_labels
        self.relation_vocab = relation_vocab
        self.initializer = initializers.xavier_initializer()
        self.relation_hidden_size=relation_hidden_size
        self.batch_size=batch_size
        self.best_dev_f1=tf.Variable(0.0,trainable=False)

        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Inputs_id")
        self.tag_s_inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="tag_label_s")  #(32, 164, 2)
        self.tag_s_ids = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                           name="tag_s_ids")  # (32, 2)
        self.mask_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Inputs_mask")
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_train')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
        # ================
        # s的识别
        self.tag_o_inputs = tf.placeholder(tf.float32, shape=[None, None,None,None], name='tag_o_label') #(32, 164, 50, 2)

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.seqlen = tf.cast(length, tf.int32)
        bert = bert_m(bert_config, init_checkpoint, self.char_inputs, self.mask_inputs, self.segment_ids,
                      self.is_training)
        self.encoder = bert.output_layer
        mask = tf.cast(self.mask_inputs, tf.float32)
        selection_loss_s = self.trans_s(relation_hidden_size, max_len)
        selection_loss_s = tf.reduce_mean(selection_loss_s,2)
        selection_loss_s = tf.reduce_sum(selection_loss_s*mask)/tf.reduce_sum(mask)

        selection_loss_op = self.trans_o(relation_hidden_size, max_len)
        election_loss_op = tf.reduce_sum(tf.reduce_mean(selection_loss_op, 3),2)
        selection_loss_op = tf.reduce_sum(election_loss_op * mask) / tf.reduce_sum(mask)

        # ==============
        #

        # select_loss, selct_tag = self.masked_loss(self.mask_inputs, self.select_logits)
        # self.select_tag = selct_tag

        self.loss =  selection_loss_s + selection_loss_op
        self.train_op = optimization.create_optimizer(self.loss, learning_rate, num_train_steps, num_warmup_steps,
                                                      False)

        # optimizer = tf.train.AdamOptimizer(0.001)
        # grads_and_vars = optimizer.compute_gradients(self.loss)
        # self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
    def masked_loss(self, mask, selection_logits):
        u_s = tf.expand_dims(mask, 2) * tf.expand_dims(mask, 1)
        v = tf.expand_dims(u_s, 2)
        selection_mask = tf.tile(v, (1, 1, 2, 1))  #
        prob = tf.nn.sigmoid(selection_logits, name='score')
        selct_tag = prob * tf.cast(selection_mask, tf.float32)
        selection_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tag_inputs, logits=selection_logits)
        selection_loss = tf.reduce_sum(selection_loss * tf.cast(selection_mask, tf.float32))  # (2,4)
        # selection_loss /= tf.reduce_sum( tf.cast(mask, tf.float32))
        return selection_loss, selct_tag

    # ======================
    # tensorflow 1.4版本使用
    # 获取s对应的向量
    def batch_gather(self, s_index, p_index):
        out = []
        for i in range(self.batch_size):
            s = tf.gather(self.encoder[i], s_index[i], axis=0)
            o = tf.gather(self.encoder[i], p_index[i], axis=0)
            s_p = tf.concat((s, o), axis=-1)
            out.append(s_p)
        return out

    def trans_s(self, relation_hidden_size, max_len):
        with tf.variable_scope('s_dense'):

            # select_v = tf.layers.dense(inputs=self.encoder, units=relation_hidden_size, activation=tf.nn.tanh)
            self.logit= tf.layers.dense(inputs=self.encoder, units=2)

            print('self.logit',self.logit)#self.logit Tensor("s_dense/pow:0", shape=(?, ?, 2), dtype=float32)
            selection_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tag_s_inputs,logits=self.logit)
            prob = tf.nn.sigmoid(self.logit, name='score_s')
            self.prob=prob ** 2
        return selection_loss


    def extrac_subject_two(self,output,subject_ids):
        """根据subject_ids从output中取出subject的向量表征
        """
        index_s=subject_ids[:, :1]#s对应的向量
        index_e=subject_ids[:, 1:]
        start = tf.batch_gather(output, index_s)# shape=(batch_size, 1, 768)
        end = tf.batch_gather(output, index_e)

        return start,end
    def trans_o(self, relation_hidden_size, max_len):
        with tf.variable_scope('o_dense'):
            # 获取对应s,p对应的向量表示


            #====================
            #方法2
            subject_s,subject_e = self.extrac_subject_two(self.encoder, self.tag_s_ids)  # shape=(?, 1536)
            subject_feature=tf.add(subject_s,subject_e)/2
            output=tf.add(self.encoder,subject_feature)


            u = tf.layers.dense(inputs=output, units=len(self.relation_vocab)*2)
            print('v',u )# Tensor("o_dense/dense_2/Tanh:0", shape=(?, ?, 100), dtype=float32

            self.logit_o=tf.reshape(u,[-1,max_len,len(self.relation_vocab),2])
            print('self.logit_o',self.logit_o)
            prob_o = tf.nn.sigmoid(self.logit_o, name='score_o')
            self.prob_o = prob_o ** 4
            print('self.prob_o',self.prob_o)
            selection_loss_o = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tag_o_inputs,
                                                                       logits=self.logit_o)

        return selection_loss_o





