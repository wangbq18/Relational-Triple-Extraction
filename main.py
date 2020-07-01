#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/30 17:39
# @Author : Erin
import os
import tensorflow as tf
import json
import numpy as np
import tokenization
import modeling
from utils import *
from model_s_po import model_fn
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string("mode", 'train',"The input datadir.", )
flags.DEFINE_string("data_dir", 'standard_format_data',"The input data dir. Should con ""for the task.")
flags.DEFINE_string( "output_dir", './model',"The output directory where the model checkpoints will be written.")

flags.DEFINE_string( "bert_config_file", 'D:/data/chinese_L-12_H-768_A-12/bert_config.json',"The config json file corresponding to the pre-trained BERT model.")
flags.DEFINE_string("init_checkpoint", 'D:/data/chinese_L-12_H-768_A-12/bert_model.ckpt',"Initial checkpoint  BERT model).")
flags.DEFINE_string("vocab_file", 'D:/data/chinese_L-12_H-768_A-12/vocab.txt',"The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool("do_lower_case", True,"Whether to lower case the input text.")

flags.DEFINE_integer("max_seq_length", 200,"The maximum total input sequence length after WordPiece tokenization.")
flags.DEFINE_integer("batch_size",2, "Total batch size for training.")
flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")
flags.DEFINE_integer("num_train_epochs", 60, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1,"Proportion of training to perform linear learning rate warmup for. ""E.g., 0.1 = 10% of training.")
#验证集评估
def evaluate_val(valid_data,model,sess,tokenizer,max_len,id2predicate):
    f1, precision, recall = evaluate(valid_data,model,sess,tokenizer,max_len,id2predicate)
    best_test_f1=model.best_dev_f1.eval()
    if f1>best_test_f1:
        tf.assign(model.best_dev_f1,f1).eval()
        print(

        ' precision: %.5f, recall: %.5f ,f1: %.5f,' % ( precision, recall,f1)
    )
    return f1>best_test_f1

def main(_):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)

    with open('../data/relation_vocab.json', 'r', encoding='utf-8') as f:

        predicate2id = json.load(f)
    id2predicate = {value: key for key, value in predicate2id.items()}

    train_data = load_data('d:/data/raw_data/train_data.json')
    valid_data=load_data('d:/data/raw_data/train_data.json')
    train_D = data_generator(train_data, FLAGS.batch_size)
    train_examples =train_data
    num_train_steps = int(
        len(train_examples) / FLAGS.batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


    model = model_fn(bert_config=modeling.BertConfig.from_json_file(FLAGS.bert_config_file),
                               init_checkpoint=FLAGS.init_checkpoint,
                               num_labels=len(predicate2id),

                               max_len=FLAGS.max_seq_length,
                               relation_hidden_size=100,
                               relation_vocab=predicate2id,
                               learning_rate=FLAGS.learning_rate,
                               num_train_steps=num_train_steps,
                               num_warmup_steps=num_warmup_steps,
                               batch_size=FLAGS.batch_size


                               )
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 模型保存路径
    checkpoint_path = os.path.join('model', 'train_model.ckpt')
    # ===============================
    # 加载bert_config文件
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state('model')

        # ===============================
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('mode_path %s' % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        # ============================
        for j in range(FLAGS.num_train_epochs):
            print('j', j)
            eval_los = 0.0
            count = 0
            step = 0
            for (batch_text, batch_token_ids, batch_mask, batch_segment_ids, batch_subject_labels,
                 batch_subject_ids, batch_object_labels) in train_D.__iter__(random=True,predicate2id=predicate2id,
                                                                             max_len=FLAGS.max_seq_length,tokenizer=tokenizer):
                # print(batch_text)
                # print(len(batch_text))
                # print(np.shape(batch_subject_labels))
                # print(batch_subject_labels[0])
                # print(np.shape(batch_object_labels))
                # print(batch_object_labels[0])
                # exit()
                count = count + 1
                feed = {model.char_inputs: batch_token_ids,
                        model.mask_inputs: batch_mask,
                        model.segment_ids: batch_segment_ids,
                        model.tag_s_inputs: batch_subject_labels,
                        model.tag_s_ids: batch_subject_ids,
                        model.tag_o_inputs: batch_object_labels,
                        model.is_training: True

                        }
                # print(sess.run(model.seqlen, feed))
                step = step + 1
                loss, _ = sess.run([model.loss, model.train_op], feed)

                eval_los = loss + eval_los
                los = eval_los / count
                if step % 20 == 0:
                    print('loss', los)

            best=evaluate_val(valid_data,model,sess,tokenizer,FLAGS.max_seq_length,id2predicate)
            if best:
                saver.save(sess, checkpoint_path)


if __name__ == "__main__":
    tf.app.run()

