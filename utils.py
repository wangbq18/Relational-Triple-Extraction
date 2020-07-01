#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/30 17:42
# @Author : Erin
import json
import numpy as np
import tokenization
import  unicodedata
from tqdm import tqdm
def load_data(file_name):
    D=[]
    with open(file_name,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            l=json.loads(line)
            D.append({'text':l['text'],
                      'spo_list':[(spo['subject'],spo['predicate'],spo['object']) for spo in l['spo_list']]
                      })
    return D
def search(pattern,sequence):
    n=len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i+n]==pattern:
            return i
    return -1
class DataGenerator(object):
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d
def convert_bert(text, max_seq_length, tokenizer):
    text = tokenization.convert_to_unicode(text)
    tokens = tokenizer.tokenize(text)
    if len(tokens) >= max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    segment_ids== [0] * len(input_ids)

    #在[SEP]之后补0
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    # print(input_ids)#[101, 7032, 4989, 148, 122, 121, 127, 2792, 3118, 2898, 4638, 161, 146, 1305, 2810, 2245, 2159, 7030, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # exit()
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids,input_mask,segment_ids

class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False,predicate2id=None,max_len=None,tokenizer=None):
        batch_text,batch_token_ids, batch_mask,batch_segment_ids =[], [], [],[]
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):
            text = tokenization.convert_to_unicode(d['text'])
            tokens = tokenizer.tokenize(text)
            tokens.insert(0, '[CLS]')
            tokens.append('[SEP]')

            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenization.convert_to_unicode(s)
                s = tokenizer.tokenize(s)
                # s = self.tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenization.convert_to_unicode(o)
                o = tokenizer.tokenize(o)
                s_idx = search(s, tokens)
                o_idx = search(o, tokens)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)

                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(token_ids)
                segment_ids = [0] * len(token_ids)
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append( token_ids)
                batch_text.append(d['text'])
                batch_mask.append(input_mask)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids,max_len)
                    batch_mask = sequence_padding(batch_mask,max_len)
                    batch_segment_ids = sequence_padding(batch_segment_ids,max_len)
                    batch_subject_labels = sequence_padding(
                        batch_subject_labels,max_len
                    )
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels,max_len)
                    yield batch_text,batch_token_ids,batch_mask, batch_segment_ids,batch_subject_labels,\
                          batch_subject_ids, batch_object_labels


                    batch_text,batch_token_ids,batch_mask, batch_segment_ids = [], [],[],[]
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []

def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')

def _is_special(ch):
    """判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

def stem(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token
def rematch(text, tokens,_do_lower_case):
    """给出原始的text和tokenize后的tokens的映射关系
    """
    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        if _do_lower_case:
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
        ch = ''.join([
            c for c in ch
            if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
        ])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))
    text, token_mapping, offset = normalized_text, [], 0
    for token in tokens:
        if _is_special(token):
            token_mapping.append([])
        else:
            token = stem(token)
            start = text[offset:].index(token) + offset
            end = start + len(token)
            token_mapping.append(char_mapping[start:end])
            offset = end

    return token_mapping
#=====================
#以下两个函数是为了把token之后的句子与原始句子对应
def deal_replace(tokens,text):
    new_token=[]
    length=0
    for w_all in tokens:
        w_all=w_all.replace('##','')
        ws=''
        for w in w_all:
            for s in w:
                ws=ws+text[length]
                length=length+1
        new_token.append(ws)
    return new_token
def deal_unk(tokens,text):
    new_token = []
    length = 0
    for w in tokens:
        w = w.replace('##', '')
        if w=='[UNK]':
            length = length + 1
            index=text[length-1]
            new_token.append(index)
        else:
            length=length+len(w)
            new_token.append(w)
    return new_token
def closed_single(text,model,sess,tokenizer,max_len,id2predicate):
    text=tokenization.convert_to_unicode(text)
    tokens=tokenizer.tokenize(text)
    new_tokens=deal_unk(tokens,text)
    new_tokens=deal_replace(new_tokens,text)
    new_tokens.insert(0,'[CLS]')
    new_tokens.append('[SEP]')

    tokens.insert(0,'[CLS]')
    tokens.append('[SEP]')
    mapping=rematch(text,new_tokens,False)
    token_ids=tokenizer.convert_tokens_to_ids(tokens)
    input_mask=[1]*len(token_ids)
    segment_ids=[0]*len(token_ids)
    token_ids=sequence_padding([token_ids],max_len)
    input_mask=sequence_padding([input_mask],max_len)
    segment_ids=sequence_padding([segment_ids],max_len)
    feed_dict={model.char_inputs:token_ids,
               model.mask_inputs:input_mask,
               model.segment_ids:segment_ids,
               model.is_training:False}
    subject_preds=sess.run(model.prob,feed_dict)
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        input_mask = np.repeat(input_mask, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        feed_dict = {model.char_inputs: token_ids,
                     model.mask_inputs: input_mask,
                     model.segment_ids: segment_ids,
                     model.tag_s_ids:subjects,
                     model.is_training: False}
        object_preds = sess.run(model.prob_o,feed_dict)
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.6)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2 and subject[0]<len(mapping) \
                            and _start<len(mapping) and _end<len(mapping) and subject[1]<len(mapping) and _start!=0:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []
class SPO(tuple):
    def __init__(self, spo):
        self.spox = (
            tuple(spo[0]),
            spo[1],
            tuple(spo[2]),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox

def sequence_padding(inputs, length=None, padding=0):
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)
def evaluate(data,model,sess,tokenizer,max_len,id2predicate):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in closed_single(d['text'],model,sess,tokenizer,max_len,id2predicate)])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )

    pbar.close()
    return f1, precision, recall
