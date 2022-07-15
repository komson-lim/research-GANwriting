import os
from ordered_set import T
from torch import ScriptFunction
import torch.utils.data as D
import random
import string
import cv2
import numpy as np
import re
from pairs_idx_wid_iam import wid2label_tr, wid2label_te

CREATE_PAIRS = False

IMG_HEIGHT = 64
IMG_WIDTH = 216
MAX_CHARS = 70
# NUM_CHANNEL = 15
NUM_CHANNEL = 2
EXTRA_CHANNEL = NUM_CHANNEL+1
NUM_WRITERS = 7000  # iam
NORMAL = True
OUTPUT_MAX_LEN = MAX_CHARS+2  # <GO>+groundtruth+<END>
DATASET = 'BEST'  # BEST or IAM

'''The folder of IAM word images, please change to your own one before run it!!'''
img_base = '../../Best-Handwritten-Corpus/'
# text_corpus = 'corpora_english/brown-azAZ.tr'
text_corpus = 'th.txt'
if DATASET == 'IAM':
    with open(text_corpus, 'r') as _f:
        text_corpus = _f.read().split()
else:
    with open(text_corpus, 'r', encoding='utf-8') as _f:
        text_corpus = _f.read().split('\n')

# src = 'Groundtruth/gan.iam.tr_va.gt.filter27'
# tar = 'Groundtruth/gan.iam.test.gt.filter27'
src = ['best2019-r31-with-label', 'best2019-r32-with-label', 'best2019-r33-with-label',
       'best2019-r34-with-label', 'best2019-r35-with-label', 'best2019-r36-with-label', 'best2020-r31-with-label']
tar = ['best2020-r33-1001to2640-with-label']


def labelDictionary():
    if DATASET == 'IAM':
        labels = list(string.ascii_lowercase + string.ascii_uppercase)
    elif DATASET == 'BEST':
        labels = "กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์"
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter


num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())
vocab_size = num_classes + num_tokens


def edits1(word, min_len=2, max_len=MAX_CHARS):
    "All edits that are one edit away from `word`."
    if DATASET == "IAM":
        letters = list(string.ascii_lowercase)
    elif DATASET == 'BEST':
        letters = "กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮฯะัาำิีึืุูเแโไๅๆ็่้๊๋์"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    if len(word) <= min_len:
        return random.choice(list(set(transposes + replaces + inserts)))
    elif len(word) >= max_len:
        return random.choice(list(set(deletes + transposes + replaces)))
    else:
        return random.choice(list(set(deletes + transposes + replaces + inserts)))


class Dataset_words(D.Dataset):
    def __init__(self, data_dict, oov):
        self.data_dict = data_dict
        self.oov = oov
        self.output_max_len = OUTPUT_MAX_LEN

    # word [0, 15, 27, 13, 32, 31, 1, 2, 2, 2]
    def new_ed1(self, word_ori):
        word = word_ori.copy()
        start = word.index(tokens['GO_TOKEN'])
        fin = word.index(tokens['END_TOKEN'])
        word = ''.join([index2letter[i-num_tokens]
                       for i in word[start+1: fin]])
        new_word = edits1(word)
        label = np.array(self.label_padding(new_word, num_tokens))
        return label

    def __getitem__(self, wid_idx_num):
        if DATASET == "IAM":
            words = self.data_dict[wid_idx_num]
            '''shuffle images'''
            np.random.shuffle(words)

            wids = list()
            idxs = list()
            imgs = list()
            img_widths = list()
            labels = list()

            for word in words:
                wid, idx = word[0].split(',')
                img, img_width = self.read_image_single(idx)
                label = self.label_padding(' '.join(word[1:]), num_tokens)
                wids.append(wid)
                idxs.append(idx)
                imgs.append(img)
                img_widths.append(img_width)
                labels.append(label)

            if len(list(set(wids))) != 1:
                print('Error! writer id differs')
                exit()

            final_wid = wid_idx_num
            num_imgs = len(imgs)
            if num_imgs >= EXTRA_CHANNEL:
                final_img = np.stack(imgs[:EXTRA_CHANNEL], axis=0)  # 64, h, w
                final_idx = idxs[:EXTRA_CHANNEL]
                final_img_width = img_widths[:EXTRA_CHANNEL]
                final_label = labels[:EXTRA_CHANNEL]
            else:
                final_idx = idxs
                final_img = imgs
                final_img_width = img_widths
                final_label = labels

                while len(final_img) < EXTRA_CHANNEL:
                    num_cp = EXTRA_CHANNEL - len(final_img)
                    final_idx = final_idx + idxs[:num_cp]
                    final_img = final_img + imgs[:num_cp]
                    final_img_width = final_img_width + img_widths[:num_cp]
                    final_label = final_label + labels[:num_cp]
                final_img = np.stack(final_img, axis=0)

            _id = np.random.randint(EXTRA_CHANNEL)
            img_xt = final_img[_id:_id+1]
            if self.oov:
                label_xt = np.random.choice(text_corpus)
                label_xt = np.array(self.label_padding(label_xt, num_tokens))
                label_xt_swap = np.random.choice(text_corpus)
                label_xt_swap = np.array(
                    self.label_padding(label_xt_swap, num_tokens))
            else:
                label_xt = final_label[_id]
                label_xt_swap = self.new_ed1(label_xt)

            final_idx = np.delete(final_idx, _id, axis=0)
            final_img = np.delete(final_img, _id, axis=0)
            final_img_width = np.delete(final_img_width, _id, axis=0)
            final_label = np.delete(final_label, _id, axis=0)

            return 'src', final_wid, final_idx, final_img, final_img_width, final_label, img_xt, label_xt, label_xt_swap
        elif DATASET == "BEST":
            img_path, label = self.data_dict[wid_idx_num]
            img, img_width = self.read_image_single(img_path)
            label = self.label_padding(label, num_tokens)
            wid = wid_idx_num
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img), axis=0)
            img_xt = img
            if self.oov:
                label_xt = np.random.choice(text_corpus)
                label_xt = np.array(self.label_padding(label_xt, num_tokens))
                label_xt_swap = np.random.choice(text_corpus)
                label_xt_swap = np.array(
                    self.label_padding(label_xt_swap, num_tokens))
            else:
                label_xt = label
                label_xt_swap = self.new_ed1(label_xt)
            idx = wid
            label = np.array(label)
            label = np.expand_dims(label, axis=0)
            label = np.concatenate((label, label), axis=0)
            return 'src', wid, idx, img, img_width, label, img_xt, label_xt, label_xt_swap

    def __len__(self):
        return len(self.data_dict)

    def read_image_single(self, file_name):
        if DATASET == "IAM":
            url = os.path.join(img_base, file_name + '.png')
        elif DATASET == 'BEST':
            url = file_name
        img = cv2.imread(url, 0)

        if img is None and os.path.exists(url):
            # image is present but corrupted
            return np.zeros((IMG_HEIGHT, IMG_WIDTH)), 0
        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(
            img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        img = img/255.  # 0-255 -> 0-1

        img = 1. - img
        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
            outImg[:, :img_width] = img
        outImg = outImg.astype('float32')

        mean = 0.5
        std = 0.5
        outImgFinal = (outImg - mean) / std
        return outImgFinal, img_width

    def label_padding(self, labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        new_label_len.append(len(ll)+2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        num = self.output_max_len - len(ll)
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num)  # replace PAD_TOKEN
        return ll


def loadData(oov):
    if DATASET == 'IAM':
        gt_tr = src
        gt_te = tar
        with open(gt_tr, 'r') as f_tr:
            data_tr = f_tr.readlines()
            data_tr = [i.strip().split(' ') for i in data_tr]
            tr_dict = dict()
            for i in data_tr:
                wid = i[0].split(',')[0]
                if wid not in tr_dict.keys():
                    tr_dict[wid] = [i]
                else:
                    tr_dict[wid].append(i)
            new_tr_dict = dict()
            if CREATE_PAIRS:
                create_pairs(tr_dict)
            for k in tr_dict.keys():
                new_tr_dict[wid2label_tr[k]] = tr_dict[k]

        with open(gt_te, 'r') as f_te:
            data_te = f_te.readlines()
            data_te = [i.strip().split(' ') for i in data_te]
            te_dict = dict()
            for i in data_te:
                wid = i[0].split(',')[0]
                if wid not in te_dict.keys():
                    te_dict[wid] = [i]
                else:
                    te_dict[wid].append(i)
            new_te_dict = dict()
            if CREATE_PAIRS:
                create_pairs(te_dict)
            for k in te_dict.keys():
                new_te_dict[wid2label_te[k]] = te_dict[k]
    elif DATASET == "BEST":
        train_folders = src
        test_folders = tar

        new_tr_dict = dict()
        i = 0
        for file_name in train_folders:
            for file in os.listdir(os.path.join(img_base, file_name)):
                if file.endswith('.label'):
                    try:
                        with open(os.path.join(img_base, file_name, file), 'r', encoding='cp874') as f:
                            for line in f:
                                temp = line.split()
                                img_path = temp[0]
                                img_path = os.path.join(
                                    img_base, file_name, img_path)
                                label = "".join(temp[1:])
                                new_tr_dict[i] = (img_path, label)
                                i += 1
                    except UnicodeDecodeError:
                        with open(os.path.join(img_base, file_name, file), 'r', encoding='utf_16') as f:
                            for line in f:
                                temp = line.split()
                                if len(temp) < 2:
                                    continue
                                img_path = temp[0]
                                img_path = os.path.join(
                                    img_base, file_name, img_path)
                                label = "".join(temp[1:])
                                new_tr_dict[i] = (img_path, label)
                                i += 1
                    break
        new_te_dict = dict()
        for file_name in test_folders:
            for file in os.listdir(os.path.join(img_base, file_name)):
                if file.endswith('.label'):
                    try:
                        with open(os.path.join(img_base, file_name, file), 'r', encoding='cp874') as f:
                            for line in f:
                                temp = line.split()
                                img_path = temp[0]
                                img_path = os.path.join(
                                    img_base, file_name, img_path)
                                label = "".join(temp[1:])
                                new_tr_dict[i] = (img_path, label)
                                i += 1
                    except UnicodeDecodeError:
                        with open(os.path.join(img_base, file_name, file), 'r', encoding='utf_16') as f:
                            for line in f:
                                temp = line.split()
                                if len(temp) < 2:
                                    continue
                                img_path = temp[0]
                                label = "".join(temp[1:])
                                img_id = int(re.findall(
                                    r"-(.*).png", img_path)[0])
                                if img_id < 1001:
                                    img_path = os.path.join(
                                        img_base, "best2020-r33-1to1000", img_path)
                                else:
                                    img_path = os.path.join(
                                        img_base, file_name, img_path)
                                new_te_dict[i] = (img_path, label)
                                i += 1
                    break
    data_train = Dataset_words(new_tr_dict, oov)
    data_test = Dataset_words(new_te_dict, oov)
    return data_train, data_test


def create_pairs(ddict):
    num = len(ddict.keys())
    label2wid = list(zip(range(num), ddict.keys()))
    print(label2wid)


if __name__ == '__main__':
    pass
