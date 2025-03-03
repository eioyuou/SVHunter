import pysam
import numpy as np
import math
import time
from statistics import mean
import json
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from os import listdir
from os.path import isfile, join
from tensorflow.keras import backend as K
import multiprocessing
from multiprocessing import Pool

from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict
from os import listdir
from os.path import isfile, join
from math import log10
import math
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binom
# %%

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
#tf.compat.v1.Session(config=config)


# %%
def cbam_block(cbam_feature, ratio=7, kernel_size=(2, 20)):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature, kernel_size)
    return cbam_feature


def channel_attention(input_feature, ratio=7):
    channel = input_feature.shape[-1]
    filters = max(1, int(channel // ratio))

    shared_layer_one_avg = tf.keras.layers.Dense(filters,
                                                 activation='relu',
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
    shared_layer_two_avg = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')

    shared_layer_one_max = tf.keras.layers.Dense(filters,
                                                 activation='relu',
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')
    shared_layer_two_max = tf.keras.layers.Dense(channel,
                                                 kernel_initializer='he_normal',
                                                 use_bias=True,
                                                 bias_initializer='zeros')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one_avg(avg_pool)
    avg_pool = shared_layer_two_avg(avg_pool)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
    max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one_max(max_pool)
    max_pool = shared_layer_two_max(max_pool)

    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])


def spatial_attention(input_feature, kernel_size):
    channel = input_feature.shape[-1]
    cbam_feature = input_feature

    avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Conv2D(filters=1,
                                          kernel_size=kernel_size,
                                          strides=1,
                                          padding='same',
                                          activation='sigmoid',
                                          kernel_initializer='he_normal',
                                          use_bias=False)(concat)

    return tf.keras.layers.multiply([input_feature, cbam_feature])


def cnn_model():
    inputs = tf.keras.Input(shape=(200, 20, 1))
    x = tf.keras.layers.Conv2D(128, kernel_size=(2, 20), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPool2D((2, 1))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(2, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 1))(x)
    x = cbam_block(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(2, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 1))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(2, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 1))(x)
    x = cbam_block(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(2, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 1))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(2, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 1))(x)
    x = cbam_block(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(2, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs, x)
    return model


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_height, patch_width):
        super().__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_height, self.patch_width, 1],
            strides=[1, self.patch_height, self.patch_width, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(
            patches,
            (
                batch_size,
                -1,
                self.patch_height * self.patch_width * tf.shape(images)[-1],
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_height": self.patch_height,
            "patch_width": self.patch_width
        })
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.expand_dims(tf.range(start=0, limit=self.num_patches, delta=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded


def init_model():
    inputs = tf.keras.Input((2000, 20, 1))
    inputss = tf.keras.layers.Reshape((10, 200, 20, 1))(inputs)
    cnn_layer_object = cnn_model()
    encoded_frames = tf.keras.layers.TimeDistributed(cnn_layer_object)(inputss)
    encoded_patches = PatchEncoder(10, 100)(encoded_frames)
    for _ in range(7):
        x1 = tf.keras.layers.LayerNormalization()(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=32, key_dim=32, dropout=0.3
        )(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = tf.keras.layers.LayerNormalization()(x2)
        x3 = tf.keras.layers.Dense(units=100, activation="gelu")(x3)
        x3 = tf.keras.layers.Dropout(0.3)(x3)
        x3 = tf.keras.layers.Dense(units=100, activation="gelu")(x3)
        x3 = tf.keras.layers.Dropout(0.3)(x3)
        encoded_patches = layers.Add()([x3, x2])

    hidden_layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    hidden_layer = tf.keras.layers.Flatten()(hidden_layer)
    hidden_layer = tf.keras.layers.Dense(units=128, activation="relu")(hidden_layer)
    hidden_layer = tf.keras.layers.Dropout(0.4)(hidden_layer)
    hidden_layer = tf.keras.layers.Dense(units=128, activation="relu")(hidden_layer)
    hidden_layer = tf.keras.layers.Dropout(0.4)(hidden_layer)
    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(hidden_layer)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4),
        metrics=[
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model


model = init_model()


# %%
def c_pos(cigar, refstart):
    number = ''
    readstart = None
    readend = None
    refend = None
    readloc = 0
    refloc = refstart

    for c in cigar:
        if c.isdigit():
            number += c
        else:
            number = int(number)


            if c in ['S', 'H'] and readstart is None:
                
                if c == 'S':
                    readstart = readloc + number
                readloc += number
            else:
                if readstart is None and c in ['M', 'I', '=', 'X']:
                    readstart = readloc

                if c in ['M', 'I', '=', 'X']:
                    readloc += number

                if c in ['M', 'D', 'N', '=', 'X']:
                    refloc += number

                
                if c in ['H', 'S'] and readstart is not None:
                    readend = readloc
                    refend = refloc

            number = ''

    if readend is None:
        readend = readloc

    if refend is None:
        refend = refloc

    return refstart, refend, readstart, readend


# %%
dic_starnd = {1: '+', 2: '-'}
signal = {1 << 2: 0, \
          1 >> 1: 1, \
          1 << 4: 2, \
          1 << 11: 3, \
          1 << 4 | 1 << 11: 4}


def detect_flag(Flag):
    back_sig = signal[Flag] if Flag in signal else 0
    return back_sig


# %%
def splitreadlist(readsp):
    read = readsp
    sv_list = []
    aligned_length = read.reference_length
    process_signal = detect_flag(read.flag)

    if (process_signal == 1 or process_signal == 2):
        if read.is_reverse:
            qry_start = read.infer_read_length() - read.query_alignment_end
            qry_end = read.infer_read_length() - read.query_alignment_start
        else:
            qry_start = read.query_alignment_start
            qry_end = read.query_alignment_end
        qry_reference_start = read.reference_start
        qry_reference_end = read.reference_end
        strand_ = '-' if read.is_reverse else '+'
        sv_list.append([qry_start, qry_end, qry_reference_start, qry_reference_end, read.reference_name, strand_])
        rawsalist = read.get_tag('SA').split(';')
        for sa in rawsalist[:-1]:
            sainfo = sa.split(',')
            tmpcontig, tmprefstart, strand, cigar, sup_mapq = sainfo[0], int(sainfo[1]), sainfo[2], sainfo[3], int(
                sainfo[4])
            refstart_2, refend_2, readstart_2, readend_2 = c_pos(cigar, tmprefstart)

            # print(refstart_2, refend_2, readstart_2, readend_2, strand,read.query_length)
            if strand == '-' and sup_mapq >= 0:
                readstart = read.query_length - readend_2
                readend = read.query_length - readstart_2
                sv_list.append([readstart, readend, refstart_2, refend_2, tmpcontig, strand])
            elif strand == '+' and sup_mapq >= 0:
                sv_list.append([readstart_2, readend_2, refstart_2, refend_2, tmpcontig, strand])

    if len(sv_list) == 0:
        return []

    return sv_list


# %%
def feature_record(alignment_current, alignment_next, ins_tra_flag=False):
    distance_on_read = alignment_next[0] - alignment_current[1]
    if alignment_current[-1] == '+':
        distance_on_reference = alignment_next[2] - alignment_current[3]
        if alignment_next[-1] == '-':  # INV:+-
            if alignment_current[3] > alignment_next[3]:
                distance_on_reference = alignment_next[3] - alignment_current[2]
            else:
                distance_on_reference = alignment_current[3] - alignment_next[2]
    else:
        distance_on_reference = alignment_current[2] - alignment_next[3]
        if alignment_next[-1] == '+':  # INV:-+
            if alignment_current[3] > alignment_next[3]:
                distance_on_reference = alignment_next[3] - alignment_current[2]
            else:
                distance_on_reference = alignment_current[3] - alignment_next[2]
    deviation = distance_on_read - distance_on_reference
    chr_ = 1 if alignment_current[-2] == alignment_next[-2] else 0
    orientation = 1 if alignment_current[-1] == alignment_next[-1] else 0
    return (alignment_current, alignment_next, chr_, orientation, distance_on_read, distance_on_reference,
            deviation, ins_tra_flag)


# %%
def feature_read_segement(svlist):
    sg_list = []
    sorted_alignment_list = sorted(svlist, key=lambda aln: (aln[0], aln[1]))
    for index in range(len(sorted_alignment_list) - 1):
        sg_list.append(feature_record(sorted_alignment_list[index], sorted_alignment_list[index + 1]))
    if len(svlist) >= 3 and sorted_alignment_list[0][-2] != sorted_alignment_list[1][-2]:
        sg_list.append(feature_record(sorted_alignment_list[0], sorted_alignment_list[-1], ins_tra_flag=True))

    return sg_list


# %%
def analyze_read_segments(read, segement_data, candidate, startt, endd, alignscore):
    sv_signatures = []
    min_sv_size = 40
    segment_overlap_tolerance = 5
    read_name = read.query_name
    for sv_sig in segement_data:
        alignment_current = sv_sig[0]
        alignment_next = sv_sig[1]
        # print(alignment_current, alignment_next)
        ref_chr = alignment_current[-2]
        chr_, orientation, distance_on_read, distance_on_reference, deviation, long_ins = sv_sig[2:]
        if chr_ == 1:
            if orientation == 1:
                if distance_on_reference >= -min_sv_size or long_ins:  # INS
                    if deviation > 0:  # INS
                        if alignment_current[-1] == '+':
                            start = (alignment_current[3] + alignment_next[2]) // 2 if not long_ins else min(
                                alignment_current[3], alignment_next[2])
                        else:
                            start = (alignment_current[2] + alignment_next[3]) // 2 if not long_ins else min(
                                alignment_current[2], alignment_next[3])
                        end = start + deviation
                        if end - start < min_sv_size:
                            continue
                        insertion_seq = 'A'
                        # insertion_seq = bam.fetch(ref_chr, start, end).upper()
                        if startt <= start and endd >= start:
                            candidate.append(
                                [start, deviation, read_name, insertion_seq, 'split', 'INS', alignscore, ref_chr])
                        # sv_signatures.append(SignatureInsertion(*sv_sig))
                    elif deviation < 0:  # DEL
                        if alignment_current[-1] == '+':
                            start = alignment_current[3]
                        else:
                            start = alignment_next[3]
                        end = start - deviation
                        if end - start < min_sv_size:
                            continue
                        if startt <= start and endd >= start:
                            candidate.append(
                                [start, -deviation, read_name, 'None', 'split', 'DEL', alignscore, ref_chr])

                    else:
                        continue
                else:

                    if alignment_current[-1] == '+':
                        start = alignment_next[2]
                        end = alignment_current[3]
                    else:
                        start = alignment_current[2]
                        end = alignment_next[3]


                    deviation = end - start
                    candidate.append([start, deviation, read_name, 'None', 'split', 'DUP', alignscore, ref_chr])

            else:  # INV
                if alignment_current[-1] == '+':  # +-
                    if alignment_next[2] - alignment_current[
                        3] >= -segment_overlap_tolerance:
                        start = alignment_current[3]
                        end = alignment_next[3]
                        if startt <= start and endd >= start:
                            deviation = end - start
                            candidate.append([start, deviation, read_name, 'None', 'split', 'INV', alignscore, ref_chr])
                        # sv_sig = (ref_chr, start, end, "suppl", read_name, "left_fwd")
                    elif alignment_current[2] - alignment_next[
                        3] >= -segment_overlap_tolerance:
                        start = alignment_next[3]
                        end = alignment_current[3]
                        if startt <= start and endd >= start:
                            # sv_sig = (ref_chr, start, end, "suppl", read_name, "left_rev")
                            deviation = end - start
                            candidate.append([start, deviation, read_name, 'None', 'split', 'INV', alignscore, ref_chr])
                    else:
                        continue
                else:  # -+
                    if alignment_next[2] - alignment_current[
                        3] >= -segment_overlap_tolerance:
                        start = alignment_current[2]
                        end = alignment_next[2]
                        if startt <= start and endd >= start:
                            deviation = end - start
                            candidate.append([start, deviation, read_name, 'None', 'split', 'INV', alignscore, ref_chr])

                    elif alignment_current[2] - alignment_next[
                        3] >= -segment_overlap_tolerance:
                        start = alignment_next[2]
                        end = alignment_current[2]
                        if startt <= start and endd >= start:
                            deviation = end - start
                            candidate.append([start, deviation, read_name, 'None', 'split', 'INV', alignscore, ref_chr])

                    else:
                        continue

        else:
            ref_chr_next = alignment_next[-2]

            if orientation == 1:

                if alignment_current[-1] == '+':
                    if ref_chr < ref_chr_next:
                        start = alignment_current[3]
                        end = alignment_next[2]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[2]
                        end = alignment_current[3]

                    candidate.append([start, read_name, 'fwd', ref_chr_next, end, 'fwd', 'TRA', alignscore, ref_chr])


                else:
                    if ref_chr < ref_chr_next:
                        start = alignment_current[2]
                        end = alignment_next[3]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[3]
                        end = alignment_current[2]

                    candidate.append([start, read_name, 'rev', ref_chr_next, end, 'rev', 'TRA', alignscore, ref_chr])
            else:
                if alignment_current[-1] == '+':
                    if ref_chr < ref_chr_next:
                        start = alignment_current[3]
                        end = alignment_next[3]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[3]
                        end = alignment_current[3]

                    candidate.append([start, read_name, 'fwd', ref_chr_next, end, 'rev', 'TRA', alignscore, ref_chr])
                else:  # -+
                    if ref_chr < ref_chr_next:
                        start = alignment_current[2]
                        end = alignment_next[2]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[2]
                        end = alignment_current[2]
                    candidate.append([start, read_name, 'rev', ref_chr_next, end, 'fwd', 'TRA', alignscore, ref_chr])


# %%
def median(lst):
    n = len(lst)
    sorted_lst = sorted(lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    else:
        return sorted_lst[mid]


# %%
def filter_entries(entries):
    best_entries = {}

    for entry in entries:
        start, end, length, name = entry
        start, end, length, name = float(start), float(end), float(length), str(name)

        if name in best_entries:
            if start > best_entries[name][0]:
                best_entries[name] = [start, end, length, name]
        else:
            best_entries[name] = [start, end, length, name]

    filtered_entries = [entry[:3] for entry in best_entries.values()]
    return filtered_entries


# %%
def merge_intervals(intervals):

    merged = [intervals[0]]

    # 遍历剩余的区间
    for current in intervals[1:]:

        last = merged[-1]

        if current[1] == last[2]:
            last[2] = current[2]

        else:
            merged.append(current)

    return merged


def mergecigar_del(infor):
    data = []
    i = 0
    while i >= 0:
        count = 0
        if i > (len(infor) - 1):
            break
        for j in range(i + 1, len(infor)):
            # print(i,j)
            if abs(infor[j][0] - infor[i][1]) <= 150:
                count = count + 1
                infor[i][1] = infor[j][1]
        # print(infor[i],count)
        lenth = abs(infor[i][0] - infor[i][1])
        data.append([infor[i][0], infor[i][1], lenth])

        if count == 0:
            i += 1
        else:
            i += (count + 1)
    return data


def mergecigar_ins(infor):
    data = []
    i = 0
    while i >= 0:
        count = 0
        if i > (len(infor) - 1):
            break
        lenth = infor[i][2]
        for j in range(i + 1, len(infor)):
            # print(i,j)
            if abs(infor[j][1] - infor[i][1]) <= 150:
                count = count + 1
                infor[i][1] = infor[j][0]  # 改[0]0
                lenth = lenth + infor[j][2]  # + abs(infor[j][0] - infor[i][0])

        data.append([infor[i][0], infor[i][0] + 1, lenth])

        if count == 0:
            i += 1
        else:
            i += (count + 1)
    return data


def cigarread(read, candidata, start, end, alignscore):
    aligned_length = read.reference_length
    read_name = read.query_name
    chr_name = read.reference_name
    data_del = []
    data_ins = []
    if aligned_length == None:
        aligned_length = 0
    cigar_del = []
    cigar_ins = []
    sta = read.reference_start
    for ci in read.cigartuples:

        if ci[0] in [0, 7, 8]:
            sta += ci[1]
        elif ci[0] == 2:  # del
            if ci[1] >= 40 and start <= sta + ci[1] and end >= sta:
                cigar_del.append([sta, sta + ci[1], ci[1]])
            sta += ci[1]
        elif ci[0] == 1:  # ins

            if ci[1] >= 40 and start <= sta + ci[1] and end >= sta:
                cigar_ins.append([sta, sta, ci[1]])
    if len(cigar_del) != 0:
        cigar_del.sort(key=lambda x: x[0])
        cigar_del = mergecigar_del(cigar_del)
        data_del.extend(cigar_del)

    if len(cigar_ins) != 0:
        # print(cigar_ins)
        # print('+++++++++++++++++++++++++++++++')
        cigar_ins.sort(key=lambda x: x[0])
        cigar_ins = mergecigar_ins(cigar_ins)
        # print(long_ins)
        # print('===============================')
        data_ins.extend(cigar_ins)
    data_del = np.array(data_del)
    data_ins = np.array(data_ins)
    for del_cigar in data_del:
        candidata.append([del_cigar[0],
                          del_cigar[2],
                          read_name,
                          'None',
                          'cigar',
                          'DEL',
                          alignscore,
                          chr_name])
    for ins_ci in data_ins:
        candidata.append([ins_ci[0],
                          ins_ci[2],
                          read_name,
                          'None',
                          'cigar',
                          'INS',
                          alignscore,
                          chr_name])


# %%

# %%

def cluster_by_length(lengths, threshold_ratio=0.7):
    sorted_lengths = sorted(lengths, key=lambda x: int(x[2]))
    mean_length = int(sorted_lengths[len(sorted_lengths) // 2][2])

    clusters = []
    current_cluster = [sorted_lengths[0]]

    for length in sorted_lengths[1:]:
        # print(length)
        if abs(int(current_cluster[-1][2]) - int(length[2])) <= mean_length * threshold_ratio:
            current_cluster.append(length)
        else:
            clusters.append(current_cluster)
            current_cluster = [length]

    clusters.append(current_cluster)

    return clusters


from sklearn.cluster import MeanShift


def mean_shift_def(chr_name, X, svtype, error_mean):
    if X.size == 0:
        return []
    if error_mean > 0.1 and (svtype == 'DEL'):
        bandwidths = 1000
    elif error_mean > 0.1 and (svtype == 'INS'):
        bandwidths = 300
    elif error_mean > 0.1 and (svtype == 'DUP'):
        bandwidths = 500
    elif error_mean > 0.1 and (svtype == 'INV'):
        bandwidths = 500
    else:
        bandwidths = 1500

    XX = np.array([[int(x[0]), int(x[1])] for x in X])
    # print(XX)

    clust = MeanShift(bandwidth=bandwidths)
    clust.fit(XX)
    cluster_dict = defaultdict(list)
    # 提取聚类
    labels = clust.labels_
    # print(labels)
    for i in range(len(labels)):
        cluster_dict[labels[i]].append(X[i])
    signature = []
    cluster_dictt = cluster_dict.values()
    cluter_length = []
    # print(signature)

    # print(signature.shape)
    for sig in cluster_dictt:
        # print(len(sig))
        if len(sig) > 1:
            # print(sig)
            cluter_length.append(cluster_by_length(sig))
        else:
            cluter_length.append([sig])

    data_cluster = []
    for lenth_one1 in cluter_length:

        for lenth_one in lenth_one1:

            readname_list = []
            data = np.array(lenth_one)
            # print(len(data))

            if len(data) == 1:
                # print(data)
                data_cluster.append(
                    [chr_name, data[0][0], data[0][2], len(data), svtype, './.', data[0][3], data[0][4]])
                continue
            # print(data)
            data = sorted(data, key=lambda x: (x[0]))

            data = [item for item in data if int(item[2]) >= 50]

            if len(data) == 0:
                continue

            first_elements = [int(item[0]) for item in data]
            third_elements = [int(item[2]) for item in data]
            start = math.ceil(median(first_elements))
            length = math.ceil(median(third_elements))
            # print(length)
            readname_list = [item[3] for item in data]
            readtype = [item[4] for item in data]
            readalignscore = [float(item[5]) for item in data]
            readalignscore_mean = np.mean(readalignscore)
            # print(readname_list)
            # print('-------------')
            data_cluster.append([chr_name, start, length, len(data), svtype, './.', readname_list, readtype])

    data_cluster = sorted(data_cluster, key=lambda center: int(center[1]))
    # print(data_cluster)
    return data_cluster


# %%
def alignment_quality_score_new2(mapq, error_rate, num_of_mismatch, read_length):

    mismatch_rate = num_of_mismatch / read_length


    mapq_reward = math.log10(mapq + 1) / 2


    error_penalty = 1 - error_rate


    mismatch_penalty = 1 - mismatch_rate


    alignment_score = (mapq_reward * error_penalty * mismatch_penalty) * 100

    return alignment_score


# %%
import math


def sv_filter_score(mapq, num_of_mismatch, read_length, num_of_insertion, num_of_deletion):

    mismatch_rate = num_of_mismatch / read_length


    insertion_rate = num_of_insertion / read_length
    deletion_rate = num_of_deletion / read_length


    total_indel_length = num_of_insertion + num_of_deletion


    mapq_penalty = math.exp(-0.05 * mapq)


    mismatch_penalty = 1 - math.exp(-5 * mismatch_rate)


    insertion_reward = math.exp(3 * insertion_rate)
    deletion_reward = math.exp(3 * deletion_rate)


    indel_length_reward = math.log10(total_indel_length + 1)


    filter_score = (
                               mapq_penalty * mismatch_penalty * insertion_reward ** 2 * deletion_reward ** 2 * indel_length_reward) * 100

    return filter_score


# %%
import math


def sv_filter_score_v7(mapq, num_of_mismatch, read_length, num_of_insertion, num_of_deletion, insertion_positions,
                       deletion_positions):
    mismatch_rate = num_of_mismatch / read_length

    mapq_score = math.log(1 + mapq)
    mismatch_penalty = 1 - math.exp(-5 * mismatch_rate)


    def concentration_score(positions, read_length):
        if len(positions) <= 1:
            return 1

        positions = sorted(positions)
        max_gap = max(positions[i + 1] - positions[i] for i in range(len(positions) - 1))

        return read_length / (len(positions) * max_gap)

    insertion_concentration = concentration_score(insertion_positions, read_length)
    deletion_concentration = concentration_score(deletion_positions, read_length)

    concentration_score = insertion_concentration * deletion_concentration

    filter_score = mapq_score * mismatch_penalty * concentration_score

    return filter_score * 1000


def get_indel_positions_from_read(read):
    insertion_positions = []
    deletion_positions = []

    if read.is_unmapped:
        return insertion_positions, deletion_positions

    ref_pos = read.reference_start
    for cigar_op, length in read.cigartuples:
        if cigar_op == 1:  # Insertion
            insertion_positions.append(ref_pos)
        elif cigar_op == 2:  # Deletion
            for i in range(length):
                deletion_positions.append(ref_pos + i)
        if cigar_op != 1:
            ref_pos += length

    return insertion_positions, deletion_positions


# %%
def mergedeleton_long(pre, bamfile, ssstart, chr_name, index):
    data = []
    candidate = []
    breakpointall = []
    del_infor = []
    ins_infor = []
    inv_infor = []
    dup_infor = []
    del_inforesult = []
    ins_inforesult = []

    for i in range(len(pre)):
        if pre[i] > 0.5:
            data.append([chr_name, index[i], index[i] + 2000])
        ssstart += 2000
    if len(data) == 0:
        return [], [], [], []
    data = merge_intervals(data)

    for chr_name, start, end in data:
        # print(start,end)
        # max_mapq = mapq_max(start,end,bamfile,chr_name)
        for read in bamfile.fetch(chr_name, start, end):
            nm = read.get_tag('NM')
            if read.has_tag('SA'):
                split_read = splitreadlist(read)
                read_mul_algin = len(split_read) - 1
            else:
                read_mul_algin = 1
            CIGAR_DEL = 2
            CIGAR_INS = 1
            CIGAR_CLIP = [4, 5]
            K_MM = 1.0
            indel = [b for a, b in read.cigartuples if a in [CIGAR_INS, CIGAR_DEL]]

            num_of_mismatch = nm - sum(indel)
            total_segment_length = sum(
                [b for a, b in read.cigartuples if a not in CIGAR_CLIP + [CIGAR_DEL]])  

            mm_rate = num_of_mismatch * K_MM / total_segment_length  
            error_rate = nm * K_MM / total_segment_length

            alignscore = alignment_quality_score(read.mapq, error_rate, num_of_mismatch, mm_rate)

            if read.is_supplementary or read.is_secondary or read.is_unmapped:
                continue
            #if alignscore <= 60 or read.mapq >= 20:
            if read.mapq >= 20:
                cigarread(read, candidate, start, end, alignscore)
                if read.has_tag('SA'):
                    split_read = splitreadlist(read)
                    # analysis_split_read(split_read, candidate, read)
                    analyze_read_segments(read, feature_read_segement(split_read), candidate, start, end, alignscore)
            # print(read.query_name)
            breakpointall.append([error_rate, read.reference_name])
    return candidate, breakpointall


# %%
def cluster_translocations(candidates, max_distance=1000):

    sorted_candidates = sorted(candidates, key=lambda x: (x[1], x[0], x[3], x[4]))

    clusters = []
    data_cluster = []
    for candidate in sorted_candidates:
        start, read_name, current_direction, ref_chr_next, end, translocation_direction, variant_type, align_score, current_chr = candidate


        found_cluster = False
        for cluster in clusters:
            if ((ref_chr_next == cluster[0][3] and current_chr == cluster[0][8])) and \
                    (abs(start - cluster[0][0]) <= max_distance and abs(end - cluster[0][4]) <= max_distance):
                cluster.append(candidate)
                found_cluster = True
                break


        if not found_cluster:
            clusters.append([candidate])

    for transs in clusters:
        if len(transs) == 1:
            data_cluster.append(
                [transs[0][-1], transs[0][0], transs[0][2], transs[0][3], transs[0][4], transs[0][5], 'BND', 1, '0/0',
                 transs[0][1]])
            # data_cluster.append([transs[0][3], transs[0][4], transs[0][5], transs[0][-2], transs[0][0], transs[0][9], 'BND', 1, '0/0', transs[0][1]])
            continue
        first_chr_name = transs[0][-1]
        first_direction = transs[0][2]
        second_chr_name = transs[0][3]
        second_direction = transs[0][5]
        first_start_elements = [int(item[0]) for item in transs]
        start1 = math.ceil(median(first_start_elements))
        second_start_elements = [int(item[4]) for item in transs]
        start2 = math.ceil(median(second_start_elements))
        read_name_list = [item[1] for item in transs]
        data_cluster.append(
            [first_chr_name, start1, first_direction, second_chr_name, start2, second_direction, 'BND', len(transs),
             '0/0', read_name_list])
        # data_cluster.append([second_chr_name, start2, second_direction, first_chr_name, start1, first_direction, 'BND', len(transs), '0/0', read_name_list])

    data_cluster = sorted(data_cluster, key=lambda x: (x[0], x[1], x[3], x[4]))

    return data_cluster


# %%

# %%
def analysis_candidate(candidatem, chr_name):
    resultlist = []
    error_mean_all = []
    signal_dict = defaultdict(list)
    signal_dict_error = defaultdict(list)

    for sv_candidate in candidatem:
        for sv_candi in sv_candidate[0]:
            if sv_candi[-1] == chr_name:
                signal_dict[chr_name].append(sv_candi)
        for sv_candi in sv_candidate[1]:
            if sv_candi[-1] == chr_name:
                signal_dict_error[chr_name].append(sv_candi)

    for chrname, sv_candidate in signal_dict.items():
        del_signal = []
        ins_signal = []
        inv_signal = []
        dup_signal = []
        tra_signal = []
        # print(chrname, len(sv_candidate))
        error_data = signal_dict_error[chrname]
        error_mean = np.mean([float(item[0]) for item in error_data])
        sv_candidate = sorted(sv_candidate, key=lambda x: x[0])
        for sv_candi in sv_candidate:
            if sv_candi[-3] == 'DEL':
                del_signal.append(
                    [sv_candi[0], sv_candi[1] + sv_candi[0], sv_candi[1], sv_candi[2], sv_candi[4], sv_candi[6]])
            if sv_candi[-3] == 'INS':
                ins_signal.append([sv_candi[0], sv_candi[0] + 1, sv_candi[1], sv_candi[2], sv_candi[4], sv_candi[6]])
            if sv_candi[-3] == 'INV':
                inv_signal.append(
                    [sv_candi[0], sv_candi[1] + sv_candi[0], sv_candi[1], sv_candi[2], sv_candi[4], sv_candi[6]])
            if sv_candi[-3] == 'DUP':
                dup_signal.append(
                    [sv_candi[0], sv_candi[1] + sv_candi[0], sv_candi[1], sv_candi[2], sv_candi[4], sv_candi[6]])
            if sv_candi[-3] == 'TRA':
                tra_signal.append(sv_candi)
        # print(len(ins_signal))
        # print(len(del_signal))
        ins_signal = np.array(sorted(ins_signal, key=lambda x: x[0]))
        del_signal = np.array(sorted(del_signal, key=lambda x: x[0]))
        inv_signal = np.array(sorted(inv_signal, key=lambda x: x[0]))
        dup_signal = np.array(sorted(dup_signal, key=lambda x: x[0]))
        tra_signal = sorted(tra_signal, key=lambda x: (x[8], x[0], x[3], x[4]))


        sum_del = mean_shift_def(chrname, del_signal, 'DEL', error_mean)

        sum_ins = mean_shift_def(chrname, ins_signal, 'INS', error_mean)
        sum_inv = mean_shift_def(chrname, inv_signal, 'INV', error_mean)
        sum_dup = mean_shift_def(chrname, dup_signal, 'DUP', error_mean)
        sum_trans = cluster_translocations(tra_signal)
        # sigbreak = signal_dict_break[chrname]
        # breakpoint_merge(sigbreak,sum_del)
        # breakpoint_merge(sigbreak,sum_ins)
        resultlist.extend(sum_del)
        resultlist.extend(sum_ins)
        resultlist.extend(sum_inv)
        resultlist.extend(sum_dup)
        resultlist.extend(sum_trans)
        error_mean_all.append(error_mean)
    return resultlist, error_mean_all


# %%
def breakpoint_merge(breakpointall, result):
    # print(breakpointall)
    for resultsum in result:
        for breakpoint in breakpointall:

            if (int(resultsum[1]) - 300) <= int(breakpoint[1]) and (int(resultsum[1]) + 300) >= int(breakpoint[1]):
                resultsum[3] = int(resultsum[3]) + 1

    # [chr_name, start, length, len(data), svtype, '0/1']


# %%
def load_all_data(x_train_name1, x_train_index, start, end, chr_name, bamfilepath):
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb')
    try:
        x_t = np.load(x_train_name1)
    except FileNotFoundError:
        return [], []
    else:
        xindex = np.load(x_train_index)
        predict1 = x_t.flatten()
        base = np.array(predict1)
        # print(x_train_index)
        # print(len(np.where(base>=0.5)[0]))
        if len(np.where(base >= 0.5)[0]) == 0:
            return [], []

        candidate, breakpointall = mergedeleton_long(base, bamfile, start, chr_name,
                                                     xindex)  # mergedeleton_long(base,bamfile,start,chr_name1,xindex)#mergedeleton_long(base,index,contig,bamfile)
        # mergedeleton_long(base,index,contig,bamfile) #mergedeleton_long(pre,bamfile,ssstart,chr_name)
        bamfile.close()

        return candidate, breakpointall


# %%
def batchdata(data, step, window=2000):
    data = data[:, :-1]
    if step != 0:
        data = data.reshape(-1, 20)[step:(step - window)]
    data = data.reshape(-1, 2000, 20, 1)

    return data


def predcit_step(base, predict):
    for i in range(len(predict)):
        if predict[i] >= 0.5:
            base[i] = 1
            base[i + 1] = 1
    return base


# %%
def define_tasks_support(resultlist,contig, bamfile_path, globle_coverage):
    tasks_support = []
    for chrname in contig:
        resultcontig = []
        for resultl in resultlist:
            for result in resultl[0]:
                if result[0] == chrname:
                    resultcontig.append(result)
        tasks_support.append((resultcontig, bamfile_path, globle_coverage, resultl[1]))
    return tasks_support


def extract_error(data):
    error_rate = {}
    for i in data:
        try:

            error_rate[i[0][0][0]] = i[1]
        except IndexError:

            print(f"IndexError: List index out of range for {i}")

    return error_rate

    


def support_read_calculate_multi(data, bamfile_path, globle_coverage, error_rate1):
    #print('eerror_rate',error_rate1)
    try:
        error_rate1 = error_rate1[0]
    except:
        error_rate1 = 0
    #error_rate1 = error_rate1[0]
    bamfile = pysam.AlignmentFile(bamfile_path, 'rb', threads=20)

    for message in data:
        # print(message)
        if message[4] == 'INS':
            contig = message[0]
            start_region = int(message[1]) - 1000
            if start_region < 0:
                start_region = 0
            end_region = int(message[1]) + 1000
            count_region = 0
            for read in bamfile.fetch(contig, start_region, end_region):
                if read.is_supplementary or read.is_secondary or read.is_unmapped:
                    continue
                count_region += 1
            # print('ins：',count_region,count_local)
            # min_support = calculate_min_support_ins(globle_coverage, count_region,0.5,0.7,0.3)CCS'
            error_data = float(error_rate1)
            min_support = calculate_min_support_ins(globle_coverage, count_region, 0.5, 0.7, 0.3, error_data)
            # min_supportT = calculate_min_supportT(globle_coverage, count_region,2,alpha=1.0, beta=1.0, gamma=1.0, low_cov_threshold=10, high_cov_threshold=30,global_threshold = 15)
            # print('ins：',globle_coverage,count_region,min_support,min_supportT)
            message.append(min_support)
            # message.append(count)
        elif message[4] == 'DEL':
            contig = message[0]
            # print(message[1])
            start_local = int(message[1])
            end_local = int(message[1]) + int(message[2])

            count_region = 0
            for read in bamfile.fetch(contig, start_local, end_local):
                if read.is_supplementary or read.is_secondary or read.is_unmapped:
                    continue
                count_region += 1
            error_data = float(error_rate1)
            min_support = calculate_min_support_del(globle_coverage, count_region, 0.5, 0.65, 0.35,
                                                    error_data)  ########################
            # min_supportT = calculate_min_supportT(globle_coverage, count_region,2,alpha=1.0, beta=1.0, gamma=1.0, low_cov_threshold=10, high_cov_threshold=30,global_threshold = 30)
            # print('del：',globle_coverage,count_region,min_support,min_supportT)
            # print('del：',count_region,count_local,min_support)
            message.append(min_support)

        elif message[4] == 'INV':
            contig = message[0]
            # print(message[1])
            start_local = int(message[1])
            end_local = int(message[1]) + int(message[2])

            count_region = 0
            for read in bamfile.fetch(contig, start_local, end_local):
                if read.is_supplementary or read.is_secondary or read.is_unmapped:
                    continue
                count_region += 1
            error_data = float(error_rate1)
            min_support = calculate_min_support_inv(globle_coverage, count_region, 0.6, 0.7, 0.3,
                                                    error_data)  ########################
            # min_supportT = calculate_min_supportT(globle_coverage, count_region,2,alpha=1.0, beta=1.0, gamma=1.0, low_cov_threshold=10, high_cov_threshold=30,global_threshold = 30)
            # print('del：',globle_coverage,count_region,min_support,min_supportT)
            # print('del：',count_region,count_local,min_support)
            message.append(min_support)
        elif message[4] == 'DUP':
            contig = message[0]
            # print(message[1])
            start_local = int(message[1])
            end_local = int(message[1]) + int(message[2])

            count_region = 0
            for read in bamfile.fetch(contig, start_local, end_local):
                if read.is_supplementary or read.is_secondary or read.is_unmapped:
                    continue
                count_region += 1
            error_data = float(error_rate1)
            min_support = calculate_min_support_dup(globle_coverage, count_region, 0.5, 0.7, 0.3,
                                                    error_data)  ########################
            # min_supportT = calculate_min_supportT(globle_coverage, count_region,2,alpha=1.0, beta=1.0, gamma=1.0, low_cov_threshold=10, high_cov_threshold=30,global_threshold = 30)
            # print('del：',globle_coverage,count_region,min_support,min_supportT)
            # print('del：',count_region,count_local,min_support)
            message.append(min_support)
        else:
            contig = message[0]
            start = max(0, (int(message[1]) - 1000))
            end = int(message[1]) + 1000
            count_region = 0
            for read in bamfile.fetch(contig, start, end):
                if read.is_supplementary or read.is_secondary or read.is_unmapped:
                    continue
                count_region += 1
            error_data = float(error_rate1)
            min_support = calculate_min_support_ins(globle_coverage, count_region, 0.5, 0.65, 0.35, error_data)
            message.append(min_support)

    bamfile.close()
    return data


def define_tasks_geno(resultlist,contig ,bamfile_path, error_rate_all):
    tasks_support = []

    for chrname in contig:
        resultcontig = []
        for resultl in resultlist:
            for result in resultl:
                if result[0] == chrname:
                    resultcontig.append(result)
                    error_rate = error_rate_all[result[0]]
        tasks_support.append((resultcontig, bamfile_path, error_rate))
    return tasks_support


def likelihood_0_0(ref_depth, alt_depth, error_rate):

    p_ref = 1 - error_rate
    p_alt = error_rate


    return (p_ref ** ref_depth) * (p_alt ** alt_depth)





def likelihood_0_1(ref_depth, alt_depth, error_rate, bias_factor=0.7):

    p_ref = bias_factor * (1 - error_rate)
    p_alt = (1 - bias_factor) * (1 - error_rate)  # 变异等位基因的概率


    return (p_ref ** ref_depth) * (p_alt ** alt_depth)
def likelihood_1_1(ref_depth, alt_depth, error_rate):

    p_ref = error_rate
    p_alt = 1 - error_rate


    return (p_ref ** ref_depth) * (p_alt ** alt_depth)


def calculate_genotype_likelihood(ref_depth, alt_depth, error_rate):

    likelihoods = {
        '0/0': likelihood_0_0(ref_depth, alt_depth, error_rate),
        '0/1': likelihood_0_1(ref_depth, alt_depth, error_rate),
        '1/1': likelihood_1_1(ref_depth, alt_depth, error_rate)
    }


    return max(likelihoods, key=likelihoods.get)

def filter_by_alignment_length(alignment, min_length=2000):

    return alignment.query_length >= min_length
def bayesian_genotype_likelihood(ref_depth, alt_depth, error_rate, prior_AA, prior_AB, prior_BB):


    likelihood_AA = likelihood_0_0(ref_depth, alt_depth, error_rate) * prior_AA
    likelihood_AB = likelihood_0_1(ref_depth, alt_depth, error_rate) * prior_AB
    likelihood_BB = likelihood_1_1(ref_depth, alt_depth, error_rate) * prior_BB
    epsilon = 1e-10

    total_likelihood = likelihood_AA + likelihood_AB + likelihood_BB
    total_likelihood = max(total_likelihood, epsilon)  # 防止除零错误
    posterior_AA = likelihood_AA / total_likelihood
    posterior_AB = likelihood_AB / total_likelihood
    posterior_BB = likelihood_BB / total_likelihood


    return max([(posterior_AA, '0/0'), (posterior_AB, '0/1'), (posterior_BB, '1/1')], key=lambda x: x[0])[1]

def calculate_minimum_overlap(start, end, variant_type):

    variant_length = end - start
    if variant_type in ['DEL', 'INV', 'DUP']:
        return min(variant_length / 2, 2000)
    elif variant_type in ['INS', 'BND']:
        return 100
    return 500

def filter_cigar_operations(alignment, max_clip_length=10000):

    cigar = alignment.cigartuples
    total_clip_length = 0
    for op, length in cigar:
        if op in [4, 5]:
            total_clip_length += length
    return total_clip_length < max_clip_length


def em_genotype(ref_depth, alt_depth, error_rate, max_iter=100):


    p_00 = 0.3
    p_01 = 0.3
    p_11 = 0.3

    for _ in range(max_iter):

        likelihood_00 = (1 - error_rate) ** ref_depth * error_rate ** alt_depth * p_00
        likelihood_01 = (0.5 * (1 - error_rate)) ** ref_depth * (0.5 * (1 - error_rate)) ** alt_depth * p_01
        likelihood_11 = error_rate ** ref_depth * (1 - error_rate) ** alt_depth * p_11
        epsilon = 1e-50
        total_likelihood = likelihood_00 + likelihood_01 + likelihood_11
        total_likelihood = max(total_likelihood, epsilon)  
        w_00 = likelihood_00 / total_likelihood
        w_01 = likelihood_01 / total_likelihood
        w_11 = likelihood_11 / total_likelihood


        p_00 = w_00
        p_01 = w_01
        p_11 = w_11
    #return p_00, p_01, p_11

    if p_00 > p_01 and p_00 > p_11:
        return '0/0'
    elif p_01 > p_00 and p_01 > p_11:
        return '0/1'
    else:
        return '1/1'


def genotype_by_depth_ratio(ref_depth, alt_depth, threshold=0.7):

    total_depth = ref_depth + alt_depth
    if total_depth == 0:
        return './.'

    ref_ratio = ref_depth / total_depth
    
    if ref_ratio >= threshold:
        return '0/0'
    elif ref_ratio <= (1 - threshold):
        return '1/1'
    else:
        return '0/1'




from collections import Counter

def combined_genotype_voting(ref_depth, alt_depth, error_rate, prior_AA, prior_AB, prior_BB):



    bayesian_geno = bayesian_genotype_likelihood(ref_depth, alt_depth, error_rate, prior_AA, prior_AB, prior_BB)


    em_geno = em_genotype(ref_depth, alt_depth, error_rate)


    depth_ratio_geno = genotype_by_depth_ratio(ref_depth, alt_depth, 0.8)

    votes = Counter()


    votes[bayesian_geno] += 1



    votes[em_geno] += 1


    votes[depth_ratio_geno] += 1


    return votes.most_common(1)[0][0]


def genotype_multi(candidates, bamfile, error_rate_list):
    bam = pysam.AlignmentFile(bamfile, 'rb')
    min_mapq = 20

    for candidate in candidates:
        # for candidate in candidate1[0]:
        #error_rate = float(error_rate_list[0])
        #error_rate = 0.01
        if candidate[6] == 'BND':
            min_support = int(candidate[-1])
            type = candidate[6]
            # print(candidate)
            reads_supporting_variant = candidate[-2]
        else:

            min_support = int(candidate[-1])
            type = candidate[4]
            reads_supporting_variant = candidate[-3]
        if len(reads_supporting_variant) < min_support:
            continue

        if type == "INS":
            max_bias =1000
            error_rate = float(0.01) if min_support <=3 else float(0.05)
            contig, start, end = candidate[0], int(candidate[1]), int(candidate[1]) + 1
        elif type == 'DEL' or type == 'INV' or type == 'DUP':
            if min_support <= 2:
                error_rate = float(0.00)
            elif min_support <= 4:
                error_rate = float(0.01)
            else:
                error_rate = float(0.05)
            #error_rate = float(0.00) if min_support <= 3 else float(0.05)
            
            max_bias = 2000
            contig, start, end = candidate[0], int(candidate[1]), int(candidate[1]) + int(candidate[2])
        else:
            up_bound = threshold_ref_count(len(reads_supporting_variant))
            
            error_rate = float(0.01) if min_support <= 2 else float(0.05)
            max_bias = 1000
            contig, start = candidate[0], int(candidate[1])
            end = start + 1
        contig_length = bam.get_reference_length(contig)
        alignment_it = bam.fetch(contig=contig, start=max(0, start - max_bias), stop=min(contig_length, end + max_bias))
        # Count reads that overlap the locus and therefore support the reference
        aln_no = 0
        reads_supporting_reference = set()

        while aln_no < 200:
            try:
                current_alignment = next(alignment_it)
            except StopIteration:
                break
            if current_alignment.query_name in reads_supporting_variant:
                continue
            if current_alignment.is_unmapped or current_alignment.is_secondary or current_alignment.mapping_quality < min_mapq or current_alignment.is_supplementary:
                continue
            aln_no += 1
            if type == "DEL" or type == "INV" or type == "DUP":
                minimum_overlap = min((end - start) / 2, 2000)


                if (
                        current_alignment.reference_start < start and current_alignment.reference_end > start + minimum_overlap) or \
                        (
                                current_alignment.reference_start < end - minimum_overlap and current_alignment.reference_end > end):
                    reads_supporting_reference.add(current_alignment.query_name)


            else:

                if current_alignment.reference_start < (start - max_bias) and current_alignment.reference_end > (
                        end + max_bias):
                    reads_supporting_reference.add(current_alignment.query_name)


                if type == 'BND':
                    if len(reads_supporting_reference) >= up_bound:
                        break

        # GT, GL, GQ, QUAL = cal_GL(len(reads_supporting_reference), len(reads_supporting_variant), type, 'CLR',min_support)
        # [first_chr_name, start1, first_direction, second_chr_name, start2, second_direction, 'BND', len(transs), '0/0', read_name_list]


        if (len(reads_supporting_reference)+len(reads_supporting_variant)) < min_support:
            geno = './.'
        else:
            geno = combined_genotype_voting(len(reads_supporting_reference), len(reads_supporting_variant), error_rate, 1/3, 1/3, 1/3)
            #geno = bayesian_genotype_likelihood(len(reads_supporting_reference), len(reads_supporting_variant), error_rate, 1/3, 1/3, 1/3)

        if type == 'BND':
            candidate[-3] = geno
        else:

            candidate[-4] = geno

    # print(GT,candidate[-3])
    # candidate.ref_reads = len(reads_supporting_reference)
    # candidate.alt_reads = len(reads_supporting_variant)
    bam.close()
    return candidates


# def genotype_multi(candidates, bamfile, error_rate_list):
#     bam = pysam.AlignmentFile(bamfile, 'rb')
#     min_mapq = 20
#
#     for candidate in candidates:
#         #error_rate = float(error_rate_list[0])
#         error_rate = float(0.05)
#         if candidate[6] == 'BND':
#             min_support = int(candidate[-1])
#             type = candidate[6]
#             reads_supporting_variant = candidate[-2]
#         else:
#             min_support = int(candidate[-1])
#             type = candidate[4]
#             reads_supporting_variant = candidate[-3]
#
#         # Fetch alignments around variant locus
#         if type == "INS":
#             max_bias = 500
#             contig, start, end = candidate[0], int(candidate[1]), int(candidate[1]) + 1
#         elif type == 'DEL' or type == 'INV' or type == 'DUP':
#             max_bias = 100
#             contig, start, end = candidate[0], int(candidate[1]), int(candidate[1]) + int(candidate[2])
#         else:
#             up_bound = threshold_ref_count(len(reads_supporting_variant))
#             max_bias = 100
#             contig, start = candidate[0], int(candidate[1])
#             end = start + 1
#
#         contig_length = bam.get_reference_length(contig)
#         alignment_it = bam.fetch(contig=contig, start=max(0, start - max_bias), stop=min(contig_length, end + max_bias))
#
#         aln_no = 0
#         reads_supporting_reference = set()
#
#         while aln_no < 500:
#             try:
#                 current_alignment = next(alignment_it)
#             except StopIteration:
#                 break
#             if current_alignment.query_name in reads_supporting_variant:
#                 continue
#             if current_alignment.is_unmapped or current_alignment.is_secondary or current_alignment.mapping_quality < min_mapq:
#                 continue
#             aln_no += 1
#             if type == "DEL" or type == "INV":
#                 #minimum_overlap = min((end - start) / 2, 100)
#                 #max_bias = min(100, (end - start) / 2)
#                 minimum_overlap = min((end - start) / 2, 100)
#                 if (current_alignment.reference_start < (end - minimum_overlap) and current_alignment.reference_end > (end + max_bias) or
#                         current_alignment.reference_start < (start - max_bias) and current_alignment.reference_end > (start + minimum_overlap)):
#                     reads_supporting_reference.add(current_alignment.query_name)
#             else:
#                 if current_alignment.reference_start < (start - max_bias) and current_alignment.reference_end > (end + max_bias):
#                     reads_supporting_reference.add(current_alignment.query_name)
#                     if type == 'BND':
#                         if len(reads_supporting_reference) >= up_bound:
#                             break
#
#         # Use the improved cal_GL function to calculate genotype likelihoods
#         geno, PL, GQ, QUAL = cal_GL(len(reads_supporting_reference), len(reads_supporting_variant),error_rate, prior)
#
#         # Assign the genotype to the candidate object
#         if type == 'BND':
#             candidate[-3] = geno
#         else:
#             candidate[-4] = geno
#
#     bam.close()
#     return candidates


# %%

def model_predict(weights_num, bamfilepath, data_path1, testpath, contigg):
    weights_path = weights_num
    model.load_weights(weights_path)
    start = 0
    infor_05 = []
    contig2length = {}
    datapath1 = data_path1
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads=20)

    print('contigg:',contigg)
    print('len:',len(contigg))
    if len(contigg) == 0:
        contig = []
        for count in range(len(bamfile.get_index_statistics())):
            contig.append(bamfile.get_index_statistics()[count].contig)
            contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    else:
        contig = np.array(contigg).astype(str)
    for count in range(len(bamfile.get_index_statistics())):
        contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    data = []
    resultlist = [['CONTIG', 'START', 'SVLEN', 'READ_SUPPORT', 'SVTYPE']]
    print('contig:',contig)
    for ww in contig:
        chr_name = ww
        chr_length = contig2length[ww]
        if 'chr' in chr_name:
            chr_name1 = chr_name[3:]
        else:
            chr_name1 = chr_name
        ider = math.ceil(chr_length / 10000000)
        start = 0
        print('+++++++', chr_name, '++++++++++')
        print('chr:', chr_name, ider)
        for i in range(ider):
            print('SVHunter_predict_chr:', chr_name, i, '/', ider)
            x_train_name1 = datapath1 + '/chr' + chr_name1 + '_' + str(start) + '_' + str(start + 10000000) + '.npy'
            x_train_index = datapath1 + '/chr' + chr_name1 + '_' + str(start) + '_' + str(
                start + 10000000) + '_index.npy'
            try:
                x_t = np.load(x_train_name1)
            except ValueError:
                start = start + 10000000
                continue
            else:

                data1 = x_t.reshape(-1, 2000, 20, 1)
                #data1_y = x_t[:, -1].reshape(-1, 1)
                #print(data1.shape, data1_y.shape)
                print(data1.shape)
                datatmp1 = tf.data.Dataset.from_tensor_slices(data1).batch(100)

                predict1 = model.predict(datatmp1, verbose=0)



                predict1 = predict1.flatten()


                np.save(testpath + '/chr' + chr_name1 + '_' + str(start) + '_' + str(start + 10000000) + '_predict.npy',
                        predict1)
                start += 10000000


# %%
def average_read_coverage(bamfilepath, chr_name, lengthh):
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads=20)
    all_length = lengthh
    chr_align_length = 0
    for read in bamfile.fetch(chr_name):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        chr_align_length += read.query_length
    return chr_align_length


# %%


def threshold_ref_count(num):
    if num <= 2:
        return 20 * num
    elif 3 <= num <= 5:
        return 9 * num
    elif 6 <= num <= 15:
        return 7 * num
    else:
        return 5 * num


def log10sumexp(log10_probs):
    # Normalization of Genotype likelihoods
    m = max(log10_probs)
    return m + log10(sum(pow(10.0, x - m) for x in log10_probs))


def normalize_log10_probs(log10_probs):
    # Adjust the Genotype likelihoods
    log10_probs = np.array(log10_probs)
    lse = log10sumexp(log10_probs)
    return np.minimum(log10_probs - lse, 0.0)


def rescale_read_counts(c0, c1, max_allowed_reads=100):
    """Ensures that n_total <= max_allowed_reads, rescaling if necessary."""
    Total = c0 + c1
    if Total > max_allowed_reads:
        c0 = int(max_allowed_reads * float(c0 / Total))
        c1 = max_allowed_reads - c0
    return c0, c1



#Define the error rate and priors globally
prior = float(1 / 3)
Genotype = ["0/0", "0/1", "1/1"]

# Function to normalize log10 probabilities
def log10sumexp(log10_probs):
    """Normalization of Genotype likelihoods"""
    m = max(log10_probs)
    return m + log10(sum(pow(10.0, x - m) for x in log10_probs))

def normalize_log10_probs(log10_probs):
    """Adjust the Genotype likelihoods"""
    log10_probs = np.array(log10_probs)
    lse = log10sumexp(log10_probs)
    return np.minimum(log10_probs - lse, 0.0)

# Rescaling function for read counts to prevent overflows
def rescale_read_counts(c0, c1, max_allowed_reads=100):
    """Ensure that n_total <= max_allowed_reads, rescaling if necessary."""
    total = c0 + c1
    if total > max_allowed_reads:
        c0 = int(max_allowed_reads * float(c0 / total))
        c1 = max_allowed_reads - c0
    return c0, c1

# # The core Genotype Likelihood calculation function
def cal_GL(c0, c1,err, prior):
    # First, rescale read counts to prevent overflows
    c0, c1 = rescale_read_counts(c0, c1)  # DR, DV

    # Calculate the original genotype likelihoods
    ori_GL00 = np.float64(pow((1 - err), c0) * pow(err, c1) * (1 - prior) / 2)
    ori_GL11 = np.float64(pow(err, c0) * pow((1 - err), c1) * (1 - prior) / 2)
    ori_GL01 = np.float64(pow(0.5, c0 + c1) * prior)

    # Normalize the genotype likelihoods
    prob = list(normalize_log10_probs([log10(ori_GL00), log10(ori_GL01), log10(ori_GL11)]))
    GL_P = [pow(10, i) for i in prob]  # Convert log10 back to linear space

    # Calculate PL (Phred-scaled likelihoods)
    PL = [int(np.around(-10 * log10(i))) for i in GL_P]

    # Calculate GQ (Genotype Quality)
    GQ = [int(-10 * log10(GL_P[1] + GL_P[2])), int(-10 * log10(GL_P[0] + GL_P[2])), int(-10 * log10(GL_P[0] + GL_P[1]))]

    # Calculate QUAL score (Quality of the 0/0 genotype)
    QUAL = abs(np.around(-10 * log10(GL_P[0]), 1))

    # Return the most likely genotype and associated metrics
    return Genotype[prob.index(max(prob))], "%d,%d,%d" % (PL[0], PL[1], PL[2]), max(GQ), QUAL





# %%
def cluster_by_predict(bamfilepath, data_path, testpath, outputpath, contigg,threads_numm=15):
    threads_num = int(threads_numm)
    thread_pool = Pool(threads_num)
    start = 0
    infor_05 = []
    contig2length = {}
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads=20)
    filename = os.path.basename(bamfilepath)
    name_without_extension = os.path.splitext(filename)[0]
    directory_path = os.path.join(outputpath, name_without_extension)
    # 创建目录
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    output_del_path = outputpath + '/' + name_without_extension + '/' + name_without_extension + '_del.vcf'
    output_ins_path = outputpath + '/' + name_without_extension + '/' + name_without_extension + '_ins.vcf'
    output_inv_path = outputpath + '/' + name_without_extension + '/' + name_without_extension + '_inv.vcf'
    output_dup_path = outputpath + '/' + name_without_extension + '/' + name_without_extension + '_dup.vcf'
    output_bnd_path = outputpath + '/' + name_without_extension + '/' + name_without_extension + '_tra.vcf'
    output_all_path = outputpath + '/' + name_without_extension + '/' + name_without_extension + '_all.vcf'
    # contigg = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']
    # contigg = ['12','13','14','15','16','17','18','19','20','21','22','X','Y']

    datapath1 = testpath
    datapath = data_path

    if len(contigg) == 0:
        contig = []
        for count in range(len(bamfile.get_index_statistics())):
            contig.append(bamfile.get_index_statistics()[count].contig)
            contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    else:
        contig = np.array(contigg).astype(str)
    for count in range(len(bamfile.get_index_statistics())):
        contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    coverage_task = [(bamfilepath, chr__name, contig2length[chr__name]) for chr__name in contig]
    global_coverage_all = thread_pool.starmap(average_read_coverage, coverage_task)
    alllength = [(contig2length[chr__name]) for chr__name in contig]
    global_coverage = math.ceil(sum(global_coverage_all) / sum(alllength))
    # tasks = [(x_train_name1,x_train_index,start,end,chr_name,bamfilepath) for x_train_name1,x_train_index,start,end,chr_name in x_train_data_all]

    data = []
    candidate = []
    x_train_data_all = []
    for ww in contig:
        chr_name = ww
        chr_length = contig2length[ww]
        if 'chr' in chr_name:
            chr_name1 = chr_name[3:]
        else:
            chr_name1 = chr_name
        ider = math.ceil(chr_length / 10000000)
        start = 0
        print('+++++++', chr_name, '++++++++++')
        print('chr:', chr_name, ider)
        for i in range(ider):
            # print('insertion_predict_chr:',chr_name,i,'/',ider)
            x_train_name1 = datapath1 + '/chr' + chr_name1 + '_' + str(start) + '_' + str(
                start + 10000000) + '_predict' + '.npy'

            x_train_index = datapath + '/chr' + chr_name1 + '_' + str(start) + '_' + str(
                start + 10000000) + '_index.npy'

            try:
                x_t = np.load(x_train_name1)
                x_i = np.load(x_train_index)
            except FileNotFoundError:
                start = start + 10000000
                continue

            x_train_data_all.append([x_train_name1, x_train_index, start, start + 10000000, chr_name])
            start += 10000000

    tasks = [(x_train_name1, x_train_index, start, end, chr_name, bamfilepath) for
             x_train_name1, x_train_index, start, end, chr_name in x_train_data_all]
    parsing_results = None
    breakpoint_all = None
    parsing_results = thread_pool.starmap(load_all_data, tasks)
    # contigg = ['12','13','14','15','16','17','18','19','20','21','22','X','Y']
    tasks_candidate = [(parsing_results, chrname) for chrname in contig]
    resultlist_begin = None
    resultlist_begin = thread_pool.starmap(analysis_candidate, tasks_candidate)

    tasks_support = define_tasks_support(resultlist_begin, contig,bamfilepath, global_coverage)
    error_all = extract_error(resultlist_begin)
    #print('error:', error_all)
    print('-----------------genotype_calculate-----------------')
    resultlist1 = thread_pool.starmap(support_read_calculate_multi, tasks_support)
    tasks_geno = define_tasks_geno(resultlist1, contig,bamfilepath, error_all)
    resultlist = thread_pool.starmap(genotype_multi, tasks_geno)

    all_result = []
    BND_result = []
    for read in resultlist:
        for read1 in read:
            if str(read1[6]) != 'BND':
                if int(read1[3]) >= int(read1[8]) and int(float(read1[2])) >= 40:
                    # if int(read1[3]) >= int(support_read)  and int(float(read1[2])) >= 50 and str(read1[4]) == 'INS' :
                    # all_result.append([str(read1[0]),int(float(read1[1])),int(float(read1[2])),int(float(read1[3])),str(read1[4]),'.'])
                    all_result.append(
                        [str(read1[0]), int(float(read1[1])), int(float(read1[2])), int(float(read1[3])), str(read1[4]),
                         read1[-4], read1[-2], read1[-1]])
            elif str(read1[6]) == 'BND':
                if int(read1[7]) >= int(read1[10]) and str(read1[0]) in contig and str(read1[3]) in contig:
                    BND_result.append([read1])

            # first_chr, start1, first_direction, second_chr, start2, second_direction, svtype, svlen, geno, reads, support_read

    ins = []
    inss = []
    for read in resultlist:
        for read1 in read:
            if str(read1[4]) == 'INS':
                if int(read1[3]) >= int(read1[8]) and int(float(read1[2])) >= 40:
                    # if int(read1[3]) >= int(support_read)  and int(float(read1[2])) >= 50 and str(read1[4]) == 'INS' :
                    ins.append(
                        [str(read1[0]), int(float(read1[1])), int(float(read1[2])), int(float(read1[3])), 'INS', '.'])
                    inss.append([str(read1[0]), int(float(read1[1])), int(float(read1[2])), int(float(read1[3])), 'INS',
                                 read1[-4], read1[-2], read1[-1]])
    # outvcfpath_ins = '/home/rtgao/code/vcf_file/vcftest/hg002_ins.vcf'

    # generate_vcf(ins, contig2length, outvcfpath_ins)
    dell = []
    delll = []
    for read in resultlist:
        for read1 in read:
            if str(read1[4]) == 'DEL':
                if int(float(read1[3])) >= int(read1[8]) and int(float(read1[2])) >= 40 and str(read1[4]) == 'DEL':
                    # if int(float(read1[3])) >= int(support_read) and int(float(read1[2])) >= 50 and str(read1[4]) == 'DEL'  :
                    dell.append(
                        [str(read1[0]), int(float(read1[1])), int(float(read1[2])), int(float(read1[3])), 'DEL', '.'])
                    delll.append(
                        [str(read1[0]), int(float(read1[1])), int(float(read1[2])), int(float(read1[3])), 'DEL',
                         read1[-4], read1[-2], read1[-1]])

    dup = []
    for read in resultlist:
        for read1 in read:
            if str(read1[4]) == 'DUP':
                if int(float(read1[3])) >= int(read1[8]) and int(float(read1[2])) >= 40 and str(read1[4]) == 'DUP':
                    # if int(float(read1[3])) >= int(support_read) and int(float(read1[2])) >= 50 and str(read1[4]) == 'DEL'  :
                    # dell.append([str(read1[0]),int(float(read1[1])),int(float(read1[2])),int(float(read1[3])),'DEL','.'])
                    dup.append([str(read1[0]), int(float(read1[1])), int(float(read1[2])), int(float(read1[3])), 'DUP',
                                read1[-4], read1[-2], read1[-1]])

    inv = []
    for read in resultlist:
        for read1 in read:
            if str(read1[4]) == 'INV':
                if int(float(read1[3])) >= int(10) and int(float(read1[2])) >= 40 and str(read1[4]) == 'INV':
                    # if int(float(read1[3])) >= int(support_read) and int(float(read1[2])) >= 50 and str(read1[4]) == 'DEL'  :

                    inv.append([str(read1[0]), int(float(read1[1])), int(float(read1[2])), int(float(read1[3])), 'INV',
                                read1[-4], read1[-2], read1[-1]])

    # outvcfpath_del = '/home/rtgao/code/vcf_file/vcftest/hg002_del.vcf'
    print('----------generate_genotype--------------')
    print('len_del:', len(delll))
    print('len_ins:', len(inss))
    print('len_inv:', len(inv))
    print('len_dup:', len(dup))
    print('len_bnd:', len(BND_result))
    # print(inss)
    # genotype(delll,'DEL',bamfile,support_read)
    # genotype(inss,'INS',bamfile,support_read)
    print('----------generate_vcf--------------')

    generate_vcf(all_result, contig2length, output_all_path, BND_result)

    # print(inss)
    generate_vcf(inss, contig2length, output_ins_path)
    generate_vcf(delll, contig2length, output_del_path)
    generate_vcf(dup, contig2length, output_dup_path)
    generate_vcf(inv, contig2length, output_inv_path)
    generate_vcf_bnd(inss, contig2length, output_bnd_path, BND_result)
    bamfile.close()




# %%

def generate_vcf(inslist, contiglength, outvcfpath, BND_result=None):
    head = """##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">\n"""
    body = ''
    for contig in contiglength:
        body += "##contig=<ID=" + contig + ",length=" + str(int(contiglength[contig])) + ">\n"
    tail = """##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the structural variant">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV:DEL=Deletion, INS=Insertion, INV=Inversion, DUP=Duplication, BND=Translocation">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t.\n"""
    vcfinfo = head + body + tail

    for rec in inslist:
        contig = rec[0]
        geno = rec[5]
        if str(rec[4]) == 'INS':
            recinfo = 'SVLEN=' + str(int(rec[2])) + ';SVTYPE=' + str(rec[4]) + ';END=' + str(
                rec[1]) + ';' + '\tGT\t' + str(geno) + '\n'
        elif str(rec[4]) == 'DEL':
            recinfo = 'SVLEN=' + str(int(rec[2])) + ';SVTYPE=' + str(rec[4]) + ';END=' + str(
                rec[1] + rec[2]) + ';' + '\tGT\t' + str(geno) + '\n'
        elif str(rec[4]) == 'INV':
            recinfo = 'SVLEN=' + str(int(rec[2])) + ';SVTYPE=' + str(rec[4]) + ';END=' + str(
                rec[1] + rec[2]) + ';' + '\tGT\t' + str(geno) + '\n'
        elif str(rec[4]) == 'DUP':
            recinfo = 'SVLEN=' + str(int(rec[2])) + ';SVTYPE=' + str(rec[4]) + ';END=' + str(
                rec[1] + rec[2]) + ';' + '\tGT\t' + str(geno) + '\n'
        vcfinfo += (contig + '\t' + str(int(rec[1])) + '\t' + '.' + '\t' + 'N' + '\t' + '<' + str(
            rec[4]) + '>' + '\t' + '.' + '\t' + 'PASS' + '\t' + recinfo)

    if BND_result is not None:
        for i, transs in enumerate(BND_result):
            for trans in transs:
                first_chr, start1, first_direction, second_chr, start2, second_direction, svtype, support_read, genotype, _, _ = trans
                geno = genotype

                if first_direction == 'fwd' and second_direction == 'fwd':
                    alt1 = f'N[{second_chr}:{start2}['
                    alt2 = f']{first_chr}:{start1}]N'
                elif first_direction == 'fwd' and second_direction == 'rev':
                    alt1 = f'N]{second_chr}:{start2}]'
                    alt2 = f'[{first_chr}:{start1}[N'
                elif first_direction == 'rev' and second_direction == 'fwd':
                    alt1 = f']{second_chr}:{start2}]N'
                    alt2 = f'N[{first_chr}:{start1}['
                else:  # first_direction == 'rev' and second_direction == 'rev'
                    alt1 = f'[{second_chr}:{start2}[N'
                    alt2 = f'N]{first_chr}:{start1}]'

                info1 = f'SVTYPE=BND;'
                vcfinfo += f'{first_chr}\t{start1}\t.\tN\t{alt1}\t.\tPASS\t{info1}\tGT\t{geno}\n'

                info2 = f'SVTYPE=BND;'
                vcfinfo += f'{second_chr}\t{start2}\t.\tN\t{alt2}\t.\tPASS\t{info2}\tGT\t{geno}\n'

    with open(outvcfpath, "w") as f:
        f.write(vcfinfo)


def generate_vcf_bnd(inslist, contiglength, outvcfpath, BND_result=None):
    head = """##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">\n"""
    body = ''
    for contig in contiglength:
        body += "##contig=<ID=" + contig + ",length=" + str(int(contiglength[contig])) + ">\n"
    tail = """##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the structural variant">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV:DEL=Deletion, INS=Insertion, INV=Inversion, DUP=Duplication, BND=Translocation">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t.\n"""
    vcfinfo = head + body + tail

    if BND_result is not None:
        for i, transs in enumerate(BND_result):
            for trans in transs:
                first_chr, start1, first_direction, second_chr, start2, second_direction, svtype, support_read, genotype, _, _ = trans
                geno = './.'

                if first_direction == 'fwd' and second_direction == 'fwd':
                    alt1 = f'N[{second_chr}:{start2}['
                    alt2 = f']{first_chr}:{start1}]N'
                elif first_direction == 'fwd' and second_direction == 'rev':
                    alt1 = f'N]{second_chr}:{start2}]'
                    alt2 = f'[{first_chr}:{start1}[N'
                elif first_direction == 'rev' and second_direction == 'fwd':
                    alt1 = f']{second_chr}:{start2}]N'
                    alt2 = f'N[{first_chr}:{start1}['
                else:  # first_direction == 'rev' and second_direction == 'rev'
                    alt1 = f'[{second_chr}:{start2}[N'
                    alt2 = f'N]{first_chr}:{start1}]'

                info1 = f'SVTYPE=BND;'
                vcfinfo += f'{first_chr}\t{start1}\t.\tN\t{alt1}\t.\tPASS\t{info1}\tGT\t{geno}\n'

                info2 = f'SVTYPE=BND;'
                vcfinfo += f'{second_chr}\t{start2}\t.\tN\t{alt2}\t.\tPASS\t{info2}\tGT\t{geno}\n'

    with open(outvcfpath, "w") as f:
        f.write(vcfinfo)


# %%
def main_all(all_data_path):
    for i in range(23, 32):
        for bamfilepath, data_path, support_read in all_data_path:
            print(bamfilepath, data_path, support_read, i)
            # circulation_weights_my(i,bamfilepath,data_path)
            print('-------------finished_predict---------')
            cluster_by_predict(bamfilepath, data_path, support_read, i)


# %%
def alignment_quality_score(mapq, error_rate, num_of_mismatch, mm_rate):
    mapq_weight = 2.0
    error_rate_weight = 5.0
    mismatch_weight = 0.1
    mm_rate_weight = 8.0
    mapq_scale = 0.1
    mapq_threshold = 20
    mismatch_scale = 0.01
    mismatch_threshold = 50
    mm_rate_exponent = 2.0
    Alignment_Quality_Score = (
                (mapq_weight * math.log(1 + mapq)) / (1 + math.exp(-mapq_scale * (mapq - mapq_threshold))) - (
                    error_rate_weight * math.sqrt(error_rate)) -
                (mismatch_weight * num_of_mismatch / (
                            1 + math.exp(-mismatch_scale * (num_of_mismatch - mismatch_threshold)))) -
                (mm_rate_weight * math.pow(mm_rate, mm_rate_exponent)))
    return abs(Alignment_Quality_Score)


# %%


def calculate_min_support_ins(c_global, c_local, mu, sigma, rho, error_rate):

    local_deviation_impact = math.tanh((c_local - c_global) / c_global)

    if error_rate <= 0.1:
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
        return round(min_support)

    else:
        if c_global <= 10:
            mu = 0.6
            min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)

        else:
            min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)

        return round(min_support)

    return round(min_support)


def calculate_min_support_del(c_global, c_local, mu, sigma, rho, error_rate):

    local_deviation_impact = math.tanh((c_local - c_global) / c_global)
    if c_global >= 80:
        return 10
    if error_rate <= 0.1:
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
        return round(min_support)
    else:
        if c_global >= 20:
            mu = 0.6


        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)

    return round(min_support)


def calculate_min_support_inv(c_global, c_local, mu, sigma, rho, error_rate):

    local_deviation_impact = math.tanh((c_local - c_global) / c_global)
    if c_global >= 80:
        return 10
    if error_rate <= 0.1:
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
        return round(min_support)
    else:
        if c_global >= 20:
            mu = 0.7


        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)

    return round(min_support)


def calculate_min_support_dup(c_global, c_local, mu, sigma, rho, error_rate):

    local_deviation_impact = math.tanh((c_local - c_global) / c_global)
    if c_global >= 80:
        return 10
    if error_rate <= 0.1:
        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)
        return round(min_support)
    else:
        if c_global >= 20:
            mu = 0.5


        min_support = mu * (c_global ** sigma) * (1 + rho * local_deviation_impact)

    return round(min_support)


def support_read_calculate(data, bamfile_path, globle_coverage):
    bamfile = pysam.AlignmentFile(bamfile_path, 'rb', threads=20)

    for message1 in data:
        for message in message1[0]:
            # print(message)
            if message[4] == 'INS':
                contig = message[0]
                start_region = int(message[1]) - 1000
                if start_region < 0:
                    start_region = 0
                end_region = int(message[1]) + 1000
                count_region = 0
                for read in bamfile.fetch(contig, start_region, end_region):
                    if read.is_supplementary or read.is_secondary or read.is_unmapped:
                        continue
                    count_region += 1
                # print('ins：',count_region,count_local)
                # min_support = calculate_min_support_ins(globle_coverage, count_region,0.5,0.7,0.3)CCS'
                error_data = float(message1[1][0])
                min_support = calculate_min_support_ins(globle_coverage, count_region, 0.5, 0.7, 0.3, error_data)
                # min_supportT = calculate_min_supportT(globle_coverage, count_region,2,alpha=1.0, beta=1.0, gamma=1.0, low_cov_threshold=10, high_cov_threshold=30,global_threshold = 15)
                # print('ins：',globle_coverage,count_region,min_support,min_supportT)
                message.append(min_support)
                # message.append(count)
            elif message[4] == 'DEL':
                contig = message[0]
                # print(message[1])
                start_local = int(message[1])
                end_local = int(message[1]) + int(message[2])

                count_region = 0
                for read in bamfile.fetch(contig, start_local, end_local):
                    if read.is_supplementary or read.is_secondary or read.is_unmapped:
                        continue
                    count_region += 1
                error_data = float(message1[1][0])
                min_support = calculate_min_support_del(globle_coverage, count_region, 0.5, 0.65, 0.35,
                                                        error_data)  ########################
                # min_supportT = calculate_min_supportT(globle_coverage, count_region,2,alpha=1.0, beta=1.0, gamma=1.0, low_cov_threshold=10, high_cov_threshold=30,global_threshold = 30)
                # print('del：',globle_coverage,count_region,min_support,min_supportT)
                # print('del：',count_region,count_local,min_support)
                message.append(min_support)

            elif message[4] == 'INV':
                contig = message[0]
                # print(message[1])
                start_local = int(message[1])
                end_local = int(message[1]) + int(message[2])

                count_region = 0
                for read in bamfile.fetch(contig, start_local, end_local):
                    if read.is_supplementary or read.is_secondary or read.is_unmapped:
                        continue
                    count_region += 1
                error_data = float(message1[1][0])
                min_support = calculate_min_support_inv(globle_coverage, count_region, 0.6, 0.7, 0.3,
                                                        error_data)  ########################
                # min_supportT = calculate_min_supportT(globle_coverage, count_region,2,alpha=1.0, beta=1.0, gamma=1.0, low_cov_threshold=10, high_cov_threshold=30,global_threshold = 30)
                # print('del：',globle_coverage,count_region,min_support,min_supportT)
                # print('del：',count_region,count_local,min_support)
                message.append(min_support)
            elif message[4] == 'DUP':
                contig = message[0]
                # print(message[1])
                start_local = int(message[1])
                end_local = int(message[1]) + int(message[2])

                count_region = 0
                for read in bamfile.fetch(contig, start_local, end_local):
                    if read.is_supplementary or read.is_secondary or read.is_unmapped:
                        continue
                    count_region += 1
                error_data = float(message1[1][0])
                min_support = calculate_min_support_dup(globle_coverage, count_region, 0.5, 0.7, 0.3,
                                                        error_data)  ########################
                # min_supportT = calculate_min_supportT(globle_coverage, count_region,2,alpha=1.0, beta=1.0, gamma=1.0, low_cov_threshold=10, high_cov_threshold=30,global_threshold = 30)
                # print('del：',globle_coverage,count_region,min_support,min_supportT)
                # print('del：',count_region,count_local,min_support)
                message.append(min_support)
            else:
                contig = message[0]
                start = max(0, (int(message[1]) - 1000))
                end = int(message[1]) + 1000
                count_region = 0
                for read in bamfile.fetch(contig, start, end):
                    if read.is_supplementary or read.is_secondary or read.is_unmapped:
                        continue
                    count_region += 1
                error_data = float(message1[1][0])
                min_support = calculate_min_support_ins(globle_coverage, count_region, 0.5, 0.65, 0.35, error_data)
                message.append(min_support)

    bamfile.close()
    return data


    #cluster_by_predict(bamfilepath,data_path,testpath,outputpath,contigg = contigg)

