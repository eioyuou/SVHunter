import pysam
#import torch
import pandas as pd
import time
import numpy as np
import math
from multiprocessing import Process
import multiprocessing
from statistics import mean
import gc

#%%
def get_read_info(bamfile):
    read_info = []
    for read in bamfile.fetch():
        read_info.append([read.reference_name,read.reference_start,read.reference_end,read.query_name,read.query_sequence])
    return read_info
#%%
def average_read_coverage(bamfile,chr_name,lengthh):
    all_length = lengthh
    chr_align_length = 0
    for read in bamfile.fetch(chr_name):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        chr_align_length += read.query_length
    depth_coverage = np.ceil(int(chr_align_length) / int(all_length))
    return depth_coverage
#%%
def startend_balance(chr_name,start,end,vcffile_path):
    index = np.arange(start,end,200)
    label = labeldata(vcffile_path,chr_name,start,end,200,index)
    vstart = np.where(label==1)[0] * 200
    nstart = np.random.choice(np.where(label==0)[0]*200,size = len(vstart),replace = False)
    return vstart,nstart
#%%
dic_starnd = {1: '+', 2: '-'}
signal = {1 << 2: 0, \
          1 >> 1: 1, \
          1 << 4: 2, \
          1 << 11: 3, \
          1 << 4 | 1 << 11: 4}
def detect_flag(Flag):
    back_sig = signal[Flag] if Flag in signal else 0
    return back_sig
#%%
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

            # 如果是软剪切或硬剪切，并且是CIGAR字符串的开始部分，更新 readloc
            if c in ['S', 'H'] and readstart is None:
                # 如果是软剪切，更新 readstartac
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

                # 对于结尾的软剪切，记录 readend 和 refend
                if c in ['H', 'S'] and readstart is not None:
                    readend = readloc
                    refend = refloc

            number = ''

    if readend is None:
        readend = readloc

    if refend is None:
        refend = refloc

    return refstart, refend, readstart, readend
#%%
def splitreadlist(readsp):
    read = readsp
    sv_list = []
    aligned_length = read.reference_length
    process_signal = detect_flag(read.flag)
    #print(process_signal)
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

            if strand == '-' and sup_mapq >= 0:
                readstart = read.query_length - readend_2
                readend = read.query_length - readstart_2
                sv_list.append([readstart, readend, refstart_2, refend_2, tmpcontig, strand])
            elif strand == '+' and sup_mapq >= 0:
                sv_list.append([readstart_2, readend_2, refstart_2, refend_2, tmpcontig, strand])

    if len(sv_list) == 0:
        return []

    return sv_list
#%%
def feature_record(alignment_current, alignment_next):
    distance_on_read = alignment_next[0] - alignment_current[1]
    if  alignment_current[-1] == '+':
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
                deviation, False)
#%%
def feature_read_segement(svlist):
    sg_list= []
    sorted_alignment_list = sorted(svlist, key=lambda aln: (aln[0], aln[1]))
    for index in range(len(sorted_alignment_list) - 1):
        sg_list.append(feature_record(sorted_alignment_list[index], sorted_alignment_list[index + 1]))
    if len(svlist) >= 3 and sorted_alignment_list[0][-2] != sorted_alignment_list[1][-2]:
        sg_list.append(feature_record(sorted_alignment_list[0], sorted_alignment_list[-1], ins_tra_flag=True))
    return sg_list
#%%
def analyze_read_segments(read, segement_data, candidate):
    sv_signatures = []
    min_sv_size = 40
    segment_overlap_tolerance = 5
    read_name = read.query_name
    for sv_sig in segement_data:
        alignment_current = sv_sig[0]
        alignment_next = sv_sig[1]
        #print(alignment_current, alignment_next)
        ref_chr = alignment_current[-2]
        chr_, orientation, distance_on_read, distance_on_reference, deviation, long_ins = sv_sig[2:]
        if chr_ == 1:
            if orientation == 1:
                if distance_on_reference >= -min_sv_size or long_ins:  # INS
                    if deviation > 0:  # INS
                        if  alignment_current[-1] == '+':
                            start = (alignment_current[3] + alignment_next[2]) // 2 if not long_ins else min(alignment_current[3], alignment_next[2])
                        else:
                            start = (alignment_current[2] + alignment_next[3]) // 2 if not long_ins else min(alignment_current[2], alignment_next[3])
                        end = start + deviation
                        if end - start < min_sv_size:
                            continue
                        insertion_seq = 'A'

                        candidate.append(
                                        [start, deviation, read_name, insertion_seq, 'INS', ref_chr])

                    elif deviation < 0:  # DEL
                        if alignment_current[-1] == '+':
                            start = alignment_current[3]
                        else:
                            start = alignment_next[3]
                        end = start - deviation
                        if end - start < min_sv_size:
                            continue

                        candidate.append([start, -deviation, read_name, 'None', 'DEL', ref_chr])

                    else:
                        continue
                else:

                    if alignment_current[-1] == '+':
                        start = alignment_next[2]
                        end = alignment_current[3]
                    else:
                        start = alignment_current[2]
                        end = alignment_next[3]

                    svlen = end - start
                    candidate.append([start, svlen, read_name, 'None', 'DUP', ref_chr])

            else:  # INV
                if  alignment_current[-1] == '+':  # +-
                    if alignment_next[2] - alignment_current[
                        3] >= -segment_overlap_tolerance:
                        start = alignment_current[3]
                        end = alignment_next[3]
                        svlen = end - start
                        candidate.append([start, svlen, read_name, 'None', 'INV', ref_chr])

                    elif alignment_current[2] - alignment_next[
                        3] >= -segment_overlap_tolerance:
                        start = alignment_next[3]
                        end = alignment_current[3]
                        svlen = end - start

                        candidate.append([start, svlen, read_name, 'None', 'INV', ref_chr])
                    else:
                        continue
                else:  # -+
                    if alignment_next[2] - alignment_current[
                        3] >= -segment_overlap_tolerance:
                        start = alignment_current[2]
                        end = alignment_next[2]
                        svlen = end - start
                        candidate.append([start, svlen, read_name, 'None', 'INV', ref_chr])

                    elif alignment_current[2] - alignment_next[
                        3] >= -segment_overlap_tolerance:
                        start = alignment_next[2]
                        end = alignment_current[2]
                        svlen = end - start
                        candidate.append([start, svlen, read_name, 'None', 'INV', ref_chr])

                    else:
                        continue

        else:  # TRA
            ref_chr_next = alignment_next[-2]

            if orientation == 1:

                if alignment_current[-1] == '+':  # ++
                    if ref_chr < ref_chr_next:
                        start = alignment_current[3]
                        end = alignment_next[2]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[2]
                        end = alignment_current[3]

                    candidate.append([ref_chr, start, 'fwd', ref_chr_next, end, 'fwd','BND', read_name])

                else:  # --
                    if ref_chr < ref_chr_next:
                        start = alignment_current[2]
                        end = alignment_next[3]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[3]
                        end = alignment_current[2]

                    candidate.append([ref_chr, start, 'rev', ref_chr_next, end, 'rev','BND', read_name])
            else:
                if alignment_current[-1] == '+':  # +-
                    if ref_chr < ref_chr_next:
                        start = alignment_current[3]
                        end = alignment_next[3]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[3]
                        end = alignment_current[3]

                    candidate.append([ref_chr, start, 'fwd', ref_chr_next, end, 'rev', 'BND', read_name])
                else:  # -+
                    if ref_chr < ref_chr_next:
                        start = alignment_current[2]
                        end = alignment_next[2]
                    else:
                        ref_chr, ref_chr_next = ref_chr_next, ref_chr
                        start = alignment_next[2]
                        end = alignment_current[2]

                    candidate.append([ref_chr, start, 'rev', ref_chr_next, end, 'fwd', 'BND', read_name])
            #sv_signatures.append(SignatureTranslocation(*sv_sig))

#%%
def merge_intervals(intervals):

    merged = [intervals[0]]
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
            #print(i,j)
            if abs(infor[j][0] - infor[i][1]) <= 150:
                count = count + 1
                infor[i][1] = infor[j][1]
        #print(infor[i],count)
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
            #print(i,j)
            if abs(infor[j][1] - infor[i][1]) <= 150:
                count = count + 1
                infor[i][1] = infor[j][0]  #改[0]0
                lenth = lenth + infor[j][2]  #+ abs(infor[j][0] - infor[i][0])

        data.append([infor[i][0], infor[i][0] + 1, lenth])

        if count == 0:
            i += 1
        else:
            i += (count + 1)
    return data


def cigarread(read):
    candidate_ins = []
    candidate_del = []
    loci_clip_sm =[]
    loci_clip_ms = []
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
        elif ci[0] == 2:  #del
            if ci[1] >= 40 :
                cigar_del.append([sta, sta + ci[1], ci[1]])
            sta += ci[1]
        elif ci[0] == 1:  #ins

            if ci[1] >= 40 :
                cigar_ins.append([sta, sta, ci[1]])
    if len(cigar_del) != 0:
        cigar_del.sort(key=lambda x: x[0])
        cigar_del = mergecigar_del(cigar_del)
        data_del.extend(cigar_del)

    if len(cigar_ins) != 0:
        #print(cigar_ins)
        #print('+++++++++++++++++++++++++++++++')
        cigar_ins.sort(key=lambda x: x[0])
        cigar_ins = mergecigar_ins(cigar_ins)
        #print(long_ins)
        #print('===============================')
        data_ins.extend(cigar_ins)
    data_del = np.array(data_del)
    data_ins = np.array(data_ins)
    for del_cigar in data_del:
        candidate_del.append([del_cigar[0],
                          del_cigar[2],
                          read_name,
                          'None',
                          'DEL',
                          chr_name])
    for ins_ci in data_ins:
        candidate_ins.append([ins_ci[0],
                          ins_ci[2],
                          read_name,
                          'None',
                          'INS',
                          chr_name])
    if read.cigartuples[-1][0] == 4 or read.cigartuples[-1][0] == 5:
        if read.is_reverse:
            loci_clip_ms.append(read.reference_end)
        else:
            loci_clip_ms.append(read.reference_start)
    if read.cigartuples[0][0] == 4 or read.cigartuples[0][0] == 5:
        if read.is_reverse:
            loci_clip_sm.append(read.reference_end)
        else:
            loci_clip_sm.append(read.reference_start)

    return candidate_del, candidate_ins, loci_clip_sm, loci_clip_ms
#%%

#%%
def loci_read_count(read):
    loci_read = np.array(read.get_reference_positions())
    return loci_read
#%%
def analysis_cigar_indels(del_cigar,ins_cigar):
    sv_sig_del = []
    sv_sig_ins = []
    for cigar in del_cigar:
        data = np.arange(cigar[0],cigar[0]+cigar[1]).tolist()
        sv_sig_del.extend(data)
    for cigar in ins_cigar:
        data = np.arange(cigar[0],cigar[0]+1).tolist()
        sv_sig_ins.extend(data)
    return sv_sig_del,sv_sig_ins
#%%
def analysis_splitread_data(split_read_candidate):
    sv_sig_del = []
    sv_sig_ins = []
    sv_sig_inv = []
    sv_sig_dup = []
    sv_sig_bnd = []
    for sv_can in split_read_candidate:
        if sv_can[-2] == 'DEL' and sv_can[1]<= 1000000:
            data = np.arange(sv_can[0],sv_can[1]+sv_can[0]).tolist()
            sv_sig_del.extend(data)
        elif sv_can[-2] == 'INS':
            data = np.arange(sv_can[0],sv_can[0]+1).tolist()
            sv_sig_ins.extend(data)
        elif sv_can[-2] == 'INV' and  sv_can[1]<= 1000000:
            data = np.arange(sv_can[0],sv_can[1]+sv_can[0]).tolist()
            sv_sig_inv.extend(data)


        elif sv_can[-2] == 'DUP' and  sv_can[1]<= 1000000:
            data = np.arange(sv_can[0],sv_can[1]+sv_can[0]).tolist()
            sv_sig_dup.extend(data)

        elif sv_can[-2] == 'BND' :
            data = np.arange(sv_can[1],sv_can[1]+1).tolist()
            data1 = np.arange(sv_can[4],sv_can[4]+1).tolist()
            sv_sig_bnd.extend(data)
            sv_sig_bnd.extend(data1)
    return sv_sig_del,sv_sig_ins,sv_sig_inv,sv_sig_dup,sv_sig_bnd
#%%
def compute_loci(del_cigar_all_rev,ins_cigar_all_rev,del_split_all_rev,ins_split_all_rev,inv_split_all_rev,dup_split_all_rev,bnd_split_all_rev,loci_read_all_rev,clip_sm_all_rev,clip_ms_all_rev,del_cigar_all_fwd,ins_cigar_all_fwd,del_split_all_fwd,ins_split_all_fwd,inv_split_all_fwd,dup_split_all_fwd,bnd_split_all_fwd,loci_read_all_fwd,clip_sm_all_fwd,clip_ms_all_fwd,start,end):
    offset = int(end-start)
    if len(del_cigar_all_rev) != 0:
        del_cigar_all_rev = np.array(del_cigar_all_rev) - start
        del_cigar_all_rev = np.bincount(del_cigar_all_rev[del_cigar_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        del_cigar_all_rev = np.zeros([offset,1])

    if len(ins_cigar_all_rev) != 0:
        ins_cigar_all_rev = np.array(ins_cigar_all_rev) - start
        ins_cigar_all_rev = np.bincount(ins_cigar_all_rev[ins_cigar_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        ins_cigar_all_rev = np.zeros([offset,1])

    if len(del_split_all_rev) != 0:
        del_split_all_rev = np.array(del_split_all_rev) - start
        del_split_all_rev = np.bincount(del_split_all_rev[del_split_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        del_split_all_rev = np.zeros([offset,1])

    if len(ins_split_all_rev) != 0:
        ins_split_all_rev = np.array(ins_split_all_rev) - start
        ins_split_all_rev = np.bincount(ins_split_all_rev[ins_split_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        ins_split_all_rev = np.zeros([offset,1])

    if len(inv_split_all_rev) != 0:
        inv_split_all_rev = np.array(inv_split_all_rev) - start
        inv_split_all_rev = np.bincount(inv_split_all_rev[inv_split_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        inv_split_all_rev = np.zeros([offset,1])

    if len(dup_split_all_rev) != 0:
        dup_split_all_rev = np.array(dup_split_all_rev) - start
        dup_split_all_rev = np.bincount(dup_split_all_rev[dup_split_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        dup_split_all_rev = np.zeros([offset,1])

    if len(bnd_split_all_rev) != 0:
        bnd_split_all_rev = np.array(bnd_split_all_rev) - start
        bnd_split_all_rev = np.bincount(bnd_split_all_rev[bnd_split_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        bnd_split_all_rev = np.zeros([offset,1])

    if len(loci_read_all_rev) != 0:
        loci_read_all_rev = np.array(loci_read_all_rev) - start
        loci_read_all_rev = np.bincount(loci_read_all_rev[loci_read_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        loci_read_all_rev = np.zeros([offset,1])

    if len(clip_sm_all_rev) != 0:
        clip_sm_all_rev = np.array(clip_sm_all_rev) - start
        clip_sm_all_rev = np.bincount(clip_sm_all_rev[clip_sm_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        clip_sm_all_rev = np.zeros([offset,1])

    if len(clip_ms_all_rev) != 0:
        clip_ms_all_rev = np.array(clip_ms_all_rev) - start
        clip_ms_all_rev = np.bincount(clip_ms_all_rev[clip_ms_all_rev>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        clip_ms_all_rev = np.zeros([offset,1])

    if len(del_cigar_all_fwd) != 0:
        del_cigar_all_fwd = np.array(del_cigar_all_fwd) - start
        del_cigar_all_fwd = np.bincount(del_cigar_all_fwd[del_cigar_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        del_cigar_all_fwd = np.zeros([offset,1])

    if len(ins_cigar_all_fwd) != 0:
        ins_cigar_all_fwd = np.array(ins_cigar_all_fwd) - start
        ins_cigar_all_fwd = np.bincount(ins_cigar_all_fwd[ins_cigar_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        ins_cigar_all_fwd = np.zeros([offset,1])

    if len(del_split_all_fwd) != 0:
        del_split_all_fwd = np.array(del_split_all_fwd) - start
        del_split_all_fwd = np.bincount(del_split_all_fwd[del_split_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        del_split_all_fwd = np.zeros([offset,1])

    if len(ins_split_all_fwd) != 0:
        ins_split_all_fwd = np.array(ins_split_all_fwd) - start
        ins_split_all_fwd = np.bincount(ins_split_all_fwd[ins_split_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        ins_split_all_fwd = np.zeros([offset,1])

    if len(inv_split_all_fwd) != 0:
        inv_split_all_fwd = np.array(inv_split_all_fwd) - start
        inv_split_all_fwd = np.bincount(inv_split_all_fwd[inv_split_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        inv_split_all_fwd = np.zeros([offset,1])

    if len(dup_split_all_fwd) != 0:
        dup_split_all_fwd = np.array(dup_split_all_fwd) - start
        dup_split_all_fwd = np.bincount(dup_split_all_fwd[dup_split_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        dup_split_all_fwd = np.zeros([offset,1])

    if len(bnd_split_all_fwd) != 0:
        bnd_split_all_fwd = np.array(bnd_split_all_fwd) - start
        bnd_split_all_fwd = np.bincount(bnd_split_all_fwd[bnd_split_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        bnd_split_all_fwd = np.zeros([offset,1])

    if len(loci_read_all_fwd) != 0:
        loci_read_all_fwd = np.array(loci_read_all_fwd) - start
        loci_read_all_fwd = np.bincount(loci_read_all_fwd[loci_read_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        loci_read_all_fwd = np.zeros([offset,1])

    if len(clip_sm_all_fwd) != 0:
        clip_sm_all_fwd = np.array(clip_sm_all_fwd) - start
        clip_sm_all_fwd = np.bincount(clip_sm_all_fwd[clip_sm_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        clip_sm_all_fwd = np.zeros([offset,1])

    if len(clip_ms_all_fwd) != 0:
        clip_ms_all_fwd = np.array(clip_ms_all_fwd) - start
        clip_ms_all_fwd = np.bincount(clip_ms_all_fwd[clip_ms_all_fwd>=0],minlength = offset+1)[0:offset].reshape(-1,1)
    else:
        clip_ms_all_fwd = np.zeros([offset,1])




    #infor = np.concatenate([loci_read,del_split,ins_split,inv_split,dup_split,del_cigar,ins_cigar,clip_sm,clip_ms,data_depth],axis = 1)

    infor = concatnate_data(del_cigar_all_rev,ins_cigar_all_rev,del_split_all_rev,ins_split_all_rev,inv_split_all_rev,dup_split_all_rev,bnd_split_all_rev,loci_read_all_rev,clip_sm_all_rev,clip_ms_all_rev,del_cigar_all_fwd,ins_cigar_all_fwd,del_split_all_fwd,ins_split_all_fwd,inv_split_all_fwd,dup_split_all_fwd,bnd_split_all_fwd,loci_read_all_fwd,clip_sm_all_fwd,clip_ms_all_fwd)

    return fun(infor)

#%%
def concatnate_data(del_cigar_all_rev,ins_cigar_all_rev,del_split_all_rev,ins_split_all_rev,inv_split_all_rev,dup_split_all_rev,bnd_split_all_rev,loci_read_all_rev,clip_sm_all_rev,clip_ms_all_rev,del_cigar_all_fwd,ins_cigar_all_fwd,del_split_all_fwd,ins_split_all_fwd,inv_split_all_fwd,dup_split_all_fwd,bnd_split_all_fwd,loci_read_all_fwd,clip_sm_all_fwd,clip_ms_all_fwd):
    infor = np.concatenate([del_cigar_all_rev,ins_cigar_all_rev,del_split_all_rev,ins_split_all_rev,inv_split_all_rev,dup_split_all_rev,bnd_split_all_rev,loci_read_all_rev,clip_sm_all_rev,clip_ms_all_rev,del_cigar_all_fwd,ins_cigar_all_fwd,del_split_all_fwd,ins_split_all_fwd,inv_split_all_fwd,dup_split_all_fwd,bnd_split_all_fwd,loci_read_all_fwd,clip_sm_all_fwd,clip_ms_all_fwd],axis = 1)
    return infor
#%%
def fun(a):
    a = a.astype(np.float32)
    a -= a.mean(axis =0)
    a /= (np.sqrt(a.var(axis =0))+1e-10)
    return a
#%%
def labeldata(vcfpath, contig, start, end, window_size, index):
    goldl = []
    if ('chr' in contig):
        contig = contig[3:]
    for rec in pysam.VariantFile(vcfpath).fetch():

        if (rec.contig != contig):
            continue
        if ((rec.info['SVTYPE'] == 'INS') or (rec.info['SVTYPE'] == 'DEL') or (rec.info['SVTYPE'] == 'INV') or (
                rec.info['SVTYPE'] == 'DUP')):
            goldl.append([rec.start, rec.stop, (rec.stop) - (rec.start), 1])
    goldl = (pd.DataFrame(goldl).sort_values([0, 1]).values).astype('float64')

    y = []
    for rec in index:
        if (((goldl[:, 1:2] > rec) & (goldl[:, :1] < (rec + window_size))).sum() != 0):
            y.append((((goldl[:, 1:2] >= rec) & (goldl[:, :1] <= (rec + window_size))) * goldl[:, 3:]).sum())

        else:
            y.append(0)
    return (np.array(y) > 0).astype('float32')


def labelbed(bed_file, contig, start, end, window_size, index):
    bnd_data = pd.read_csv(bed_file, sep='\t', header=None).values.tolist()
    bnd_data = sorted(bnd_data, key=lambda x: x[-1])
    chr_values, start_values, svlen_values, svtype_values = [], [], [], []
    for i in range(len(bnd_data)):
        if bnd_data[i][4] != 'TRA':
            chr_values.append(bnd_data[i][0])
            start_values.append(bnd_data[i][1])
            svlen_values.append(bnd_data[i][3] - bnd_data[i][1])
        else:
            chr_values.append(bnd_data[i][0])
            start_values.append(bnd_data[i][1])
            svlen_values.append(bnd_data[i][2] + ':' + str(bnd_data[i][3]))
        svtype_values.append(bnd_data[i][4])

    result_dict = {}
    for i in range(len(chr_values)):
        if chr_values[i] in result_dict:
            result_dict[chr_values[i]].append([start_values[i], svlen_values[i], svtype_values[i]])
        else:
            result_dict[chr_values[i]] = [[start_values[i], svlen_values[i], svtype_values[i]]]

    goldl = []
    if ('chr' in contig):
        contig = contig[3:]
    for rec in result_dict[contig]:

        if ((rec[2] == 'DEL') or (rec[2] == 'INV') or (rec[2] == 'DUP')):
            #print(rec[0], rec[0] + rec[1], rec[1], 1)
            goldl.append([rec[0], rec[0] + rec[1], rec[1], 1])
        elif (rec[2] == 'TRA' or rec[2] == 'INS'):
            goldl.append([rec[0], rec[0] + 1, 1, 1])
    goldl = (pd.DataFrame(goldl).sort_values([0, 1]).values).astype('float64')

    #print(goldl)
    y = []
    for rec in index:
        if (((goldl[:, 1:2] >= rec) & (goldl[:, :1] <= (rec + window_size))).sum() != 0):
            y.append((((goldl[:, 1:2] >= rec) & (goldl[:, :1] <= (rec + window_size))) * goldl[:, -1]).sum())

        else:
            y.append(0)
    return (np.array(y) > 0).astype('float32')
#%%
def create_data(bamfile_long_path,chr_name,start,end):
    read_count = 0
    bamfile = pysam.AlignmentFile(bamfile_long_path,'rb',threads = 20)

    timem = time.time()

    loci_read_all_rev = []
    loci_read_all_fwd = []
    del_cigar_all_rev = []
    ins_cigar_all_rev = []
    clip_sm_all_rev = []
    clip_sm_all_fwd = []
    clip_ms_all_rev = []
    clip_ms_all_fwd = []
    del_cigar_all_fwd = []
    ins_cigar_all_fwd = []
    split_read_candidate_fwd = list()
    split_read_candidate_rev = list()
    for read in bamfile.fetch(chr_name,start,end):
        if read.is_unmapped or read.is_duplicate or read.is_secondary or read.is_supplementary:
            continue

        read_flag = detect_flag(read.flag)
        if read_flag%2 == 0:
            del_cigar,ins_cigar,clip_sm,clip_ms = cigarread(read)
            del_cigar_all_rev.extend(del_cigar)
            ins_cigar_all_rev.extend(ins_cigar)
            clip_sm_all_rev.extend(clip_sm)
            clip_ms_all_rev.extend(clip_ms)
            if read.has_tag('SA'):
                splitread = splitreadlist(read)
                analyze_read_segments(read,feature_read_segement(splitread),split_read_candidate_rev)
            loci_read = loci_read_count(read)
            loci_read_all_rev.extend(loci_read)
        else:
            del_cigar,ins_cigar,clip_sm,clip_ms = cigarread(read)
            del_cigar_all_fwd.extend(del_cigar)
            ins_cigar_all_fwd.extend(ins_cigar)
            clip_sm_all_fwd.extend(clip_sm)
            clip_ms_all_fwd.extend(clip_ms)
            if read.has_tag('SA'):
                splitread = splitreadlist(read)
                analyze_read_segments(read,feature_read_segement(splitread),split_read_candidate_fwd)

            loci_read = loci_read_count(read)
            loci_read_all_fwd.extend(loci_read)

    del_cigar_all_rev,ins_cigar_all_rev = analysis_cigar_indels(del_cigar_all_rev,ins_cigar_all_rev)
    del_split_all_rev,ins_split_all_rev,inv_split_all_rev,dup_split_all_rev,bnd_split_all_rev = analysis_splitread_data(split_read_candidate_rev)

    del_cigar_all_fwd,ins_cigar_all_fwd = analysis_cigar_indels(del_cigar_all_fwd,ins_cigar_all_fwd)
    del_split_all_fwd,ins_split_all_fwd,inv_split_all_fwd,dup_split_all_fwd,bnd_split_all_fwd = analysis_splitread_data(split_read_candidate_fwd)

    all_data = compute_loci(del_cigar_all_rev,ins_cigar_all_rev,del_split_all_rev,ins_split_all_rev,inv_split_all_rev,dup_split_all_rev,bnd_split_all_rev,loci_read_all_rev,clip_sm_all_rev,clip_ms_all_rev,del_cigar_all_fwd,ins_cigar_all_fwd,del_split_all_fwd,ins_split_all_fwd,inv_split_all_fwd,dup_split_all_fwd,bnd_split_all_fwd,loci_read_all_fwd,clip_sm_all_fwd,clip_ms_all_fwd,start,end)




    gc.collect()
    time_end = time.time() - timem
    print('------------',chr_name,start,end)
    bamfile.close()
    return all_data
#%%
def process_region(bamfile_long_path, chr_name_long, start, end, outputpath,window_size):
    time_s = time.time()

    print("Executing process_region",start,end)
    all_data = create_data(bamfile_long_path,chr_name_long,start,end)#bamfile_long_path,chr_name,start,end,mapquality,alignedlength
    index = np.arange(start,end,window_size)
    index_all = []
    all_data = all_data.reshape(-1,all_data.shape[1]*window_size)
    index = np.array(index)

    x_data = np.array(all_data)
    index_all = np.array(index)
    print(x_data.shape,index_all.shape)
    all_data = x_data
    print(all_data.shape)
    if 'chr' in chr_name_long:
        filename_data = outputpath + '/'   + chr_name_long + '_' + str(start)+ '_' + str(end) +'.npy'
        filename_index = outputpath + '/'  + chr_name_long + '_' + str(start)+ '_' + str(end) +'_index.npy'

    else:
        filename_data = outputpath + '/chr'   + chr_name_long + '_' + str(start)+ '_' + str(end) +'.npy'
        filename_index = outputpath + '/chr'  + chr_name_long + '_' + str(start)+ '_' + str(end) +'_index.npy'

    np.save(filename_data,all_data)
    np.save(filename_index,index_all)
    gc.collect()
    print("Done",chr_name_long,start,end,time.time()-time_s)


#%%
def create_data_long(bamfile_long_path,outputpath,contig,window_size,threadss= 10 ):
    time_st = time.time()
    #print(bamfile_long_path)
    threadss = int(threadss)
    bamfile_long = pysam.AlignmentFile(bamfile_long_path,'rb',threads = threadss)
    ref_name_long = bamfile_long.get_reference_name
    chr_length_long = bamfile_long.lengths
    contig2length = {}

    if len(contig) == 0:
        contig = []
        for count in range(len(bamfile_long.get_index_statistics())):
            contig.append(bamfile_long.get_index_statistics()[count].contig)
            contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]
    else:
        contig = np.array(contig).astype(str)
    for count in range(len(bamfile_long.get_index_statistics())):
        contig2length[bamfile_long.get_index_statistics()[count].contig] = bamfile_long.lengths[count]

    for ww in contig:
        chr_name_long = ww
        chr_length = contig2length[ww]
        ider = math.ceil(chr_length/1e7)
        start = 0
        end = int(1e7)
        print(chr_name_long,ider)
        time_q = time.time()
        countt = 0
        while ( countt < ider):
            if(len(multiprocessing.active_children()) < int(threadss)):

                p = Process(target=process_region, args=(bamfile_long_path, chr_name_long, start, end,outputpath,window_size))

                start += int(1e7)
                end += int(1e7)
                p.start()
                countt += 1

            else:
                time.sleep(2)

        print(time.time()-time_q)

    # 等待所有任务完成




    print(time.time()-time_st)


    #create_data_long(bamfile_long_path,outputpath,contig,window_size =2000,threadss = 15)