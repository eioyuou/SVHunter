# Description: This script is used to call SVs using the trained model. The script can be used to generate data for training the model or to call SVs using the trained model.
import os
import tensorflow as tf
import sys
from SVHunter_generate_data import create_data_long
# from insertion_model import gru_model
from SVHunter_detect import cluster_by_predict, model_predict
import ast
import tensorflow as tf
import time
mode = sys.argv[1]
print('len',len( sys.argv))
debug = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#print('-----------------')
#print(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8],sys.argv[9])
#print('-----------------')
def parse_contigg(contigg_str):
    if contigg_str.strip() == '[]':
        return []

    contigg_str = contigg_str.strip('[]')
    contigg_list = contigg_str.split(',')
    contigg_list = [item.strip().strip("'").strip('"') for item in contigg_list]

    return contigg_list

if (mode == 'generate'):
    if (len(sys.argv) not in [4, 5, 6]):
        debug = 1
    else:
        print('Produce data')
        print(len(sys.argv))
        if (len(sys.argv) == 6):
            contigg_str = sys.argv[5]
            contigg = parse_contigg(contigg_str)
            bamfilepath_long, outputpath, max_work, includecontig = sys.argv[2], sys.argv[3], sys.argv[4], contigg
        if (len(sys.argv) == 5):
            bamfilepath_long, outputpath, max_work, includecontig = sys.argv[2], sys.argv[3], sys.argv[4], []
        if (len(sys.argv) == 4):
            bamfilepath_long, outputpath, max_work, includecontig = sys.argv[2], sys.argv[3], 15, []
        print('bamfile path: ', bamfilepath_long)
        print('output path: ', outputpath)
        print('max_worker: ', max_work)
        print('includecontig: ', includecontig)
        # print('max_worker set to ', str(max_worker))
        if (includecontig == []):
            print('All chromosomes within bamfile will be used:')
            create_data_long(bamfilepath_long, outputpath, includecontig, 2000,
                                 max_work)
        else:
            print('Following chromosomes will be used:')
            print(includecontig)
            # bamfile_long_path,outputpath,contig,window_size =2000,threadss = 15)
            create_data_long(bamfilepath_long, outputpath, includecontig,2000,
                                 max_work)
        print('\n\n')
        print('Completed.')
        print('\n\n')

elif (mode == 'call'):
    if (len(sys.argv) not in [7, 8, 9, 10]):
        debug = 1
    else:
        timenow = time.time()
        print('testing')
        if (len(sys.argv) == 10):
                contigg_str = sys.argv[8]
                print('contigg_str:',len(contigg_str))
                if len (contigg_str) == 0:
                    contigg = []
                else:
                    contigg = parse_contigg(contigg_str)
                
                predict_weight, datapath, bamfilepath, predict_path, outputvcfpath, thread, num_nv= sys.argv[2], sys.argv[3], \
                sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[9]

        elif (len(sys.argv) == 9):
            contigg_str = sys.argv[8]
            contigg = parse_contigg(contigg_str)
            predict_weight, datapath, bamfilepath, predict_path, outputvcfpath, contigg, thread = sys.argv[2], sys.argv[3], \
            sys.argv[4], sys.argv[5], sys.argv[6],   contigg, sys.argv[7]
        elif (len(sys.argv) == 8):
            predict_weight, datapath, bamfilepath, predict_path, outputvcfpath, thread ,contigg = sys.argv[2], sys.argv[3], \
            sys.argv[4], sys.argv[5], sys.argv[6],sys.argv[7], []
        else:
            predict_weight, datapath, bamfilepath, predict_path, outputvcfpath, thread, contigg = sys.argv[2], \
            sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], 15, []
        print('bamfile path: ', bamfilepath)
        print('weight path: ', predict_weight)
        print('data file path: ', datapath)
        print('predict path: ', predict_path)
        print('vcf path: ', outputvcfpath)
        print('thread: ', thread)
        if (contigg == []):
            print('All chromosomes within bamfile will be used')
        else:
            print('Following chromosomes will be used')
            print(contigg)

        model_predict(predict_weight,bamfilepath, datapath,predict_path, contigg)
        print('\n\n')
        print('Completed, Predict result saved in current folder.')
        cluster_by_predict(bamfilepath, datapath, predict_path, outputvcfpath, contigg, thread)
        print('\n\n')
        print('Completed, Result saved in outputvcfpath folder.')
        print('\n\n')
        print('Time taken', time.time() - timenow)

else:
    debug = 1


if (debug == 1):
    print('\n\n')
    print('Useage:')
    print('Produce data for call sv')
    print(
        'python SVHunter.py generate bamfile_path_long output_data_folder max_work includecontig(default:[](all chromosomes))')
    print('Call sv:')
    print(
        'python SVHunter.py call predict_weight,datapath,bamfilepath,predict_path ,outvcfpath, thread,includecontig(default:[](all chromosomes)')
# In[ ]:
