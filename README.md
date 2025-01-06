# SVHunter
SVHunter is a long-read-based structural variation detection through transformer model.
SVHunter can detect and genotype DEL/INS/DUP/INV/TRA.
The shared pre-trained model is compatible with CCS, CLR, and ONT sequencing data, allowing for structural variation detection across these platforms without the need for additional training or fine-tuning.

## Installation
### Requirements
* python 3.11, numpy, pandas, TensorFlow 2.12.1, pysam, math,  scikit-learn 1.5.1
### 1. Create a virtual environment  
```
#create
conda create -n SVHunter python=3.11
#activate
conda activate SVHunter
#deactivate
conda deactivate
```   
### 2. clone SVHunter
* After creating and activating the SVHunter virtual environment, download SVHunter from github:
```　 
git clone https://github.com/eioyuou/SVHunter.git
cd SVHunter
```
### 3. Install 
```　
conda activate SVHunter
conda install python 3.11, numpy, pandas, TensorFlow 2.12.1, pysam, math,  scikit-learn 1.5.1
```
## Usage
### 1.Produce data for call SV
```　 
python SVHunter.py generate bamfile_path_long output_data_folder thread includecontig(default:[](all chromosomes))
    
bamfile_path_long is the path to the alignment file between the reference genome and the long-read dataset.;    
output_data_folder is the directory used to store the generated data.;  
thread specifies the number of threads to use;  
includecontig is the list of contigs to perform detection on (default: [], meaning all contigs are used).
   
eg: python SVHunter.py generate ./long_read.bam ./outpath 16 [12,13,14,15,16,17,18,19,20,21,22] 

``` 
### 2.Call SV 

SVHunter achieves optimal performance with GPU acceleration. While it can be run on a CPU, the computational efficiency will be significantly lower. Users are encouraged to use a GPU for the best experience.
```　 
python SVHunter.py call predict_weight,datapath,bamfilepath,predict_path,outvcfpath,thread,includecontig,num_gpus

Parameters:
- predict_weight: path to the model weights file
- datapath: folder used to store evaluation data 
- bamfilepath: path to the alignment file between reference and long read set
- predict_path: path for model prediction data
- outvcfpath: path for output vcf file
- thread: number of threads to use
- includecontig: list of contigs to perform detection on (default: [], meaning all contigs will be used)
- num_gpus: number of GPUs to use (optional parameter, default: auto-detect and use all available GPUs)

Usage examples:
# Use default GPU configuration
python SVHunter.py call ./predict_weight.h5 ./datapath ./long_read.bam ./predict_path ./outvcfpath 10 [12,13,14,15,16,17,18,19,20,21,22]

# Specify using 2 GPUs
python SVHunter.py call ./predict_weight.h5 ./datapath ./long_read.bam ./predict_path ./outvcfpath 10 [12,13,14,15,16,17,18,19,20,21,22] 2

# Use all contigs and specify 1 GPU
python SVHunter.py call ./predict_weight.h5 ./datapath ./long_read.bam ./predict_path ./outvcfpath 10 [] 1  
```  


