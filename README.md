# Text Generation from Knowledge Graphs with Graph Transformers

This is a Pytorch re-implementation of the paper, [Text Generation from Knowledge Graphs with Graph Transformers](https://google.com), which is accepted for publication at [NAACL 2019](http://naacl2019.org/).


# Instructions

Training:
```
CUDA_VISIBLE_DEVICES=5 python train.py -save tmp_useless
CUDA_VISIBLE_DEVICES=5 python try_train.py -save tmp_useless
CUDA_VISIBLE_DEVICES=6 python train.py -title -save tmp_no_title
```
Use ``--help`` for a list of all training options.

To generate, use 
```
python try_generate.py -ckpt ~/proj/GraphWriter/tmp/19.vloss-3.609007.lr-0.1 -save ~/proj/GraphWriter/tmp

python3.6 generator.py -save <SAVED MODEL>
``` 
with the appropriate model flags used to train the model

To evaluate, run
```
python eval.py ~/proj/GraphWriter/outputs/tmp.pred ~/proj/GraphWriter/outputs/tmp.gold
python eval/bleu.py -refs ~/proj/GraphWriter/outputs/tmp.gold -hyps ~/proj/GraphWriter/outputs/tmp.pred -verbose

python3.6 eval.py <GENERATED TEXTS> <GOLD TARGETS>
```


# AGENDA Dataset

The AGENDA dataset is available in a user-friendly json format in /data/unprocessed.tar.gz
Preprocessed data is also available in /data.


## Citation
If this work is useful in your research, please cite our paper.
```
@inproceedings{koncel2019text,
  title={{T}ext {G}eneration from {K}nowledge {G}raphs with {G}raph {T}ransformers},
  author={Rik Koncel-Kedziorski, Dhanush Bekal, Yi Luan, Mirella Lapata, and Hannaneh Hajishirzi},
  booktitle={NAACL},
  year={2019}
}
```

