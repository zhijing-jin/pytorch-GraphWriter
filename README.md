# Text Generation from Knowledge Graphs with Graph Transformers

This is a Pytorch re-implementation of the paper, [Text Generation from Knowledge Graphs with Graph Transformers](https://google.com), which is accepted for publication at [NAACL 2019](http://naacl2019.org/).


# Instructions

Training:
```
python train.py -save tmp
```
To generate, use 
```
python try_generate.py -ckpt <PATH_TO_MODEL> -save tmp
``` 
with the appropriate model flags used to train the model

To evaluate, run
```
# get BLEU1, BLEU2, BLEU3, BLEU4
python eval.py ~/proj/GraphWriter/outputs/tmp.pred ~/proj/GraphWriter/outputs/tmp.gold

# get official bleu score
python eval/bleu.py -refs outputs/tmp.gold -hyps outputs/tmp.pred -verbose
```


# AGENDA Dataset

The AGENDA dataset is available in a user-friendly json format in /data/unprocessed.tar.gz
Preprocessed data is also available in /data.


