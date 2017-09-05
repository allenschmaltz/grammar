This is an earlier version of OpenNMT used for training, tuning, and testing the core models with diffs in the paper "Adapting Sequence Models for Sentence Correction", EMNLP 2017.

Note that an option -tag_weight for translate.lua allows an additive weight to be applied to the four diff tags.

# OpenNMT: Open-Source Neural Machine Translation

The most recent version and documentation for OpenNMT is available here: https://github.com/opennmt/opennmt

## Installation

OpenNMT only requires a vanilla Torch install with few dependencies.

### Dependencies

* `nn`
* `nngraph`
* `tds`
* `penlight`

GPU training requires:

* `cunn`
* `cutorch`

Multi-GPU training additionally requires:

* `threads`

## Quickstart

OpenNMT consists of three commands:

1) Preprocess the data.

```th preprocess.lua -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo```

2) Train the model.

```th train.lua -data data/demo-train.t7 -save_model model```

3) Translate sentences.

```th translate.lua -model model_final.t7 -src data/src-test.txt -tag_weight 0.0 -output pred.txt```

## OpenNMT Citation

A <a href="https://arxiv.org/abs/1701.02810">technical report</a> on OpenNMT is available. If you use the system for academic work, please cite:

```
    @ARTICLE{2017opennmt,
         author = { {Klein}, G. and {Kim}, Y. and {Deng}, Y.
                    and {Senellart}, J. and {Rush}, A.~M.},
         title = "{OpenNMT: Open-Source Toolkit
                   for Neural Machine Translation}",
         journal = {ArXiv e-prints},
         eprint = {1701.02810} }
```
