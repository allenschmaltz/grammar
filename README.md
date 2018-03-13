# grammar


Under development.

A trained model (trained on the NUCLE + Lang-8 data) is available here (in the folder w_n_w_l8_pyrep_s79_t100_rnn750_la2_v50000_brnn_opennmt_b48):

https://drive.google.com/drive/folders/1Dsdp4Pgtfm-_MW5tOnQ26OkJuljnyAL7?usp=sharing

This is for use with the code in https://github.com/allenschmaltz/grammar/tree/master/code/_dev/constrained, which implements constrained decoding (i.e., changes not conforming to the diff tag semantics are not allowed during beam search). An example of decoding the CoNLL dev set appears in [example_run.sh](code/notes/example_run.sh).

For reference, the effectiveness on the CoNLL dev/test data is slightly higher than that of the original model used in the EMNLP paper. (This is due to the model and not due to changes in decoding.) Results for unconstrained and constrained decoding are included below (and are not significantly different):

Unconstrained decoding followed by a post-hoc fix of the tags:

Dev:

Precision: 0.49; Recall: 0.15; F_0.5: 0.34

Test:

Precision: 0.54; Recall: 0.24; F_0.5: 0.43

Constrained decoding:

Dev:

Precision: 0.49; Recall: 0.15; F_0.5: 0.34

Test:

Precision: 0.53; Recall: 0.24; F_0.5: 0.43

For historical purposes, the model used in the original paper (for use with the [original code](code/replication_fork)) is available at the above link in the folder legacy/word_nucle_w_lang8_v1_srclen79_trglen100_rnn750_la2_v50000_brnn_opennmt_batchsize48.

## License (for trained models)

The models linked above are provided solely for research purposes and are provided as-is without warranty of any kind. They were trained with the NUCLE and Lang-8 data (see references in the paper cited below) and usage must conform to the original licenses of those data sources.

## Citation/Reference

Allen Schmaltz, Yoon Kim, Alexander Rush, and Stuart Shieber. 2017. Adapting sequence models for sentence correction. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, pages 2807-2813. https://www.aclweb.org/anthology/D17-1298. ([Appendix](http://aclweb.org/anthology/attachments/D/D17/D17-1298.Attachment.zip)) ([.bib](http://aclweb.org/anthology/D/D17/D17-1298.bib))
