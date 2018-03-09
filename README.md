# grammar


Under development.

A trained model (trained on the NUCLE + Lang-8 data) is available here:

https://drive.google.com/drive/folders/1Dsdp4Pgtfm-_MW5tOnQ26OkJuljnyAL7?usp=sharing

This is for use with the code in https://github.com/allenschmaltz/grammar/tree/master/code/_dev/constrained. An example of decoding the CoNLL dev set appears in code/notes/example_run.sh.

For reference, the effectiveness on the CoNLL dev/test data is slightly higher than that of the original model used in the EMNLP paper. (This is due to the model and not due to changes in decoding.)

Dev:

Precision   : 0.4877
Recall      : 0.1532
F_0.5       : 0.3394

Test:

Precision   : 0.5375
Recall      : 0.2404
F_0.5       : 0.4310

## Citation/Reference

"Adapting Sequence Models for Sentence Correction", EMNLP 2017
