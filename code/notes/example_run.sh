
###############################################################################
####  Example run on CoNLL dev (i.e., 2013 data)
# The CoNLL dev/test data and the M^2 scorer are available here: http://www.comp.nus.edu.sg/~nlp/conll14st.html
###############################################################################


REPO_DIR=
cd ${REPO_DIR}/code/_dev/constrained


SES=2  # gpu id (if not included, uses CPU)
DATA_DIR= # conll2013 data dir


MODEL_DIR=
MODEL_FILE=

TUNING_WEIGHT=0.6  # value found via search on the dev set

OUTPUT_FILE=
OUTPUT_LOG_FILE=


python translate.py \
-src ${DATA_DIR}/revised_official-preprocessed.m2.source_sents.txt \
-tgt ${DATA_DIR}/revised_official-preprocessed.m2.source_sents.txt \
-output ${OUTPUT_FILE} \
-model ${MODEL_DIR}/${MODEL_FILE} \
-beam_size 10 \
-batch_size 10 \
-min_length 0 \
-max_length 274 \
-replace_unk \
-n_best 1 \
-verbose \
-gpu ${SES} \
-tag_weight ${TUNING_WEIGHT} >${OUTPUT_LOG_FILE}


# remove the diff tags:
python ${REPO_DIR}/code/post_processing/tags_to_sentences.py -i ${OUTPUT_FILE} -o ${OUTPUT_FILE}_nosyms.txt

# run the M^2 scorer on the ${OUTPUT_FILE}_nosyms.txt file
