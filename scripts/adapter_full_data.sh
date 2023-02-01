cd /apdcephfs/share_47076/yujiaqin/diffusion_T5

TASKS=(acronym_identification ade_corpus_v2-classification ade_corpus_v2-dosage ade_corpus_v2-effect adversarialqa aeslc ag_news ai2_arc amazon_polarity anli app_reviews aqua_rat art aslg_pc12 biomrc blimp-anaphor_gender_agreement blimp-anaphor_number_agreement blimp-determiner_noun_agreement_with_adj_irregular_1 blimp-ellipsis_n_bar_1 blimp-ellipsis_n_bar_2 blimp-existential_there_quantifiers_1 blimp-irregular_past_participle_adjectives blimp-sentential_negation_npi_licensor_present blimp-sentential_negation_npi_scope blimp-wh_questions_object_gap boolq break-QDMR break-QDMR-high-level circa climate_fever codah common_gen commonsense_qa cos_e cosmos_qa crawl_domain crows_pairs dbpedia_14 definite_pronoun_resolution discovery dream duorc e2e_nlg_cleaned eli5-askh eli5-asks eli5-eli5 emo emotion empathetic_dialogues ethos-directed_vs_generalized ethos-disability ethos-gender ethos-national_origin ethos-race ethos-religion ethos-sexual_orientation financial_phrasebank freebase_qa gigaword glue-cola glue-mnli glue-mrpc glue-qnli glue-qqp glue-rte glue-sst2 glue-wnli google_wellformed_query hate_speech18 hate_speech_offensive hatexplain health_fact hellaswag hotpot_qa imdb jeopardy kilt_ay2 kilt_fever kilt_hotpotqa kilt_nq kilt_trex kilt_wow kilt_zsre lama-conceptnet lama-google_re lama-squad lama-trex liar limit math_qa mc_taco medical_questions_pairs mocha multi_news numer_sense onestop_english openbookqa paws piqa poem_sentiment proto_qa qa_srl qasc quail quarel quartz-no_knowledge quartz-with_knowledge quoref race-high race-middle reddit_tifu-title reddit_tifu-tldr ropes rotten_tomatoes samsum scicite sciq scitail search_qa sick sms_spam social_i_qa spider squad-no_context squad-with_context superglue-cb superglue-copa superglue-multirc superglue-record superglue-rte superglue-wic superglue-wsc swag tab_fact trec trec-finegrained tweet_eval-emoji tweet_eval-emotion tweet_eval-hate tweet_eval-irony tweet_eval-offensive tweet_eval-sentiment tweet_eval-stance_abortion tweet_eval-stance_atheism tweet_eval-stance_climate tweet_eval-stance_feminist tweet_eval-stance_hillary tweet_qa web_questions wiki_auto wiki_bio wiki_qa wiki_split wikisql wino_grande wiqa xsum yahoo_answers_topics yelp_polarity yelp_review_full)

TASK=${TASKS[$(( ($TASK_INDEX) % 160 ))]}

# TASKS="glue-sst2"
DATA_DIR=/apdcephfs/share_47076/yujiaqin/CrossFit_T5/data_full_new
TUNE_METHOD=adapter
SAVE_PATH=models
IDENTIFIER=full_data_adapter_size2
PRETRAINED_MODEL_PATH=/apdcephfs/share_47076/yujiaqin/CrossFit_T5/pretrained_models
lrs="2e-4"

for lr in $lrs
do

echo "Seed: $SEED, Task: $TASK, Identifier: $IDENTIFIER"

python tune_singletask.py \
--task_dir ${DATA_DIR}/${TASK} \
--do_train \
--do_predict \
--learning_rate_list ${lr} \
--bsz_list 32 \
--train_iters 15000 \
--model ${PRETRAINED_MODEL_PATH}/t5.1.1.lm100k.base \
--tokenizer_path ${PRETRAINED_MODEL_PATH}/t5-v1_1-base \
--output_dir ${SAVE_PATH}/${IDENTIFIER}_lr${lr}/${TASK}/${SEED} \
--predict_batch_size 32 \
--tune_method ${TUNE_METHOD} \
--valid_interval 100 \
--output_interval 1000000 \
--log_interval 100 \
--one_prefix \
--seed ${SEED} \
--apply_adapter \
--adapter_type houlsby \
--adapter_size 2 \
--SGD_noise \
--load_init_seed ${SEED} \
--choose_dev_1000

done
