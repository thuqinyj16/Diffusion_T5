cd /apdcephfs/share_47076/yujiaqin/CrossFit_prompt_T5

TASK="glue-sst2"

# TASKS="superglue-rte tweet_eval-sentiment discovery glue-rte superglue-wsc scicite glue-mrpc tweet_eval-stance_hillary tweet_eval-offensive emotion hatexplain glue-cola sick paws ethos-sexual_orientation glue-qqp tweet_eval-emotion sms_spam health_fact glue-mnli imdb ethos-disability glue-wnli scitail trec-finegrained yahoo_answers_topics liar glue-sst2 tweet_eval-stance_abortion circa tweet_eval-stance_climate glue-qnli tweet_eval-emoji ethos-directed_vs_generalized ade_corpus_v2-classification wiki_auto hate_speech_offensive superglue-wic google_wellformed_query tweet_eval-irony ethos-gender onestop_english trec rotten_tomatoes kilt_fever"

IDENTIFIER=diffusion_full_data_prompt
SEED_IDX=$(( ($TASK_INDEX) % 160 + $OFFSET ))

echo "seed_idx: $SEED_IDX"

echo "Task: $TASK, Full data prompt tuning, Identifier: $IDENTIFIER"

python tune_hps_singletask_t5large_full.py \
--task_dir /apdcephfs/share_47076/yujiaqin/CrossFit_T5/data_full_new/${TASK}/ \
--do_train \
--do_predict \
--learning_rate_list 0.3 \
--bsz_list 16 \
--total_steps 100000 \
--eval_period 100 \
--warmup_steps 0 \
--model t5.1.1.lm100k.base \
--output_dir models/${IDENTIFIER}/singletask-${TASK}/$SEED_IDX \
--predict_batch_size 8 \
--do_prompt \
--gradient_accumulation_steps 1 \
--seed $SEED_IDX

done
