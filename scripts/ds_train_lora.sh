DATE=`date +%Y%m%d`
dataset="alpaca_zh"
LORA_RANK=256
LORA_A=512
LORA_DROPOUT=0.1
LR=5e-4
EPOCH=5
max_length=2048
model=Baichuan2-13B-Chat
exp_name=${dataset}.${model}.raw.lora.${max_length}
exp_name+=.${DATE}.lr${LR}.E${EPOCH}.l_rank${LORA_RANK}.l_alpha${LORA_A}.l_drop${LORA_DROPOUT}
SAVE=output/${dataset}/${DATE}/${exp_name}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
mkdir -p ${SAVE}
    # --lora_target q_proj,k_proj,vs_proj,o_proj \
#     --lora_target W_pack,o_proj \
    # --deepspeed ds_config_zero2_offload.json \
    # --deepspeed ds_configs/ds_config_zero3.json \

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port $MASTER_PORT src/train_bash.py \
    --deepspeed ds_configs/ds_config_zero3.json \
    --stage sft \
    --do_train \
    --max_sample 10000 \
    --dataset alpaca_zh \
    --overwrite_cache True \
    --model_name_or_path official_model/${model} \
    --output_dir ${SAVE} \
    --run_name ${exp_name} \
    --overwrite_output_dir \
    --cutoff_len ${max_length} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --preprocessing_num_workers 8 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs ${EPOCH} \
    --learning_rate $LR \
    --finetuning_type lora \
    --lora_target W_pack,o_proj \
    --lora_rank $LORA_RANK \
    --lora_alpha ${LORA_A} \
    --lora_dropout ${LORA_DROPOUT} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy epoch \
    --plot_loss \
    --template baichuan2 \
    --fp16 \
    --report_to wandb \
    2>&1 | tee ${SAVE}/log.txt



MASTER_PORT=$(shuf -n 1 -i 10000-65535)
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port $MASTER_PORT src/train_bash.py \
    --stage sft \
    --model_name_or_path official_model/${model} \
    --adapter_name_or_path ${SAVE} \
    --max_sample 100 \
    --do_predict \
    --dataset alpaca_gpt4_zh \
    --overwrite_cache True \
    --finetuning_type lora \
    --output_dir ${SAVE}/predict \
    --overwrite_cache \
    --per_device_eval_batch_size 1 \
    --preprocessing_num_workers 8 \
    --cutoff_len ${max_length} \
    --predict_with_generate \
    --plot_loss \
    --fp16 \
    --template baichuan2
