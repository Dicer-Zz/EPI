gpu_id=0
seed=42

batch_size=32
epochs=5
dataset=5ds
order="ag yelp amazon yahoo dbpedia"
n_per_class=2000

method=epi
model_name_or_path=bert-base-uncased
rep_mode=avg
query_mode=mahalanobis
prompt_mode=prefix
# lr=0.03
# pre_seq_len=16

log_file=logs/${dataset}_${method}_${prompt_mode}.log

for lr in 0.03 0.01 0.05 0.005; do
    for pre_seq_len in 4 8 16 32; do
        python run.py \
            --dataset ${dataset} \
            --method ${method} \
            --model_name_or_path ${model_name_or_path} \
            --rep_mode ${rep_mode} \
            --query_mode ${query_mode} \
            --prompt_mode ${prompt_mode} \
            --pre_seq_len ${pre_seq_len} \
            --n_per_class ${n_per_class} \
            --batch_size ${batch_size} \
            --lr ${lr} \
            --epochs ${epochs} \
            --gpu_id ${gpu_id} \
            --seed ${seed} >> ${log_file}
    done
done