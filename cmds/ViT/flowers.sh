
save_dir="."

devices="0,1"
port=7256
n_gpu=2

backPruneRatio=0.8
lr=3e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name flowers-lr${lr}-B128-BackRazor${backPruneRatio} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset flowers --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 2000 --eval_every 1000
