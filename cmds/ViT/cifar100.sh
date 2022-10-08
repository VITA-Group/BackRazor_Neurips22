save_dir="."

devices="4,5,6,7"
port=7258
n_gpu=4


backPruneRatio=0.9
lr=0.01

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name cifar100-lr${lr}-B128-BackRazor${backPruneRatio} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 20000 --eval_every 1000
