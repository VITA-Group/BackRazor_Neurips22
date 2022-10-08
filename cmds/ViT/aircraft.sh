save_dir="."

devices="2,3"
port=6777
n_gpu=2

backPruneRatio=0.8

lr=0.03
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name aircraft-lr${lr}-B128E100-coTuneTrans-colorJitter-lrHead10x-BackRazor${backPruneRatio} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset aircraft --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} \
--train_batch_size 128 --eval_batch_size 128 \
--cotuning_trans --HeadLr10times \
--num_steps 6000 --eval_every 1000

