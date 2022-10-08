lr=1e-5
gpu=0
backRazor_pruneRatio=0.9

lr=3e-6
gpu=1
backRazor_pruneRatio=0.9

lr=1e-6
gpu=2
backRazor_pruneRatio=0.9


python CNN/tinytl_fgvc_train.py --transfer_learning_method full_noBN \
    --train_batch_size 8 --test_batch_size 100 --net resnet18 \
    --n_epochs 50 --init_lr ${lr} --opt_type adam \
    --label_smoothing 0.1 --distort_color None --frozen_param_bits 8 --origin_network --fix_bn_stat \
    --gpu ${gpu} --dataset cifar100 --path .exp/batch8/cifar100_res18_lr${lr}_backRazor${backRazor_pruneRatio} \
    --backRazor --backRazor_pruneRatio ${backRazor_pruneRatio} --backRazor_pruneRatio_head ${backRazor_pruneRatio} \
    --no_decay_keys None


lr=3e-5
gpu=3

lr=1e-5
gpu=4

lr=3e-6
gpu=5

lr=1e-6
gpu=6


python CNN/tinytl_fgvc_train.py --transfer_learning_method full_noBN \
    --train_batch_size 8 --test_batch_size 100 --net resnet18 \
    --n_epochs 50 --init_lr ${lr} --opt_type adam \
    --label_smoothing 0.1 --distort_color None --frozen_param_bits 8 --origin_network --fix_bn_stat \
    --gpu ${gpu} --dataset cifar100 --path .exp/batch8/cifar100_res18_lr${lr} \
    --no_decay_keys None
