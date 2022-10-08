gpu=0
lr=3e-4
pruneRatio=0.9
backRazor_pruneRatio=${pruneRatio}
backRazor_pruneRatio_head=${pruneRatio}

python CNN/tinytl_fgvc_train.py --transfer_learning_method full \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr ${lr} --opt_type adam \
    --label_smoothing 0.7 --distort_color torch --frozen_param_bits 8 --origin_network  \
    --gpu ${gpu} --dataset car --path .exp/batch8/car_baseline_lr${lr}_origin_backRazor${backRazor_pruneRatio}HeadR${backRazor_pruneRatio_head} \
    --backRazor --backRazor_pruneRatio ${backRazor_pruneRatio} --backRazor_pruneRatio_head ${backRazor_pruneRatio_head}
