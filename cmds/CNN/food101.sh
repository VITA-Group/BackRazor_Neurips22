gpu=0
pruneRatio=0.9

label_smoothing=0.1
backRazor_pruneRatio=${pruneRatio}
backRazor_pruneRatio_head=${pruneRatio}
lr=3e-5

python CNN/tinytl_fgvc_train.py --transfer_learning_method full_noBN \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr ${lr} --opt_type adam \
    --label_smoothing ${label_smoothing} --distort_color None --frozen_param_bits 8 --origin_network \
    --gpu ${gpu} --dataset food101 --path .exp/batch8/food101_full_lr${lr}_origin_fixBn_backRazor${backRazor_pruneRatio}HeadR${backRazor_pruneRatio_head} \
    --backRazor --backRazor_pruneRatio ${backRazor_pruneRatio} --backRazor_pruneRatio_head ${backRazor_pruneRatio_head}

