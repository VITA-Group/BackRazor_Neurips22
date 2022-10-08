lr=3e-4
gpu=4
backRazor_pruneRatio=0.9

python CNN/tinytl_fgvc_train.py --transfer_learning_method full_noBN \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr ${lr} --opt_type adam \
    --label_smoothing 0.3 --distort_color torch --frozen_param_bits 8 --origin_network \
    --gpu ${gpu} --dataset aircraft --path .exp/batch8/aircraft_lr${lr}_backRazor${backRazor_pruneRatio} \
    --backRazor --backRazor_pruneRatio ${backRazor_pruneRatio} --backRazor_pruneRatio_head ${backRazor_pruneRatio}
