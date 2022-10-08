label_smoothing=0.7
gpu=0
lr=1e-4
for trial in 1 2 3 4
do
python CNN/tinytl_fgvc_train.py --transfer_learning_method full \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr ${lr} --opt_type adam --manual_seed ${trial} \
    --label_smoothing ${label_smoothing} --distort_color None --frozen_param_bits 8 --origin_network  \
    --gpu ${gpu} --dataset flowers102 --path .exp/batch8/flowers102_full_lr${lr}_labelSmooth${label_smoothing}_origin_trial${trial}
done

