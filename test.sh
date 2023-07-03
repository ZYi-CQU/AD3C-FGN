CUDA_VISIBLE_DEVICES=1 python3 ./trans_anchor_map_cls.py \
    --cuda \
    --batch_size 256 \
    --nepoch 1000 \
    --data_root '/data/home_backup/yangwanli/data/' \
    --dataset 'CUB' \
    --image_embedding 'res101' \
    --class_embedding 'att' \
    --resSize 2048 \
    --attSize 312 \
    --preprocessing \
    --lr 0.0001 \
    --ap_lambda 10 \
    --manualSeed 342 \
    --epoch_da 100 \
    --epoch_cls 20 \
    --nz 312 \
    --log_path 'dasunseen.xls' \
    --gama 1 \
    --version 'ours' \
