CUDA_VISIBLE_DEVICES=2 python3 ./test.py \
    --cuda \
    --batch_size 256 \
    --nepoch 1000 \
    --data_root '/data/' \
    --dataset 'CUB' \
    --image_embedding 'res101' \
    --class_embedding 'att' \
    --resSize 2048 \
    --attSize 312 \
    --preprocessing \
    --manualSeed 342 \
    --epoch_da 100 \
    --epoch_cls 20 \
    --log_path 'dasunseen.xls' \
    --version 'AD3C' \
    --G_model_path './models/AD3C_G_CUB.pkl' \
    --C_model_path './models/AD3C_cls_CUB.pkl'
