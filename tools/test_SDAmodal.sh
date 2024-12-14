CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config "experiments/COCOA/pcnet_m/config_SDAmodal.yaml" \
    --load-model "D:/HocTap/KLTN/models/ckpt_SDAmodal"\
    --order-method "aw" \
    --amodal-method "aw_sdm5" \
    --order-th 0.5 \
    --amodal-th 0.5 \
    --annotation "D:/HocTap/KLTN/dataset/mp3d/mp3d_eval.json" \
    --image-root "D:/HocTap/KLTN/dataset/mp3d/mp3d_eval_10.29" \
    --test-num -1 \
    --output "test"
