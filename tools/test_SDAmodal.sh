CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    --config experiments/COCOA/pcnet_m/config_SDAmodal.yaml \
    --load-model "ckpt_pth"\
    --order-method "aw" \
    --amodal-method "aw_sdm5" \
    --order-th 0.5 \
    --amodal-th 0.5 \
    --annotation "data/COCOA/annotations/COCO_amodal_test2014.json" \
    --image-root "data/COCOA/test2014" \
    --test-num -1 \
    --output "output_pth"
