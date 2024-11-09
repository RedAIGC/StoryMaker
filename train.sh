export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

model_path=Unstable_Diffusers_V11_-_Diffusers  #  or sdxl-1.0
vit_path=CLIP-ViT-H-14-laion2B-s32B-b79K

pretrained_ip_adapter=ip-adapter-faceid-plusv2_sdxl.bin
pretrained_ip_plus=ip-adapter-plus_sdxl_vit-h.bin

train(){
    model_name=${1}
    train_dir=${2}
    batch_size=${3}
    nvec=${4}
    steps=${5}
    lr=${6}
    noise_offset=${7}
    output_dir=${8}

    accelerate launch --num_processes=8  --main_process_port=23225 --mixed_precision="bf16" \
            train_sm.py  --pretrained_model_name_or_path=${model_path}   \
            --image_encoder_path=${vit_path}      --ip_loss=0.1   --mask_loss_weight=5  --bg_tokens=20  \
            --num_tokens=${nvec}    --faceid_loss=0  \
            --noise_offset=0.05  --drop_prompt=0.2  \
            --resolution=960  \
            --train_batch_size=1   --max_train_steps=${steps}    \
            --gradient_accumulation_steps=8 \
            --dataloader_num_workers=16   \
            --learning_rate=${lr} \
            --weight_decay=0.01 \
            --output_dir=${output_dir} \
            --save_steps=1000 \
            --pretrained_ip_plus=${pretrained_ip_plus} \
            --pretrained_ip_adapter=${pretrained_ip_adapter} \
            
}

LR=1e-4
nvec=20
BATCH_SIZE=1
NOISE_OFFSET=0.05
STEPS=100000
MODEL_NAME=sdxl10
OUTPUT_DIR=output

base_model_name=${MODEL_NAME}
output_dir=${OUTPUT_DIR}/unstable11_bs64_lr${LR}_drop02_r128_snr0_fl5_ip01new6_al70
train $base_model_name $train_dir $BATCH_SIZE $nvec $STEPS $LR $NOISE_OFFSET $output_dir 
        
