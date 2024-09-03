import cv2, os
import torch
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()
import pillow_heif
pillow_heif.register_avif_opener()  
from diffusers.utils import load_image
from diffusers import EulerAncestralDiscreteScheduler, UniPCMultistepScheduler

from insightface.app import FaceAnalysis
from pipeline_sdxl_storymaker import StableDiffusionXLStoryMakerPipeline

def resize_img(input_image, max_side=1280, min_side=960, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


# Load face encoder
app = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to models
face_adapter = f'checkpoints/mask.bin'
image_encoder_path = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'   #  from https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
base_model_path = 'huaquan/YamerMIX_v11'  # from https://huggingface.co/huaquan/YamerMIX_v11

pipe = StableDiffusionXLStoryMakerPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
)
pipe.cuda()
pipe.load_storymaker_adapter(image_encoder_path, face_adapter, scale=0.8, lora_scale=0.8)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

def demo():
    prompt = "a person is taking a selfie, the person is wearing a red hat, and a volcano is in the distance"
    n_prompt = "bad quality, NSFW, low quality, ugly, disfigured, deformed"

    image = Image.open("examples/ldh.png").convert('RGB')
    mask_image = Image.open("examples/ldh_mask.png").convert('RGB')
    
    # image = resize_img(image)
    face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face

    generator = torch.Generator(device='cuda').manual_seed(666)
    for i in range(4):
        output = pipe(
            image=image, mask_image=mask_image, face_info=face_info,
            prompt=prompt,
            negative_prompt=n_prompt,
            ip_adapter_scale=0.8, lora_scale=0.8,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=1280, width=960,
            generator=generator,
        ).images[0]
        output.save(f'examples/results/ldh666_{i}.jpg')

def demo_two():
    prompt = "A man and a woman are taking a selfie, and a volcano is in the distance"
    n_prompt = "bad quality, NSFW, low quality, ugly, disfigured, deformed"

    image = Image.open("examples/ldh.png").convert('RGB')
    mask_image = Image.open("examples/ldh_mask.png").convert('RGB')
    image_2 = Image.open("examples/tsy.png").convert('RGB')
    mask_image_2 = Image.open("examples/tsy_mask.png").convert('RGB')
    
    face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_info_2 = app.get(cv2.cvtColor(np.array(image_2), cv2.COLOR_RGB2BGR))
    face_info_2 = sorted(face_info_2, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    
    generator = torch.Generator(device='cuda').manual_seed(666)
    for i in range(4):
        output = pipe(
            image=image, mask_image=mask_image,face_info=face_info,  #  first person
            image_2=image_2, mask_image_2=mask_image_2,face_info_2=face_info_2,  # second person
            prompt=prompt,
            negative_prompt=n_prompt,
            ip_adapter_scale=0.8, lora_scale=0.8,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=1280, width=960,
            generator=generator,
        ).images[0]
        output.save(f'examples/results/ldh_tsy666_{i}.jpg')

def demo_swapcloth():
    prompt = "a person is taking a selfie, and a volcano is in the distance"
    n_prompt = "bad quality, NSFW, low quality, ugly, disfigured, deformed"

    image = Image.open("examples/ldh.png").convert('RGB')
    mask_image = Image.open("examples/ldh_mask.png").convert('RGB')
    cloth = Image.open("examples/cloth2.png").convert('RGB')
    
    face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face

    generator = torch.Generator(device='cuda').manual_seed(666)
    for i in range(4):
        output = pipe(
            image=image, mask_image=mask_image, face_info=face_info, cloth=cloth,
            prompt=prompt,
            negative_prompt=n_prompt,
            ip_adapter_scale=0.8, lora_scale=0.8,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=1280, width=960,
            generator=generator,
        ).images[0]
        output.save(f'examples/results/ldh_cloth_{i}.jpg')


if __name__ == "__main__":
    # single portrait generation
    demo()

    # two portrait generation
    # demo_two()
