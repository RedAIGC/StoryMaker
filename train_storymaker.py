import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import logging
import numpy as np 
import torch, pdb, math
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from transformers.models.clip.modeling_clip import CLIPPreTrainedModel, CLIPModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoTokenizer, PretrainedConfig

from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
from ip_adapter.ip_adapter_faceid import faceid_plus
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
    from ip_adapter.attention_processor_faceid import (
            LoRAAttnProcessor2_0 as LoRAAttnProcessor,
        )
    from ip_adapter.attention_processor_faceid import (
            LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor,
        )

import warnings, traceback
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint*")
Image.MAX_IMAGE_PIXELS = None

logger = get_logger(__name__)

import random, cv2
import string
from tqdm import tqdm

def collate_fn(data):
    images_gt = torch.stack([example["image_gt"] for example in data])
    images_ref = torch.stack([example["image_ref"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids2 = torch.cat([example["text_input_ids2"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    clip_faces = torch.cat([example["clip_face"] for example in data], dim=0)
    face_id_embeds = torch.cat([example["face_id_embed"] for example in data], dim=0)
    face_kps_abs = torch.cat([example["face_kps_abs"] for example in data], dim=0)
    face_unnorm_embeds = torch.cat([example["face_unnorm_embed"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    topleft = torch.stack([example["topleft"] for example in data])
    masks_gt = torch.stack([example["mask_gt"] for example in data])
    style_idx = torch.stack([example["style"] for example in data])
    return {
        "images_gt": images_gt,
        "images_ref": images_ref,
        "text_input_ids": text_input_ids,
        "text_input_ids2": text_input_ids2,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "topleft":topleft,
        "face_id_embeds": face_id_embeds, "face_kps_abs": face_kps_abs,
        "face_unnorm_embeds": face_unnorm_embeds,
        "clip_faces": clip_faces,
        "masks_gt": masks_gt, "style_idx": style_idx,
    }
    
from mp_dataset import MasktileDataset
class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        
        
    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, faceid_embeds, use_faceid, \
                    face_embeds, use_facekps, controlnet_image, controlnet, ctrl_clipemb, added_cond_kwargs):
        ip_tokens = self.image_proj_model(faceid_embeds, image_embeds, face_embeds=face_embeds, is_training=1)
            
        B, C, D = ip_tokens.shape 
        ip_tokens = ip_tokens.view(1, B*C, D)  #  多人ip-embeds need reshape, batchsize must be 1
        if controlnet:
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=ip_tokens,
                added_cond_kwargs=added_cond_kwargs,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )
        else:
            down_block_res_samples, mid_block_res_sample = None, None
            
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        weight_dtype =  torch.float16
        noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=None if down_block_res_samples is None else [ sample.to(dtype=weight_dtype) for sample in down_block_res_samples ],
                    mid_block_additional_residual=None if down_block_res_samples is None else mid_block_res_sample.to(dtype=weight_dtype),
                ).sample
        return noise_pred
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--use_whichemb",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=960,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument(
        "--lr_lora",
        type=float,
        default=None,
        help="Learning rate to use.",
    )
    parser.add_argument(
        "--old_vfeature",
        action="store_true",
        default=False,
        help="whether to",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--noise_offset", type=float, default=0.05, help="The scale of noise offset.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument("--bg_tokens", type=int, default=20)
    parser.add_argument(
        "--ctrl_clipemb",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cropref",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--hstack_ref",
        type=int,
        default=0,
    )
    parser.add_argument( "--bg_ref", type=int, default=0, )
    parser.add_argument( "--drop_pose", type=int, default=0, )
    parser.add_argument( "--style_emb", type=int, default=0, )
    parser.add_argument(
        "--use_vseg",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_faceid",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--use_facekps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_headseg",
        type=int,
        default=0,
    )
    parser.add_argument( "--faceid_loss", type=float, default=0.1, )
    parser.add_argument( "--mse_loss", type=float, default=0, )
    parser.add_argument( "--use_unnorm", type=int, default=0, )
    parser.add_argument( "--add_anime", type=float, default=0, )
    parser.add_argument( "--sort_person", type=float, default=0, )
    parser.add_argument(
        "--ip_attn_len",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--invproj",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--split_ip",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pretrained_ip_plus", type=str, default='',
    )
    parser.add_argument(
        "--drop_prompt", type=float, default=0.2,
    )
    parser.add_argument(
        "--ip_loss", type=float, default=0.1,
    )
    parser.add_argument(
        "--ip_loss_only_person", type=int, default=0,
    )
    parser.add_argument(
        "--mask_loss_weight", type=float, default=5,
    )
    parser.add_argument(
        "--mmdiff_clip_path", type=str, default=None,
    )
    parser.add_argument(
        "--instantid_path", type=str, default=None,
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

import random
import string
def generate_random_string(length):
    # 生成随机的数字和字母
    letters = string.ascii_letters + string.digits
    # 生成指定长度的随机字符串
    return ''.join(random.choice(letters) for i in range(length))


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )
    tokenizer2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, use_fast=False
    )
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    
    controlnet = None
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        controlnet.requires_grad_(False)
 
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    image_proj_model = faceid_plus(
        cross_attention_dim=unet.config.cross_attention_dim,
        id_embeddings_dim=512,
        clip_embeddings_dim=image_encoder.config.hidden_size,
    )
    
    # init adapter modules
    attn_procs = {}; lora_rank = args.lora_rank
    unet_sd = unet.state_dict(); #print(unet.attn_processors.keys())
    ip_attn_names = []
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank, lora_scale=args.scale)
        else:
            layer_name = name.split(".processor")[0]
            attn_procs[name] = LoRAIPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, \
                                    rank=lora_rank, num_tokens=20,  ip_loss=args.ip_loss, lora_scale=args.scale,scale=args.scale)
            ip_attn_names.append(name)
    ip_attn_names_len = len(ip_attn_names)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    if args.pretrained_ip_adapter:
        state_dict = torch.load(args.pretrained_ip_adapter, map_location="cpu")
        if 'image_proj' in state_dict:
            aa = state_dict["image_proj"]
        elif 'image_proj_model' in state_dict:
            aa = state_dict["image_proj_model"]
        try:
            image_proj_model.load_state_dict(aa)
            logger.info(f"Loading pretrain proj weights successful, modelpath={args.pretrained_ip_adapter}")
        except:
            dict_b = torch.load(args.pretrained_ip_plus)
            image_proj_model.load_model(aa, dict_b['image_proj'])
            logger.info("Loading resample weights")
        
        if 'ip_adapter' in state_dict:
            state_dict = state_dict['ip_adapter']; print('use faceid-adapter attention weights:', len(state_dict))
        adapter_modules.load_state_dict(state_dict, strict=False)
        logger.info("Loading existing ip adapter weights")

    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules)
    

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    if args.controlnet_model_name_or_path:
        controlnet.to(accelerator.device, dtype=weight_dtype)
    # optimizer
    
    lr_lora =  args.learning_rate if args.lr_lora is None else args.lr_lora 
    params_to_opt = itertools.chain([ { "params": itertools.chain(ip_adapter.image_proj_model.parameters()),  "lr": args.learning_rate },
                        { "params": itertools.chain(ip_adapter.adapter_modules.parameters()), "lr": lr_lora, }, ])

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MasktileDataset(args, tokenizer=tokenizer, tokenizer2=tokenizer2, t_drop_rate=args.drop_prompt, i_drop_rate=0.05, ti_drop_rate=0.05,)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    text_encoders = [text_encoder, text_encoder_two]
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)/ args.gradient_accumulation_steps)
    
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0; st=time.time()
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    num_token = 20
    bg_tokens=num_token 

    def tensor_2_numpy(tensor, h=None,w=None):
        if h is not None:
            tensor = F.interpolate(tensor, size=(h, w), mode='bilinear',)
            data = (tensor*0.5+0.5).clamp(0,1)
        else:
            data = tensor
        data = data.squeeze(0).permute(1,2,0).float().detach().cpu()*255
        if data.shape[2]==1:
            data = np.tile(data, (1,1,3))
        img = np.array(data, np.uint8)
        return img

    def get_loss_ip(mask_gt, step, img_gt):
        B, lsh,lsw, img_num = mask_gt.shape
        if args.mask_loss_weight>0:
            dr = 2
        tmp = mask_gt.permute(0,3,1,2)
        tmp = F.interpolate(tmp, size=(lsh//dr, lsw//dr), mode='bilinear',)
        mask_gt = tmp.permute(0,2,3,1)
        B, lsh,lsw, img_num = mask_gt.shape
        th, tw = math.ceil(lsh/2), math.ceil(lsw/2)
        ls = lsh*lsw; # ip_attn_len = 0
        attn_list = [0]*img_num
        mask_person = mask_gt.sum(dim=-1).clamp(0,1);  
        mask_area = mask_person.sum();  start_idx = 0
       
        mask_bg = 1-mask_person; # bg_area = max(1e-3, lsh*lsw-mask_area)
        start_idx=0;   attn_bg = 0
        mask_area= lsh*lsw
        for name in ip_attn_names:
            attn_probs = unet.attn_processors[name].attn_probs  #  batch, latentsize, ipembed-len
            B, tlen, attn_ls = attn_probs.shape
            if attn_ls != ls:
                attn_probs = attn_probs.view(B, tlen, th, tw)
                attn_probs = F.interpolate(attn_probs, size=(lsh, lsw), mode='bilinear',)
            else:
                continue #  apply the localization loss to the downsampled cross-attention maps, i.e., the middle 5 blocks of the U-Net, which are known to contain more semantic information
                
            bg_prob = attn_probs[:,:bg_tokens, :,:].float().sum(dim=1)
            attn_bg+=bg_prob
            for i in range(start_idx, img_num):
                cur_prob = attn_probs[:,num_token*i+bg_tokens:num_token*(i+1)+bg_tokens, :,:].float().sum(dim=1)
                attn_list[i-start_idx] += cur_prob
        loss_ip = 0; res = []
        
        for i, attn in enumerate(attn_list):
            attn_mask = attn/60
            cur_loss = F.mse_loss(attn_mask.float(), mask_gt[:,:,:,i].float(), reduction="none")
            loss_ip += cur_loss.sum()/max(mask_area, 1e-5)
            if  i==0:  # 顺便计算bg loss
                attn_bg = attn_bg/60
                cur_loss = F.mse_loss(attn_bg.float(), mask_bg.float(), reduction="mean")
                loss_ip += cur_loss
                
            if step%1000==0:
                print(step, i, attn_mask.min().item(), attn_mask.max().item(), [B, lsh,lsw, img_num], args.ip_attn_len, loss_ip)
                if True:
                    if i==0:
                        img = tensor_2_numpy(img_gt, lsh, lsw)
                        img = img[:,:,::-1]
                        mask_img = tensor_2_numpy(mask_person.unsqueeze(1))
                        res.append(np.vstack([img, mask_img]))
                        mask = tensor_2_numpy(mask_bg.unsqueeze(1))
                        attn = tensor_2_numpy(attn_bg.unsqueeze(1))
                        res.append(np.vstack([mask, attn]))
                        # pdb.set_trace()
                    mask = tensor_2_numpy(mask_gt[:,:,:,i].unsqueeze(1))
                    attn = tensor_2_numpy(attn_mask.unsqueeze(1))
                    res.append(np.vstack([mask, attn]))
        if len(res)>1:
            sname = generate_random_string(4); os.makedirs(os.path.join(args.output_dir, 'attn_mask'), exist_ok=True)
            cv2.imwrite(os.path.join(args.output_dir, 'attn_mask', f'{step:05d}_{sname}_{attn_mask.min().item():.4f}.jpg'), np.hstack(res))
        return torch.nan_to_num(loss_ip/img_num, nan=1e-5)  #
            

    if args.faceid_loss>0:
        from arcface import face_align_torch
        from arcface import get_model
        facenet = get_model('r34', fp16=False)
        arcface_path = './arcface/resnet34.pth'
        print(arcface_path)
        facenet.load_state_dict(torch.load(arcface_path))
        facenet.to(accelerator.device, dtype=torch.float32)
        facenet.requires_grad_(False)
        facenet.eval()

    def get_each_face_and_faceid_loss(gt_face_in, noise_face_in):
        gt_face = gt_face_in.div(255).sub(0.5).div(0.5)
        noise_face = noise_face_in.div(255).sub(0.5).div(0.5)
        face_emb=facenet(torch.cat([gt_face, noise_face], dim=0))
        cosine = F.cosine_similarity(face_emb[0:1].detach(), face_emb[1:2])
        loss = 1-cosine.mean()
        return loss
        
    def get_loss_faceid(step, img_gt, noisy_latents, noise_pred, timesteps, face_kps_abs, mask_gt):  #  https://arxiv.org/pdf/2312.06354
        loss_id = 0; loss_mse = 0
        if timesteps[0]>250:
            return 0,0
        x0=noise_scheduler.step(noise_pred, timesteps, noisy_latents).pred_original_sample
        noise_img = vae.decode(x0/vae.config.scaling_factor).sample
        # nimg = image_processor.postprocess(noise_img.detach().cpu())[0]  #  not support grad
        data = (noise_img*0.5+0.5).clamp(0,1)*255
        # data = data*255  #  .squeeze(0).permute(1,2,0)
        img_num, _,_ = face_kps_abs.shape
        img_gt = (img_gt*0.5+0.5).clamp(0,1)*255  # RGB
        for i in range(img_num):
            grid = F.affine_grid(face_kps_abs[i:i+1], size=[1, 3, 112, 112])
            # pdb.set_trace()
            gt_face = F.grid_sample(img_gt, grid=grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # [1, C, H, W]
            noise_face = F.grid_sample(data, grid=grid, mode="bilinear", padding_mode="zeros", align_corners=False) 
            # face_image = face_align.norm_crop(data, landmark=face_kps.numpy(), image_size=224)  #  224
            loss_id+=get_each_face_and_faceid_loss(gt_face, noise_face, )

        if step%100==0:
            t = timesteps.cpu().numpy()[0]
            nimg = np.array(data.detach().cpu().squeeze(0).permute(1,2,0), np.uint8)[:,:,::-1] # to BGR
            gimg = np.array(img_gt.detach().cpu().squeeze(0).permute(1,2,0), np.uint8)[:,:,::-1] # to BGR
            nface = np.array(noise_face.detach().cpu().squeeze(0).permute(1,2,0), np.uint8)[:,:,::-1] # to BGR
            gface = np.array(gt_face.detach().cpu().squeeze(0).permute(1,2,0), np.uint8)[:,:,::-1] # to BGR
            sname = generate_random_string(4); os.makedirs(os.path.join(args.output_dir, 'reverse'), exist_ok=True)
            cv2.imwrite(os.path.join(args.output_dir, 'reverse', f'{step:05d}_{sname}_{t}_{loss_id.item():.3f}.jpg'), np.hstack([gimg, nimg]))
            cv2.imwrite(os.path.join(args.output_dir, 'reverse', f'{step:05d}_{sname}_{t}_face.jpg'), np.hstack([gface, nface]))
        
        return loss_id, loss_mse

    noise_scheduler.alphas_cumprod=noise_scheduler.alphas_cumprod.to(accelerator.device)
    
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            try:
                load_data_time = time.perf_counter() - begin
                
                with accelerator.accumulate(ip_adapter):
                    with torch.no_grad():
                        latents = vae.encode(batch["images_gt"].to(accelerator.device)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()
                    add_time_ids = batch['topleft']
                    prompt_embeds_list = []
                    for id_name, text_encoder in zip(['text_input_ids', 'text_input_ids2'], text_encoders):
                        prompt_embeds = text_encoder(
                            batch[id_name].to(text_encoder.device),
                            output_hidden_states=True,
                        )
                        pooled_prompt_embeds = prompt_embeds[0]
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                        bs_embed, seq_len, _ = prompt_embeds.shape
                        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                        prompt_embeds_list.append(prompt_embeds)
                    encoder_hidden_states = torch.concat(prompt_embeds_list, dim=-1)
                    add_text_embeds = pooled_prompt_embeds.view(bs_embed, -1)
                    add_text_embeds = add_text_embeds.to(accelerator.device)
                    add_time_ids = add_time_ids.to(dtype=add_text_embeds.dtype)

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    if batch["drop_image_embeds"][0]:
                        clip_images = torch.zeros_like(batch["clip_images"])
                        face_id_embeds = torch.zeros_like(batch["face_id_embeds"])
                        clip_faces = torch.zeros_like(batch["clip_faces"])
                    else:
                        clip_images = batch["clip_images"]
                        face_id_embeds = batch["face_id_embeds"]
                        clip_faces = batch["clip_faces"]
                    with torch.no_grad():    #  B,257, 1024
                        image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                        
                        face_embeds = image_encoder(clip_faces.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                        
                    midinfo={"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    # ControlNet conditioning.
                    controlnet_image = batch["images_ref"].to(dtype=weight_dtype)
                    
                    noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds, face_id_embeds, args.use_faceid, face_embeds, args.use_facekps, \
                                controlnet_image, controlnet, args.ctrl_clipemb, \
                                added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids, })
                    
                    if args.snr_gamma is None:
                        if args.mask_loss_weight>0 and batch["drop_image_embeds"][0]<1:
                            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                            mask_gt = batch["masks_gt"].to(accelerator.device, dtype=weight_dtype)
                            
                            mask_gt = mask_gt.sum(dim=-1).unsqueeze(1).clamp(0,1)
                            loss = loss + loss*mask_gt*args.mask_loss_weight
                            loss = loss.mean()
                        else:
                            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                        
                    else:
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                        if args.mask_loss_weight>0:
                            mask_gt = batch["masks_gt"].to(accelerator.device, dtype=torch.float32)
                            mask_gt = mask_gt.sum(dim=-1).unsqueeze(1).clamp(0,1)
                            loss = loss + loss*mask_gt*args.mask_loss_weight
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    if args.faceid_loss>0 and batch["drop_image_embeds"][0]<1:
                        loss_faceid, loss_mse = get_loss_faceid(step,  batch["images_gt"], noisy_latents, noise_pred.float(), timesteps, batch["face_kps_abs"].float(), mask_gt)
                        loss += loss_faceid*args.faceid_loss
                        
                    mask_gt = batch["masks_gt"].to(accelerator.device, dtype=torch.float32)
                    loss_ip = get_loss_ip(mask_gt, step, batch["images_gt"])
                    loss += loss_ip*args.ip_loss

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:  #  error
                        params_to_clip = params_to_opt
                        accelerator.clip_grad_norm_(params_to_clip, 1)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=False)
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                if accelerator.is_main_process:
                    if global_step>1 and global_step % args.save_steps == 0:
                        print(f'save checkpoint, global step={global_step}, save steps={args.save_steps}')
                        weight_name = (f"checkpoint-{global_step}" )
                        save_path = os.path.join(args.output_dir, weight_name)
                        save_progress(
                            ip_adapter,
                            accelerator,
                            args,
                            save_path,
                        )
                logs = {"loss": loss.detach().item(),"ip_loss": loss_ip.detach().item(), "lr": args.learning_rate, "data_time":load_data_time, "train_time":time.perf_counter() - begin}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                begin = time.perf_counter()

                if global_step >= args.max_train_steps:
                    break
            except Exception as e:
                traceback.print_exc()
                
    accelerator.wait_for_everyone()
    accelerator.end_training()
                
def save_progress(trained_embdding_net, accelerator, args, save_path, safe_serialization=True):
    attention = trained_embdding_net.module.adapter_modules.state_dict()
    image_proj_model = trained_embdding_net.module.image_proj_model.state_dict()
    logger.info(f"Saving embeddings to {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    learned_embeds_dict = {'image_proj_model':image_proj_model, 'ip_adapter':attention, }
    torch.save(learned_embeds_dict, save_path+'/mask.bin')
    
    
if __name__ == "__main__":
    main()    
