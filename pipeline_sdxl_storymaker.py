# Copyright 2024 The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import math

import numpy as np
import PIL.Image
from PIL import Image
import torch, traceback, pdb
import torch.nn.functional as F

from diffusers.image_processor import PipelineImageInput

from diffusers.models import ControlNetModel

from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

from diffusers import StableDiffusionXLPipeline
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from insightface.utils import face_align

from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available
from ip_adapter.ip_adapter_faceid import faceid_plus

from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
from ip_adapter.attention_processor_faceid import LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor, LoRAAttnProcessor2_0 as LoRAAttnProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate insightface
        >>> import diffusers
        >>> from diffusers.utils import load_image
        >>> import cv2
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        
        >>> from insightface.app import FaceAnalysis
        >>> from pipeline_sdxl_storymaker import StableDiffusionXLStoryMakerPipeline

        >>> # download 'buffalo_l' under ./models
        >>> app = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        >>> app.prepare(ctx_id=0, det_size=(640, 640))
        
        >>> # download models under ./checkpoints
        >>> storymaker_adapter = f'./checkpoints/ip-adapter.bin'
        
        >>> pipe = StableDiffusionXLStoryMakerPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe.cuda()
        
        >>> # load adapter
        >>> pipe.load_storymaker_adapter(storymaker_adapter)

        >>> prompt = "a person is taking a selfie, the person is wearing a red hat, and a volcano is in the distance"
        >>> negative_prompt = "bad quality, NSFW, low quality, ugly, disfigured, deformed"

        >>> # load an image
        >>> image = load_image("your-example.jpg")
        >>> # load the mask image of portrait
        >>> mask_image = load_image("your-mask.jpg")
        
        >>> face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))[-1]

        >>> # generate image
        >>> image = pipe(
        ...     prompt, image=image, mask_image=mask_image,face_info=face_info, controlnet_conditioning_scale=0.8
        ... ).images[0]
        ```
"""

def bounding_rectangle(ori_img, mask):
    """
    Calculate the bounding rectangle of multiple rectangles.
    Args:
        rectangles (list of tuples): List of rectangles, where each rectangle is represented as (x, y, w, h)
    Returns:
        tuple: The bounding rectangle (x, y, w, h)
    """
    contours, _ = cv2.findContours(mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(contour) for contour in contours]
                
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    for x, y, w, h in rectangles:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    try:
        crop = ori_img[min_y:max_y, min_x:max_x]
        mask = mask[min_y:max_y, min_x:max_x]
    except:
        traceback.print_exc()
    return crop, mask


    
class StableDiffusionXLStoryMakerPipeline(StableDiffusionXLPipeline):
    
    def cuda(self, dtype=torch.float16, use_xformers=False):
        self.to('cuda', dtype)
        if hasattr(self, 'image_proj_model'):
            self.image_proj_model.to(self.unet.device).to(self.unet.dtype)
        
    def load_storymaker_adapter(self, image_encoder_path, model_ckpt, image_emb_dim=512, num_tokens=20, scale=0.8, lora_scale=0.8):     
        self.clip_image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
        self.set_image_proj_model(model_ckpt, image_emb_dim, num_tokens)
        self.set_ip_adapter(model_ckpt, num_tokens)
        self.set_ip_adapter_scale(scale, lora_scale)
        print(f'successful load adapter.')
        
    def set_image_proj_model(self, model_ckpt, image_emb_dim=512, num_tokens=16):
        
        image_proj_model = faceid_plus(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            clip_embeddings_dim=1280,
        )
        image_proj_model.eval()
        
        self.image_proj_model = image_proj_model.to(self.device, dtype=self.dtype)
        state_dict = torch.load(model_ckpt, map_location="cpu")
        if 'image_proj_model' in state_dict:
            state_dict = state_dict["image_proj_model"]
        self.image_proj_model.load_state_dict(state_dict)
        
    def set_ip_adapter(self, model_ckpt, num_tokens, lora_rank=128):
        
        unet = self.unet
        attn_procs = {}
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
                attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank).to(unet.device, dtype=unet.dtype)
            else:
                attn_procs[name] = LoRAIPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank).to(unet.device, dtype=unet.dtype)
        unet.set_attn_processor(attn_procs)
        
        state_dict = torch.load(model_ckpt, map_location="cpu")
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        if 'ip_adapter' in state_dict:
            state_dict = state_dict['ip_adapter']
        ip_layers.load_state_dict(state_dict)
    
    def set_ip_adapter_scale(self, scale, lora_scale=0.8):
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, LoRAIPAttnProcessor) or isinstance(attn_processor, LoRAAttnProcessor):
                attn_processor.scale = scale
                attn_processor.lora_scale = lora_scale

    def crop_image(self, ori_img, ori_mask, face_info):
        ori_img = np.array(ori_img)
        ori_mask = np.array(ori_mask)
        crop, mask = bounding_rectangle(ori_img, ori_mask)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)/255.
        crop = (255*np.ones_like(mask)*(1-mask)+mask*crop).astype(np.uint8)
        # cv2.imwrite('examples/results/0crop.jpg', crop[:,:,::-1])
        # cv2.imwrite('examples/results/0mask.jpg', (mask*255).astype(np.uint8))
        
        face_kps = face_info['kps']
        # face_image = face_align.norm_crop(crop, landmark=face_kps.numpy(), image_size=224)  #  224
        face_image = face_align.norm_crop(ori_img, landmark=face_kps, image_size=224)  #  224
        clip_face = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        
        ref_img = Image.fromarray(crop)
        ref_img = ref_img.resize((224, 224))  
        clip_img = self.clip_image_processor(images=ref_img, return_tensors="pt").pixel_values
        return clip_img, clip_face, torch.from_numpy(face_info.normed_embedding).unsqueeze(0)

    def _encode_prompt_image_emb(self, image, image_2, mask_image, mask_image_2, face_info, face_info_2, cloth, cloth_2, \
                                 device, num_images_per_prompt, dtype, do_classifier_free_guidance):
        crop_list = []; face_list = [];  id_list = []
        if image is not None:
            clip_img, clip_face, face_emb = self.crop_image(image, mask_image, face_info)
            crop_list.append(clip_img)
            face_list.append(clip_face)
            id_list.append(face_emb)
        if image_2 is not None:
            clip_img, clip_face, face_emb = self.crop_image(image_2, mask_image_2, face_info_2)
            crop_list.append(clip_img)
            face_list.append(clip_face)
            id_list.append(face_emb)
        if cloth is not None:
            crop_list = []
            clip_img = self.clip_image_processor(images=cloth.resize((224, 224)), return_tensors="pt").pixel_values
            crop_list.append(clip_img)
        if cloth_2 is not None:
            clip_img = self.clip_image_processor(images=cloth_2.resize((224, 224)), return_tensors="pt").pixel_values
            crop_list.append(clip_img)
        assert len(crop_list)>0, f"input error, images is None"
        clip_image = torch.cat(crop_list, dim=0).to(device, dtype=dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2] 
        clip_face = torch.cat(face_list, dim=0).to(device, dtype=dtype)
        clip_face_embeds = self.image_encoder(clip_face, output_hidden_states=True).hidden_states[-2] 
        id_embeds = torch.cat(id_list, dim=0).to(device, dtype=dtype)
        # print(f'clip_image_embeds: {clip_image_embeds.shape}, clip_face_embeds:{clip_face_embeds.shape}, id_embeds:{id_embeds.shape}')
        if do_classifier_free_guidance:
            prompt_image_emb = self.image_proj_model(id_embeds, clip_image_embeds, clip_face_embeds)
            B, C, D = prompt_image_emb.shape
            prompt_image_emb = prompt_image_emb.view(1, B*C, D)
            neg_emb = self.image_proj_model(torch.zeros_like(id_embeds), torch.zeros_like(clip_image_embeds), torch.zeros_like(clip_face_embeds))
            neg_emb = neg_emb.view(1, B*C, D)
            prompt_image_emb = torch.cat([neg_emb, prompt_image_emb], dim=0)
        else:
            prompt_image_emb = torch.cat([prompt_image_emb], dim=0)
            B, C, D = prompt_image_emb.shape
            prompt_image_emb = prompt_image_emb.view(1, B*C, D)
        
        # print(f'prompt_image_emb: {prompt_image_emb.shape}')
        bs_embed, seq_len, _ = prompt_image_emb.shape
        prompt_image_emb = prompt_image_emb.repeat(1, num_images_per_prompt, 1)
        prompt_image_emb = prompt_image_emb.view(bs_embed * num_images_per_prompt, seq_len, -1)
        
        return prompt_image_emb.to(device=device, dtype=dtype)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: Union[torch.Tensor, PIL.Image.Image] = None,
        image_2: PipelineImageInput = None,
        mask_image_2: Union[torch.Tensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],

        # IP adapter
        ip_adapter_scale=None,
        lora_scale=None,
        face_info = None,
        face_info_2 = None,
        cloth = None,
        cloth_2 = None,

        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
                and `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, pooled text embeddings are generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                weighting). If not provided, pooled `negative_prompt_embeds` are generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned containing the output images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. set ip_adapter_scale
        if ip_adapter_scale is not None and lora_scale is not None:
            self.set_ip_adapter_scale(ip_adapter_scale, lora_scale)

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt=prompt,
        #     prompt_2=prompt_2,
        #     height=height, width=width,
        #     callback_steps=callback_steps,
        #     negative_prompt=negative_prompt,
        #     negative_prompt_2=negative_prompt_2,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     pooled_prompt_embeds=pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        # )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.unet.device
        # pdb.set_trace()
        # 3.1 Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        
        # 3.2 Encode image prompt
        prompt_image_emb = self._encode_prompt_image_emb(image, image_2, mask_image, mask_image_2, face_info, face_info_2, cloth,cloth_2,
                                                         device, num_images_per_prompt,
                                                         self.unet.dtype, self.do_classifier_free_guidance)
        
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6.5 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.2 Prepare added time ids & embeddings
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        encoder_hidden_states = torch.cat([prompt_embeds, prompt_image_emb], dim=1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)