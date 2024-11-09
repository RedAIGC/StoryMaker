import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pillow_heif import register_heif_opener
register_heif_opener()
import pillow_heif
pillow_heif.register_avif_opener()  #  support .avif image at 08.10

import os, glob, random, pdb, cv2, math, json, time, traceback
import numpy as np 
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer

from transformers import CLIPImageProcessor
from insightface.utils import face_align
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

def resize_img_scale(image, img_scale=[960, 1280]):
    ori_w, ori_h = image.size
    max_long_edge = max(img_scale)
    max_short_edge = min(img_scale)
    scale_factor = min(max_long_edge / max(ori_h, ori_w), max_short_edge / min(ori_h, ori_w))
    img_w = round(ori_w * float(scale_factor))
    img_h = round(ori_h * float(scale_factor))
    img_w, img_h = map(lambda x: x - x % 64, (img_w, img_h))
    image = image.resize((img_w, img_h))
    return image

# https://blog.csdn.net/qq_37541097/article/details/134766540
def cal_torch_theta(opencv_theta: np.ndarray, src_h: int, src_w: int, dst_h: int, dst_w: int):
    m = np.concatenate([opencv_theta, np.array([[0., 0., 1.]], dtype=np.float32)])
    m_inv = np.linalg.inv(m)

    a = np.array([[2 / (src_w - 1), 0., -1.],
                  [0., 2 / (src_h - 1), -1.],
                  [0., 0., 1.]], dtype=np.float32)

    b = np.array([[2 / (dst_w - 1), 0., -1.],
                  [0., 2 / (dst_h - 1), -1.],
                  [0., 0., 1.]], dtype=np.float32)
    b_inv = np.linalg.inv(b)

    pytorch_m = a @ m_inv @ b_inv
    return pytorch_m[:2]  #  3x2

class DynamicResize(object):
    def __init__(self, scale_size=960):
        self.img_scale = scale_size

    def __call__(self, image):
        ori_w, ori_h = image.size
        max_long_edge = int(self.img_scale*1.5)
        max_short_edge = self.img_scale
        scale_factor = min(max_long_edge / max(ori_h, ori_w), max_short_edge / min(ori_h, ori_w))
        img_w = round(ori_w * float(scale_factor))
        img_h = round(ori_h * float(scale_factor))
        img_w, img_h = map(lambda x: x - x % 64, (img_w, img_h))
        image = image.resize((img_w, img_h))
        return image
    
def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):  #  0-4分别是左上，右上，中间，左下，右下
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)
        # import pdb; pdb.set_trace()
    out_img_pil = Image.fromarray(out_img.astype(np.uint8)) #  cv2 to pil, bgr to rgb
    return out_img_pil

class MasktileDataset(Dataset):
    def __init__(self, args=None, tokenizer=None, tokenizer2=None, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, debug=False, istest=False):#TODO:mode
        self.args = args
        self.size = 1024
        self.wh = [960, 1280]
        self.hstack_ref = 0
        self.use_vseg = 1
        self.use_faceid = 1
        self.use_facekps = 1
        self.use_headseg = 0
        self.use_unnorm = 0
        self.load_caption_once = 1
        self.debug = debug
        self.istest = istest
        self.mask_ratio = 16
        self.faceid_loss = 0
        self.mse_loss = 0
        self.drop_pose = 1
        self.add_anime = 0
        self.sort_person = 0
        if args:
            self.wh = [args.resolution, int(args.resolution*1.34)]
            self.hstack_ref = args.hstack_ref  
            self.use_vseg = args.use_vseg  # default=1
            self.use_faceid = args.use_faceid
            self.use_facekps = args.use_facekps
            self.use_headseg = args.use_headseg
            self.use_unnorm = args.use_unnorm
            self.drop_pose = args.drop_pose
            self.add_anime = args.add_anime
            self.sort_person = args.sort_person
            if args.mask_loss_weight>0:
                self.mask_ratio = 8
            self.faceid_loss=args.faceid_loss
            
        self.centercrop = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.conditioning_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.clip_image_processor = CLIPImageProcessor()
        
        self.key_list = {1:[], 2:[], 3:[]}
        self.total_caption = {}
        if self.load_caption_once and not self.istest:
            imgpath = 'examples/datasets'
            self.read_valid_json(imgpath, addkey=['2'])  #  65w

        self.img_list = self.key_list[1] + 2*self.key_list[2] + 5*self.key_list[3]  #
        
        if not self.debug:
            random.shuffle(self.img_list)
        
        self.tokenizer = tokenizer;  self.tokenizer2 = tokenizer2  
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.color_list = np.array([[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255]])
        

    def __len__(self):
        return len(self.img_list)

    def read_valid_json(self, imgpath, addkey=['1', '2', '3']):
        jsonname = 'test.json'
        valid_data = json.load(open(os.path.join(imgpath, jsonname), 'r'))
        for i in range(1,4):
            if str(i) in addkey:
                self.key_list[i] += valid_data[str(i)]
        print(imgpath, len(valid_data['1']), len(valid_data['2']), len(valid_data['3']))
        print('total key_list:', len(self.key_list[1]), len(self.key_list[2]), len(self.key_list[3]))
        

    def __getitem__(self, idx):
        data = self.img_list[idx] 
        return self.help_realistic(data)

    def help_realistic(self, data):
        cur_img, cur_pose, cur_mask, text, blen, faceid_path = data

        name = cur_img.split('/')[-1]; imgname = name
        
        ori_img = Image.open(cur_img).convert("RGB")
        oriw, orih = ori_img.size
        cur_img = resize_img_scale(ori_img, self.wh)
        gt_w, gt_h = cur_img.size

        cur_pose = Image.open(cur_pose).convert("RGB")
        box_w, box_h = cur_pose.size  #  bbox shape same to pose, use it to rescale bbox
        ori_mask = Image.open(cur_mask).convert("RGB").resize((oriw, orih))
        
        bbox = np.zeros((blen, 1))
        mp_list, maxw, maxh, face_list, mask_list,face_kps_abs = self.crop_refimg(ori_img, ori_mask, bbox, faceid_path=faceid_path, return_mask=True, imgname=imgname)
        
        faceid_list = []
        for idx, ref_img in enumerate(mp_list):
            face_id_embed = torch.load(os.path.join(faceid_path, f'{idx}.bin'), map_location="cpu")['id']
            
            if self.faceid_loss>0:
                M=face_align.estimate_norm(face_kps_abs[idx]/oriw*gt_w, image_size=112, )
                Mt = cal_torch_theta(M, gt_h,gt_w, 112, 112)
                faceid_list.append([ref_img, face_id_embed, face_list[idx], mask_list[idx], Mt])
            else:
                faceid_list.append([ref_img, face_id_embed, face_list[idx], mask_list[idx]])
        if self.sort_person:
            faceid_list = sorted(faceid_list, key=lambda x:  (np.nonzero(x[3][:,:,0])[1].min()+np.nonzero(x[3][:,:,0])[1].max())//2 )
        # else:
        #     random.shuffle(faceid_list)
        if self.faceid_loss>0:
            clip_image, face_id_embed, clip_face, mask_gt, face_kps_abs = self.concat_clip_faceid_addface(faceid_list, gt_h, gt_w)
        else:
            clip_image, face_id_embed, clip_face, mask_gt = self.concat_clip_faceid_addface(faceid_list, gt_h, gt_w)
            face_kps_abs = torch.zeros_like(face_id_embed)
        face_unnorm_embed = torch.zeros_like(face_id_embed)
               
        image_gt = self.transform(cur_img)
        pose_cond = self.conditioning_transforms(cur_pose.resize((gt_w, gt_h)))
        
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        text_input_ids2 = self.tokenizer2(
            text,
            max_length=self.tokenizer2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        w,h = self.wh
        topleft = [gt_h, gt_w, 0, 0, h, w]
        
        return {
            "image_gt": image_gt,  #  B,3,resolution,resolution
            "image_ref": pose_cond,  #  B,3,resolution,resolution
            "text_input_ids": text_input_ids,  #B,77
            "text_input_ids2": text_input_ids2,  #B,77
            "clip_image": clip_image,   #  B,3,224,224
            "drop_image_embed": drop_image_embed,
            "topleft": torch.tensor([topleft]), 
            "face_id_embed": face_id_embed, "face_kps_abs": face_kps_abs,
            "face_unnorm_embed": face_unnorm_embed,
            "clip_face": clip_face,
            "mask_gt": mask_gt,  "style": torch.tensor([0])
        }

    def crop_refimg(self, ori_img, ori_mask, bbox=None, rotate=1, faceid_path=None, return_mask=False, imgname='aaa'):
        mp_list = []; crop_list = []; face_list = [];  mask_list = []; head_list = [];  face_kps_abs = []
        w, h = ori_img.size
        blen = bbox.shape[0]
        maxh, maxw = 0,0
        for i in range(min(3, blen)):
            cv2_mask = cv2.cvtColor(np.array(ori_mask), cv2.COLOR_RGB2BGR)
            mask = cv2.inRange(cv2_mask, self.color_list[i], self.color_list[i])
            
            mask = np.tile(mask[:,:,None], (1,1,3)); 
            mask_ori = cv2.erode(mask, np.ones((7,7), np.uint8), iterations=1)
            mask = cv2.dilate(mask, np.ones((7,7), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0);  
            mask_list.append(mask)
            crop, mask, mask_ori, lefttop = self.bounding_rectangle(ori_img, mask, mask_ori)
            crop = (255*np.ones_like(mask)*(1-mask)+mask*np.array(crop)).astype(np.uint8)
            
            face_kps = torch.load(os.path.join(faceid_path, f'{i}.bin'), map_location="cpu")['kps']
            
            face_image = face_align.norm_crop(crop, landmark=face_kps.numpy(), image_size=224)  #  224
            clip_face = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
            face_list.append(clip_face)
            if self.faceid_loss>0 or return_mask:
                kps_abs = face_kps.numpy()+lefttop; face_kps_abs.append(kps_abs)
                
            ref_img = Image.fromarray(crop);  crop_list.append(ref_img)
            
            rw,rh = ref_img.size
            
            if rotate:
                rw,rh = ref_img.size
                angle_range = 10
                ref_img = ref_img.rotate(random.uniform(-angle_range, angle_range), fillcolor = 'white', expand=False)
            ref_img = ref_img.resize((224, 224), resample=LANCZOS)  #  square keep full info after 05.23
            
            mp_list.append(ref_img)
            w, h = ref_img.size
            maxw, maxh = max(maxw, w), max(maxh, h)
        if self.debug:
            tmp = [[crop, mask] for crop, mask in zip(crop_list, mask_list)]
            tmp = sorted(tmp, key=lambda x: (np.nonzero(x[1][:,:,0])[1].min()+np.nonzero(x[1][:,:,0])[1].max())//2)
            crop_list = [x[0] for x in tmp]
            debug_cropimg(ori_img, ori_mask, crop_list, head_list, imgname)
        if faceid_path is not None:
            if return_mask:
                return mp_list, maxw, maxh, face_list, mask_list, face_kps_abs
            else:
                return mp_list, maxw, maxh, face_list
        return mp_list, maxw, maxh

    def concat_clip_faceid_addface(self, mp_list, gt_h, gt_w):
        out_list = []; id_list = []; face_list = [];  mask_list = []; unnorm_list = []
        for each in mp_list:
            if self.use_unnorm:
                ref_img, id_embed, unnorm_embed, face_clip, mask = each
                unnorm_list.append(unnorm_embed)
            elif self.faceid_loss>0:
                ref_img, id_embed, face_clip, mask, kps = each
                unnorm_list.append(torch.from_numpy(kps).unsqueeze(0))  # to 1x5x2
            else:
                ref_img, id_embed, face_clip, mask = each
            clip_image = self.clip_image_processor(images=ref_img, return_tensors="pt").pixel_values
            out_list.append(clip_image)
            id_list.append(id_embed)
            face_list.append(face_clip)
            mask = cv2.resize(mask, (gt_w//self.mask_ratio, gt_h//self.mask_ratio))
            mask_list.append(torch.from_numpy(mask[:,:,:1]/255.))

        clip_image = torch.cat(out_list, dim=0)
        id_embed = torch.cat(id_list, dim=0)
        clip_face = torch.cat(face_list, dim=0)
        mask = torch.cat(mask_list, dim=2)
        if self.use_unnorm:
            unnorm_embed = torch.cat(unnorm_list, dim=0)
            return clip_image, id_embed, unnorm_embed, clip_face, mask
        elif self.faceid_loss>0:
            face_kps = torch.cat(unnorm_list, dim=0) #  num_imgs, 5, 2
            return clip_image, id_embed, clip_face, mask, face_kps
        return clip_image, id_embed, clip_face, mask

    def bounding_rectangle(self, ori_img, mask, mask_ori):
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
            crop = np.array(ori_img.crop((min_x, min_y, max_x, max_y)))
            mask = mask[min_y:max_y, min_x:max_x]
            mask_ori = mask_ori[min_y:max_y, min_x:max_x]
        except:
            traceback.print_exc()
            pass 
        return crop, mask/255., mask_ori, np.array([min_x,min_y]).reshape(1,2)  #  left, top

def debug_cropimg(ori_img, ori_mask, crop_list, head_list=[], imgname=None):
    w, h = ori_img.size
    img2 = Image.new("RGB", (w*2, h*2), "black")
    img2.paste(ori_img, (0,0))
    img2.paste(ori_mask, (w,0))
    img2.paste(crop_list[0], (0,h))
    if len(head_list)>0:
        img2.paste(head_list[0], (w//2,h))
    if len(crop_list)>1:
        img2.paste(crop_list[1], (w, h))
        if len(head_list)>0:
            img2.paste(head_list[1], (w//2*3,h))
        if len(crop_list)>2:
            img2.paste(crop_list[2], (w//3, h//3))
            if len(head_list)>0:
                img2.paste(head_list[2], (w//3*2,h//3))
    if imgname is None:
        imgname=generate_random_string(16)+'.jpg'
    savename = os.path.join(savepath, imgname)
    img2.save(savename)
    print(imgname, h,w, len(crop_list))
    # pdb.set_trace()
  
savepath = 'imgs/'; os.makedirs(savepath, exist_ok=True)
import random
import string
def generate_random_string(length):
    # 生成随机的数字和字母
    letters = string.ascii_letters + string.digits
    # 生成指定长度的随机字符串
    return ''.join(random.choice(letters) for i in range(length))

import torch.utils.data.distributed as dist
def test_datasets():
    base_model = 'stable-diffusion-xl-base-1.0'
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, subfolder="tokenizer",  use_fast=False
    )
    tokenizer2 = AutoTokenizer.from_pretrained(
        base_model, subfolder="tokenizer_2",  use_fast=False
    )
    train_dataset = MasktileDataset(args=0, tokenizer=tokenizer, tokenizer2=tokenizer2, debug=True)

    t0=time.time(); res = []
    for idx in range(1000):
        data = train_dataset.__getitem__(idx)
    


if __name__ == '__main__':
    test_datasets()
