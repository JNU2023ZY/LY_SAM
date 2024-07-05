# from PIL import Image
from numpy.lib.type_check import imag
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import cv2
import importlib
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import math
import random
import collections
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

DIR_PATH = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(DIR_PATH)
import vot_utils
from tools.transfer_predicted_mask2vottype import transfer_mask


AOT_PATH = os.path.join(os.path.dirname(__file__), '../dmaot')
sys.path.append(AOT_PATH)

import dmaot.dataloaders.video_transforms as tr
from torchvision import transforms
from dmaot.networks.engines import build_engine
from dmaot.utils.checkpoint import load_network
from dmaot.networks.models import build_vos_model
from utils.metric import pytorch_iou

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)

class AOTTracker(object):
    def __init__(self, cfg, gpu_id):
        self.with_crop = False
        self.EXPAND_SCALE = None
        self.small_ratio = 12
        self.mid_ratio = 100
        self.large_ratio = 0.5
        self.AOT_INPUT_SIZE = (465, 465)
        self.gpu_id = gpu_id
        
        self.frame = []
        self.mask = []
        self.obj_num = 0

        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.engine = build_engine(cfg.MODEL_ENGINE,
                                   phase='eval',
                                   aot_model=self.model,
                                   gpu_id=gpu_id,
                                   short_term_mem_skip=cfg.TEST_SHORT_TERM_MEM_GAP,
                                   long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP)

        self.transform = transforms.Compose([
        # tr.MultiRestrictSize_(cfg.TEST_MAX_SHORT_EDGE,
        #                         cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, cfg.TEST_INPLACE_FLIP,
        #                         cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
        tr.MultiRestrictSize(cfg.TEST_MAX_SHORT_EDGE,
                                cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP,
                                cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
        tr.MultiToTensor()
        ])  
        self.model.eval()

    # add the first frame and label
    def add_first_frame(self, frame, mask, object_num):
        sample = {
            'current_img': frame,
            'current_label': mask,
        }
        sample['meta'] = {
            'obj_num': object_num,
            'height': frame.shape[0],
            'width': frame.shape[1],
        }
        sample = self.transform(sample)
        
        frame = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
        mask = sample[0]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)

        mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")
        
        # add reference frame
        self.engine.add_reference_frame(frame, mask, frame_step=0, obj_nums=object_num)
        
        self.frame = frame.cpu().tolist()
        self.mask = mask.cpu().tolist()
      
        self.obj_num = object_num

    
    def track(self, image):
        
        height = image.shape[0]
        width = image.shape[1]
        
        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
        }
        sample = self.transform(sample)
        output_height = sample[0]['meta']['height']
        output_width = sample[0]['meta']['width']
        image = sample[0]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)

        self.engine.match_propogate_one_frame(image)
        pred_logit = self.engine.decode_current_logits((output_height, output_width))
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1,
                                    keepdim=True).float()

        _pred_label = F.interpolate(pred_label,
                                    size=self.engine.input_size_2d,
                                    mode="nearest")
        self.engine.update_memory(_pred_label)

        mask = pred_label.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        return mask


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def _rect_from_mask(mask):
    if len(np.where(mask==1)[0]) == 0:
        return None
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def select_tracker(img, mask):
    img_sz = img.shape[0] * img.shape[1]
    _, _, w, h = _rect_from_mask(mask)
    max_edge = max(w, h)
    rect_sz = max_edge * max_edge
    ratio = img_sz / rect_sz
    print("ratio = {ratio}")
    if ratio > 900:
        return "aot_mix"
    else:
        return "aot"

class LYAOTTracker(object):
    def __init__(self, cfg, config, local_track=False,sam_refine=False,sam_refine_iou=0):
        self.mask_size = []
        self.max_mask = []
        self.max_bbox = []
        self.max_score = [-1 for i in range(10)]
        self.drop = 0
        self.fist_frame = []
        self.first_mask = []
        self.first_seg_mask = None
        self.f_bbox = None
        self.fist_obj_num = 0
        self.local_track = local_track
        self.aot_tracker = AOTTracker(cfg, config['gpu_id'])
        self.replace = False
        # SAM
        self.sam_refine=sam_refine
        if self.sam_refine:
            model_type = 'vit_h' #'vit_h'
            sam_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'segment_anything_hq/pretrained_model/sam_hq_vit_h.pth')
            output_mode = "binary_mask"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=torch.device('cuda'))
            self.mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
            self.mask_prompt = SamPredictor(sam)
        self.sam_refine_iou=sam_refine_iou

    def get_box(self, label):
        thre = np.max(label) * 0.5
        label[label > thre] = 1
        label[label <= thre] = 0
        a = np.where(label != 0)
        height, width = label.shape
        ratio = 0.075

        if len(a[0]) != 0:
            bbox1 = np.stack([np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])])
            w, h = np.max(a[1]) - np.min(a[1]), np.max(a[0]) - np.min(a[0])
            x1 = max(int(bbox1[0] - w * ratio), 0)
            y1 = max(int(bbox1[1] - h * ratio), 0)
            x2 = min(int(bbox1[2] + w * ratio), width)
            y2 = min(int(bbox1[3] + h * ratio), height)
            bbox = np.array([x1, y1, x2, y2])
        else:
            bbox = np.array([0, 0, 0, 0])
        return bbox
        
    def get_first_box(self, label, obj_num):
        bbox = []
        for i in range(obj_num):
            t = np.zeros_like(label)
            t[label == i + 1] = 1
            t[label != i + 1] = 0
            a = np.where(t != 0)
            height, width = label.shape
            ratio = 0.075

            if len(a[0]) != 0:
                bbox1 = np.stack([np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])])
                w, h = np.max(a[1]) - np.min(a[1]), np.max(a[0]) - np.min(a[0])
                x1 = max(int(bbox1[0] - w * ratio), 0)
                y1 = max(int(bbox1[1] - h * ratio), 0)
                x2 = min(int(bbox1[2] + w * ratio), width)
                y2 = min(int(bbox1[3] + h * ratio), height)
                bbox.append(np.array([x1, y1, x2, y2]))
            else:
                bbox.append(np.array([0, 0, 0, 0]))
            

                                
        return bbox        
        

    def initialize(self, image, mask,object_num):
        self.tracker = self.aot_tracker
        self.tracker.add_first_frame(image, mask,object_num)
        self.aot_mix_tracker = None
        self.mask_size = mask.shape
        self.first_frame = self.tracker.frame
        self.first_mask = self.tracker.mask
        self.drop = 0
        
        self.fist_obj_num = self.tracker.obj_num
        
        first_mask = np.array(self.first_mask)
        size = first_mask.shape
        first_mask = first_mask.reshape(size[2],size[3])
            
            
                
        self.f_bbox = self.get_first_box(first_mask,self.fist_obj_num)
#            first_bbox_mask = first_mask[f_bbox[1]:f_bbox[3], f_bbox[0]:f_bbox[2]]
        self.first_seg_mask = np.array(self.seg_first_mask(first_mask, self.fist_obj_num))
        
        
    def max3(self, a,b,c):
        max = 0
        if a > b:
            if a >c:
                max = a
            else:
                max = c
        elif b >c:
            max = b
            
        else:
            max = c
        
        return max
        
    def min3(self, a,b,c):
        min = 0
        if a < b:
            if a < c:
                min = a
            else:
                min = c
        elif b < c:
            min = b
            
        else:
            min = c
        
        return min
        
    def seg_first_mask(self, mask , obj_num):
        masks = []
        for i in range(obj_num):
            m = np.zeros_like(mask)
            m[mask == i + 1] = 1
            m[mask != i + 1] = 0
            
            masks.append(m)
        
        return masks
    
        
    def mask_pad(self, mask,bbox,w,h):
        b_w = bbox[2] - bbox[0]
        b_h = bbox[3] - bbox[1]
        right = 0
        left = 0
        up = 0
        down = 0
        if (w - b_w) % 2 != 0:
            right = int((w - b_w) / 2)
            left = right + 1
        else:
            right = int((w - b_w) / 2)
            left = right
            
        if (h - b_h) % 2 != 0:
            up = int((h - b_h) / 2)
            down = up + 1
        else:
            up = int((h - b_h) / 2)
            down = up   
            
        mask = np.pad(mask,((up, down), (left, right)), "constant", constant_values=0)       

        return mask
        
    def cosine_sim(self, m1, m2):
    

        vec1 = np.array(m1,dtype=np.int16).flatten()
        vec2 = np.array(m2,dtype=np.int16).flatten()


            
        # Compute dot product
        dot_product = np.dot(vec1, vec2)

    
        # Compute norms
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
    
        # Compute cosine similarity
        cosine_sim = dot_product / (norm1 * norm2)
    
        return cosine_sim
           

    def track(self, image):
        m = self.tracker.track(image)
        m = F.interpolate(torch.tensor(m)[None, None, :, :],
                          size=self.mask_size, mode="nearest").numpy().astype(np.uint8)[0][0]

        if self.sam_refine:
            obj_list = np.unique(m)
            mask_ = np.zeros_like(m)
            mask_2 = np.zeros_like(m)
            masks_ls = []
            score_mask = 0
            score_out = 0

            obj_n = 0
            
            for i in obj_list:
                mask = (m == i).astype(np.uint8)
                if i == 0 or mask.sum() == 0:
                    masks_ls.append(mask_)
                    continue
                bbox = self.get_box(mask)
                
                # box prompt
                self.mask_prompt.set_image(image)
                masks_, iou_predictions, _ = self.mask_prompt.predict(box=bbox)
                select_index = list(iou_predictions).index(max(iou_predictions))
                output = masks_[select_index].astype(np.uint8)
                iou = pytorch_iou(torch.from_numpy(output).cuda().unsqueeze(0),
                                  torch.from_numpy(mask).cuda().unsqueeze(0), [1])
                iou = iou.cpu().numpy()
                
                
                ma = mask
                o = output

                
                m_bbox = self.get_box(ma)
                o_bbox = self.get_box(o)
                
                if self.replace == False:
                    first_bbox_mask = self.first_seg_mask[obj_n][self.f_bbox[obj_n][1]:self.f_bbox[obj_n][3], self.f_bbox[obj_n][0]:self.f_bbox[obj_n][2]]
                    m_bbox_mask = ma[m_bbox[1]:m_bbox[3], m_bbox[0]:m_bbox[2]]
                    o_bbox_mask = o[o_bbox[1]:o_bbox[3], o_bbox[0]:o_bbox[2]]
                

                else:
                    self.first_seg_mask = np.array(self.max_mask)
                    self.f_bbox = self.max_bbox
                    first_bbox_mask = self.first_seg_mask[obj_n][self.f_bbox[obj_n][1]:self.f_bbox[obj_n][3], self.f_bbox[obj_n][0]:self.f_bbox[obj_n][2]]
                    m_bbox_mask = ma[m_bbox[1]:m_bbox[3], m_bbox[0]:m_bbox[2]]
                    o_bbox_mask = o[o_bbox[1]:o_bbox[3], o_bbox[0]:o_bbox[2]]
                    
                    
                    
                    
                x1_min = self.min3(self.f_bbox[obj_n][0], m_bbox[0], o_bbox[0])
                y1_min = self.min3(self.f_bbox[obj_n][1],m_bbox[1],o_bbox[1])
                x2_max = self.max3(self.f_bbox[obj_n][2],m_bbox[2],o_bbox[2])
                y2_max = self.max3(self.f_bbox[obj_n][3],m_bbox[3],o_bbox[3])
                
                w = x2_max - x1_min
                h = y2_max - y1_min
                
                first_bbox_mask = self.mask_pad(first_bbox_mask,self.f_bbox[obj_n],w,h)
                m_bbox_mask = self.mask_pad(m_bbox_mask, m_bbox, w, h)
                o_bbox_mask = self.mask_pad(o_bbox_mask, o_bbox, w, h)
                
                
                
                first_bbox_mask = first_bbox_mask.flatten()
                m_bbox_mask = m_bbox_mask.flatten()
                o_bbox_mask = o_bbox_mask.flatten()
                
                dist1=np.linalg.norm(first_bbox_mask-m_bbox_mask)
                dist2=np.linalg.norm(first_bbox_mask-o_bbox_mask)
                
                dot_fm = np.dot(first_bbox_mask, m_bbox_mask)
                dot_fo = np.dot(first_bbox_mask, o_bbox_mask)
                
                          
                normf = np.linalg.norm(first_bbox_mask)
                normm = np.linalg.norm(m_bbox_mask)
                normo = np.linalg.norm(o_bbox_mask)
                
                score_mask = dot_fm / (normf * normm)
                score_out = dot_fo / (normf * normo)
                    
                    
                if score_mask > score_out:
                    a= score_mask
                    flag=1
                else :
                    a= score_out
                    flag=0
                #a=max(score_out,score_mask)
                
                if  a > self.max_score[obj_n] or self.replace == True:
                    self.max_score[obj_n] = a
                    
                    
                    if flag==1 :
                        if self.drop == 1:
                            self.max_mask[obj_n] = ma
                            self.max_bbox[obj_n] = m_bbox
                            
                        #self.max_bbox = self.get_box(ma)
                        else:
                            self.max_mask.append(ma)
                            self.max_bbox.append(m_bbox)
                            

                    else :
                        if self.drop == 1 :
                            self.max_mask[obj_n] = o
                            self.max_bbox[obj_n] = o_bbox
                            
                        #self.max_bbox = self.get_box(o)
                        else:
                            self.max_mask.append(o)
                            self.max_bbox.append(o_bbox)
                            
                   
                
                
                if iou < self.sam_refine_iou:
                    output = mask
                elif score_mask > score_out:
                    output = mask

                    
                masks_ls.append(output)
                mask_2 = mask_2 + output * i
                
                obj_n = obj_n + 1
                
                if self.replace == True:
                    self.replace = False
                
            self.drop = 1
            masks_ls = np.stack(masks_ls)
            masks_ls_ = masks_ls.sum(0)
            masks_ls_argmax = np.argmax(masks_ls, axis=0)
            rs = np.where(masks_ls_ > 1, masks_ls_argmax, mask_2)
            rs = np.array(rs).astype(np.uint8)

            return rs
            
        return m
        

#####################
# config
#####################
vis_results = True
local_track = False
sam_refine = True
sam_refine_iou = 10
sam_refine_iou/=100.0
muti_object = True
save_mask = False
confidence_setto_1 = True

config = {
    'exp_name': 'default',
    'model': 'swinb_dm_deaotl',
    'pretrain_model_path': 'dmaot/pretrain_models/SwinB_DeAOTL_PRE_YTB_DAV_VIP_MOSE_OVIS_LASOT_GOT.pth',
    'config': 'pre_ytb_dav',
    'long_max': 10,
    'long_gap': 30,
    'short_gap': 2,
    'patch_wised_drop_memories': False,
    'patch_max': 999999,
    'gpu_id': 0,
}

# get first frame and mask
handle = vot_utils.VOT("mask", multiobject=True)

objects = handle.objects()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

# get first
image = read_img(imagefile)


# Get merged-mask
merged_mask = np.zeros((image.shape[0], image.shape[1]))
object_num = len(objects)
object_id = 1
for object in objects:
    mask = make_full_size(object, (image.shape[1], image.shape[0]))
    mask = np.where(mask > 0, object_id, 0)    
    merged_mask += mask
    object_id += 1
    # print("Save")
    


# set cfg
engine_config = importlib.import_module('configs.' + f'{config["config"]}')
cfg = engine_config.EngineConfig(config['exp_name'], config['model'])
cfg.TEST_CKPT_PATH = os.path.join(DIR_PATH, config['pretrain_model_path'])
cfg.TEST_LONG_TERM_MEM_MAX = config['long_max']
cfg.TEST_LONG_TERM_MEM_GAP = config['long_gap']
cfg.TEST_SHORT_TERM_MEM_GAP = config['short_gap']
cfg.PATCH_TEST_LONG_TERM_MEM_MAX = config['patch_max']
cfg.PATCH_WISED_DROP_MEMORIES = True if config['patch_wised_drop_memories'] else False

### init trackers
tracker = LYAOTTracker(cfg, config, local_track, sam_refine, sam_refine_iou)

# initialize tracker
tracker.initialize(image, merged_mask,object_num)
mask_size = merged_mask.shape
frame = 1
while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    image = read_img(imagefile)
    
    pred_masks = []
    if (frame % 200) == 0:
        tracker.replace = True
    m = tracker.track(image)
    
    for i in range(object_num):
        m_temp = m.copy()
        m_temp[m_temp != i + 1] = 0
        m_temp[m_temp != 0] = 1
        pred_masks.append(m_temp)

    handle.report(pred_masks)
    frame = frame + 1
