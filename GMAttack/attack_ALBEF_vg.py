import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import grounding_dataset
from dataset.utils import grounding_eval

from refTools.refer_python3 import REFER

from attack import *
from torchvision import transforms
from PIL import Image


def load_ref_model(device):
    print("Creating reference model")

    # Bert
    ref_tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)
    ref_model = ref_model.to(device).eval()

    print("Loading reference model successfully")
    return ref_model, ref_tokenizer


def my_text_attack(text_attacker, model_ALBEF, ref_tokenizer, texts, images, criterion, device, sample_num=10, rate=0.7):
    text_input = ref_tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
    text_output = model_ALBEF.inference_text(text_input)
    text_embeds = text_output['text_embed'][:, 0, :].detach()
    image_output = model_ALBEF.inference_image(images)
    image_embeds = image_output['image_embed'][:, 0, :].detach()

    adv_text_bank = text_attacker.attack(model_ALBEF, texts, image_embeds, sample_num=sample_num)

    final_adv = []
    if sample_num==1:
        for i in range(len(texts)):
            final_adv.append(adv_text_bank[i][0])
        return final_adv

    for i in range(len(texts)):
        score = []
        for j in range(len(adv_text_bank[i])):
            adv_text = adv_text_bank[i][j]
            adv_input = ref_tokenizer(adv_text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
            adv_output = model_ALBEF.inference_text(adv_input)
            adv_embed = adv_output['text_embed'][:, 0, :].detach()
            loss1 = criterion(adv_embed.log_softmax(dim=-1), image_embeds[i].softmax(dim=-1))
            loss2 = criterion(adv_embed.log_softmax(dim=-1), text_embeds[i].softmax(dim=-1))
            score.append(loss1 * rate - loss2 * (1-rate))
        idx = score.index(max(score))      
        final_adv.append(adv_text_bank[i][idx])
    
    return final_adv


def my_image_text_attack(test_loader, ref_model, ref_tokenizer, model, block_num, mod):
    model.eval()
    device = args.gpu[0]
            
    num_iters = config['num_iters']
    sample_num = config['sample_num']
    alpha = config['alpha']
    print('test_image_text_attack: sample_num = %d' % sample_num)
    
    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    metric_logger = utils.MetricLogger(delimiter="  ")
    image_attacker = ImageAttacker(config['epsilon'] / 255., preprocess=images_normalize, bounding=(0, 1), cls=False)
    text_attacker = myBertAttack(ref_model, ref_tokenizer, cls=True)
     
    result = []
    for images, texts, ref_ids in metric_logger.log_every(test_loader, 10, 'Evaluation:'):
        images = images.to(device)

        if mod:
            origin_output = model.inference_image(images_normalize(images))
            origin_embeds = origin_output['image_embed'].flatten(1).detach()
            text_input = ref_tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
            text_output = model.inference_text(text_input)
            text_embeds = text_output['text_embed'][:, 0, :].detach()

            image_attack = image_attacker.attack(images, num_iters)
            for i in range(num_iters):
                image_diversity = next(image_attack)
                adv_output = model.inference_image(image_diversity)
                adv_embeds = adv_output['image_embed'].flatten(1)
                adv_embeds_ = adv_output['image_embed'][:, 0, :]

                loss1 = criterion(adv_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1))
                loss2 = criterion(F.normalize(adv_embeds_, dim=-1).log_softmax(dim=-1), 
                                  F.normalize(text_embeds, dim=-1).softmax(dim=-1))
                loss = loss1 + alpha * loss2
                loss.backward()
            images_adv = next(image_attack)
            images_adv_norm = images_normalize(images_adv)

            texts_adv = my_text_attack(text_attacker, model, ref_tokenizer, texts, images_adv_norm, criterion, device, sample_num=sample_num)
        else:
            images_adv_norm = images_normalize(images)
            texts_adv = texts

        text_inputs = ref_tokenizer(texts_adv, padding='longest', return_tensors="pt").to(device)
        
        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
        
        image_embeds = model.visual_encoder(images_adv_norm)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
        output = model.text_encoder(text_inputs.input_ids,
                                    attention_mask = text_inputs.attention_mask,
                                    encoder_hidden_states = image_embeds,
                                    encoder_attention_mask = image_atts,
                                    return_dict = True,
                                    )

        vl_embeddings = output.last_hidden_state[:,0,:]
        vl_output = model.itm_head(vl_embeddings)
        loss = vl_output[:,1].sum()

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            mask = text_inputs.attention_mask.view(text_inputs.attention_mask.size(0),1,-1,1,1)

            grads = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients().detach()
            cams = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map().detach()

            cams = cams[:, :, :, 1:].reshape(images.size(0), 12, -1, 24, 24) * mask
            grads = grads[:, :, :, 1:].clamp(min=0).reshape(images.size(0), 12, -1, 24, 24) * mask

            gradcam = cams * grads
            gradcam = gradcam.mean(1).mean(1)

        for r_id, cam in zip(ref_ids, gradcam):
            result.append({'ref_id':r_id.item(), 'pred':cam})

        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = False

    return result


def main(args, config):
    device = args.gpu[0]
    mod = 1

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    grd_test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')
    test_loader = DataLoader(grd_test_dataset, batch_size=config['batch_size'], num_workers=4)
    print("Creating dataset successfully")
    
    ## refcoco evaluation tools
    refer = REFER(config['refcoco_data'], 'refcoco+', 'unc')
    dets = json.load(open(config['det_file'],'r'))
    cocos = json.load(open(config['coco_file'],'r'))    

    #### Model #### 
    ref_model, ref_tokenizer = load_ref_model(device)
    
    model = ALBEF(config = config, text_encoder=args.text_encoder, tokenizer=ref_tokenizer)
    checkpoint = torch.load('/checkpoints/ALBEF_refcoco.pth', map_location='cpu')
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint
    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.','')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    msg = model.load_state_dict(state_dict,strict=False)
    model = model.to(device)

    result = my_image_text_attack(test_loader, ref_model, ref_tokenizer, model, args.block_num, mod=mod)

    grounding_acc = grounding_eval(result, dets, cocos, refer, alpha=0.5, mask_size=24)

    log_stats = {**{f'{k}': v for k, v in grounding_acc.items()},
                 'eval type': mod, 'eps': config['epsilon'], 'iters':config['num_iters'], 'sample_num':config['sample_num'], 'alpha':config['alpha']}

    with open(os.path.join(args.output_dir, "log.txt"),"a+") as f:
        f.write(json.dumps(log_stats) + "\n")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Grounding.yaml')
    parser.add_argument('--output_dir', default='./output/VG')
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
