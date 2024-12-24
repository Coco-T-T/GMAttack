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

from models.clip import clip
from models.tokenization_bert import BertTokenizer

from transformers import BertForMaskedLM

import utils

from attack import *
from torchvision import transforms

from dataset import pair_dataset
from PIL import Image
from torchvision import transforms


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
    text_embeds = text_output['text_embed'].flatten(1).detach()
    image_output = model_ALBEF.inference_image(images)
    image_embeds = image_output['image_embed'].flatten(1).detach()

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
            adv_embed = adv_output['text_embed'].flatten(1).detach()
            loss1 = criterion(adv_embed.log_softmax(dim=-1), image_embeds[i].softmax(dim=-1))
            # loss1 = criterion(text_embeds[i].log_softmax(dim=-1), image_embeds[i].softmax(dim=-1))
            loss2 = criterion(adv_embed.log_softmax(dim=-1), text_embeds[i].softmax(dim=-1))
            score.append(loss1 * rate - loss2 * (1-rate))
        idx = score.index(max(score))      
        final_adv.append(adv_text_bank[i][idx])
    
    return final_adv
            

def my_image_text_attack(model, ref_model, data_loader, ref_tokenizer, device, config, mod):
    model.float()
    model.eval()
    device = args.gpu[0]
    start_time = time.time()

    num_iters = config['num_iters']
    sample_num = config['sample_num']
    alpha = config['alpha']
    print('test_image_text_attack: sample_num = %d' % sample_num)

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    metric_logger = utils.MetricLogger(delimiter="  ")
    image_attacker = ImageAttacker(config['epsilon'] / 255., preprocess=images_normalize, bounding=(0, 1), cls=False)
    text_attacker = myBertAttack(ref_model, ref_tokenizer, cls=False)
    
    print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    image_feats = torch.zeros(num_image, model.visual.output_dim)
    text_feats = torch.zeros(num_text, model.visual.output_dim)
    
    for images, texts, texts_ids in metric_logger.log_every(data_loader, 10, 'Evaluation:'):
        images = images.to(device)

        if mod:
            origin_output = model.inference_image(images_normalize(images))
            origin_embeds = origin_output['image_embed'].flatten(1).detach()
            text_input = ref_tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
            text_output = model.inference_text(text_input)
            text_embed = text_output['text_embed'].flatten(1).detach()

            image_attack = image_attacker.attack(images, num_iters)
            for i in range(num_iters):
                image_diversity = next(image_attack)
                adv_output = model.inference_image(image_diversity)
                adv_embeds = adv_output['image_embed'].flatten(1)

                loss1 = criterion(adv_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1))
                loss2 = criterion(F.normalize(adv_embeds, dim=-1).log_softmax(dim=-1), 
                                  F.normalize(text_embed, dim=-1).softmax(dim=-1))
                loss = loss1 + alpha * loss2
                loss.backward()
            images_adv = next(image_attack)
            images_adv_norm = images_normalize(images_adv)

            texts_adv = my_text_attack(text_attacker, model, ref_tokenizer, texts, images_adv_norm, criterion, device, sample_num=sample_num)
        else:
            images_adv_norm = images_normalize(images)
            texts_adv = texts

        images_ids = [data_loader.dataset.txt2img[i.item()] for i in texts_ids]
        
        with torch.no_grad():
            output = model.inference(images_adv_norm, texts_adv)
            image_feats[images_ids] = output['image_feat'].cpu().float().detach()
            text_feats[texts_ids] = output['text_feat'].cpu().float().detach()
    
    sims_matrix = image_feats @ text_feats.t()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def main(args, config):
    device = args.gpu[0]
    mod = 1

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Model ####
    ref_model, ref_tokenizer = load_ref_model(device) 
    model, preprocess = clip.load(args.image_encoder, device=device)
    model.set_tokenizer(ref_tokenizer)
    model = model.to(device)

    #### Dataset ####
    print("Creating dataset")
    n_px = model.visual.input_resolution
    test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])
    test_dataset = pair_dataset(config['test_file'], test_transform, config['image_root'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=4)
    print("Creating dataset successfully")

    score_i2t, score_t2i = my_image_text_attack(model, ref_model, test_loader, ref_tokenizer, device, config, mod=mod)

    result = itm_eval(score_i2t, score_t2i, test_dataset.img2txt, test_dataset.txt2img)
    print(result)
    log_stats = {**{f'test_{k}': v for k, v in result.items()},
                 'eval type': mod, 'eps': config['epsilon'], 'iters':config['num_iters'], 'sample_num':config['sample_num'], 'alpha':config['alpha'], 'dataset':config['dataset']}
    with open(os.path.join(args.output_dir, "log_CLIP.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    # parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='./output/VR')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--image_encoder', default='ViT-B/16')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
