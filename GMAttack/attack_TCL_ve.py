import argparse
import os
import ruamel.yaml as yaml
from pathlib import Path
import random
import torch.backends.cudnn as cudnn
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from transformers import BertForMaskedLM

from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import ve_dataset
from PIL import Image
from torchvision import transforms
from attack import *


# When attack unimodal embedding, not using "--cls" will raise an expected error 
# due to the different sequence length of image embedding and text embedding.
# image cls=False
# text cls=False
# image+text cls=True


def load_ref_model(device):
    print("Creating reference model")

    # Bert
    ref_tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)
    ref_model = ref_model.to(device).eval()

    print("Loading reference model successfully")
    return ref_model, ref_tokenizer


def my_text_attack(text_attacker, model_ALBEF, ref_tokenizer, texts, images, criterion, device, sample_num=10, rate=0.7):
    # model_ALBEF  target
    # ref_model  generator
    # tokenizer  
    
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
            # loss1 = criterion(text_embeds[i].log_softmax(dim=-1), image_embeds[i].softmax(dim=-1))
            loss2 = criterion(adv_embed.log_softmax(dim=-1), text_embeds[i].softmax(dim=-1))
            score.append(loss1 * rate - loss2 * (1-rate))
        idx = score.index(max(score))      
        final_adv.append(adv_text_bank[i][idx])
    
    return final_adv
            

def my_image_text_attack(test_loader, ref_model, ref_tokenizer, model_ALBEF, mod):
    #### Image Attacker ####
    model_ALBEF.eval()
    device = args.gpu[0]

    num_iters = config['num_iters']
    sample_num = config['sample_num']
    alpha = config['alpha']
    print('test_image_text_attack: sample_num = %d' % sample_num)

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    metric_logger = utils.MetricLogger(delimiter="  ")
    image_attacker = ImageAttacker(config['epsilon'] / 255., preprocess=images_normalize, bounding=(0, 1), cls=False)
    # text_attacker = myBertAttack(ref_model, ref_tokenizer, cls=False)
    text_attacker = myBertAttack(ref_model, ref_tokenizer, cls=True) # update
    
    for images, texts, targets in metric_logger.log_every(test_loader, 10, 'Evaluation:'):
        images, targets = images.to(device), targets.to(device)

        if mod:
            origin_output = model_ALBEF.inference_image(images_normalize(images))
            origin_embeds = origin_output['image_embed'].flatten(1).detach()
            text_input = ref_tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
            text_output = model_ALBEF.inference_text(text_input)
            text_embeds = text_output['text_embed'][:, 0, :].detach()

            image_attack = image_attacker.attack(images, num_iters)
            for i in range(num_iters):
                image_diversity = next(image_attack)
                adv_output = model_ALBEF.inference_image(image_diversity)
                adv_embeds = adv_output['image_embed'].flatten(1)
                adv_embeds_ = adv_output['image_embed'][:, 0, :]

                loss1 = criterion(adv_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1))
                loss2 = criterion(F.normalize(adv_embeds_, dim=-1).log_softmax(dim=-1), 
                                  F.normalize(text_embeds, dim=-1).softmax(dim=-1))
                loss = loss1 + alpha * loss2
                loss.backward()
            images_adv = next(image_attack)
            images_adv_norm = images_normalize(images_adv)

            # texts_adv = my_text_attack(text_attacker, model_ALBEF, ref_tokenizer, texts, images_adv_norm, criterion, device, sample_num=sample_num)
            texts_adv = my_text_attack(text_attacker, model_ALBEF, ref_tokenizer, texts, images_normalize(images), criterion, device, sample_num=sample_num)
        else:
            images_adv_norm = images_normalize(images)
            texts_adv = texts

        text_inputs = ref_tokenizer(texts_adv, padding='longest', return_tensors="pt").to(device)
    
        with torch.no_grad():
            prediction = model_ALBEF(images_adv_norm, text_inputs, targets=targets, train=False)

        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)
        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


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
    datasets = ve_dataset(config['test_file'], test_transform, config['image_root'])
    test_loader = DataLoader(datasets, batch_size=config['batch_size_test'], num_workers=4)
    print("Creating dataset successfully")

    #### Model ####
    ref_model, ref_tokenizer = load_ref_model(device) 

    model_ALBEF = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=ref_tokenizer)
    checkpoint = torch.load("/checkpoints/TCL_ve.pth", map_location='cpu')
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model_ALBEF.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
    msg = model_ALBEF.load_state_dict(state_dict, strict=False)
    model_ALBEF = model_ALBEF.to(device)

    test_stats = my_image_text_attack(test_loader, ref_model, ref_tokenizer, model_ALBEF, mod=mod)

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'eval type': mod, 
                 'eps': config['epsilon'], 'iters':config['num_iters'], 'sample_num':config['sample_num'], 'alpha':config['alpha']}

    with open(os.path.join(args.output_dir, "log_TCL.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VE.yaml')
    parser.add_argument('--output_dir', default='./output/VE')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
