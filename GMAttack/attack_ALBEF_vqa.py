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

from models.model_vqa import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.vqa_dataset import vqa_collate_fn
from dataset import create_dataset, create_loader
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


def my_text_attack(text_attacker, model, texts, images, criterion, device, sample_num=10, rate=0.7):
    text_input = model.tokenizer(texts, padding='longest', truncation=True, max_length=35, return_tensors="pt").to(device)
    text_output = model.inference_text(text_input)
    text_embeds = text_output['text_embed'][:, 0, :].detach()
    image_output = model.inference_image(images)
    image_embeds = image_output['image_embed'][:, 0, :].detach()

    adv_text_bank = text_attacker.attack(model, texts, image_embeds, sample_num=sample_num)

    final_adv = []
    if sample_num==1:
        for i in range(len(texts)):
            final_adv.append(adv_text_bank[i][0])
        return final_adv
    
    for i in range(len(texts)):
        score = []
        for j in range(len(adv_text_bank[i])):
            adv_text = adv_text_bank[i][j]
            adv_input = model.tokenizer(adv_text, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
            adv_output = model.inference_text(adv_input)
            adv_embed = adv_output['text_embed'][:, 0, :].detach()
            loss1 = criterion(adv_embed.log_softmax(dim=-1), image_embeds[i].softmax(dim=-1))
            loss2 = criterion(adv_embed.log_softmax(dim=-1), text_embeds[i].softmax(dim=-1))
            score.append(loss1 * rate - loss2 * (1-rate))
        idx = score.index(max(score))      
        final_adv.append(adv_text_bank[i][idx])
    
    return final_adv


def black_box_predict(model, ref_tokenizer, image, text, answer_list, answer_candidates, device):
    text_input = ref_tokenizer(text, padding='longest', return_tensors="pt").to(device)
    topk_ids, topk_probs, = model(image, text_input,
                                  answer_candidates, train=False,
                                  k=128)
    out_v = []
    for topk_id, topk_prob in zip(topk_ids, topk_probs):
        _, pred = topk_prob.max(dim=0)
        out_v.append({'answer': answer_list[topk_id[pred]]})
    return out_v
            

def my_image_text_attack(test_loader, ref_model, ref_tokenizer, model):
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
    
    cnt = 0 
    acc_list=[]
    answer_list = test_loader.dataset.answer_list
    answer_list_ = [answer + config['eos'] for answer in answer_list]
    answer_candidates = ref_tokenizer(answer_list_, padding='longest', return_tensors='pt').to(device) 
    for i, (images, texts, question_ids) in enumerate(metric_logger.log_every(test_loader, 10, 'Evaluation:')):
        cnt = cnt + 1
        if cnt > 1250:
            break

        images = images.to(device, non_blocking=True)
        pred_ans = black_box_predict(model, ref_tokenizer, images, texts, answer_list, answer_candidates, device)
        
        origin_output = model.inference_image(images_normalize(images))
        origin_embeds = origin_output['image_embed'].flatten(1).detach()
        text_input = model.tokenizer(texts, padding='longest', truncation=True, max_length=35, return_tensors="pt").to(device)
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

        texts_adv = my_text_attack(text_attacker, model, texts, images_adv_norm, criterion, device, sample_num=sample_num)

        with torch.no_grad():
            adv_pred_ans = black_box_predict(model, ref_tokenizer, images_adv, texts_adv, answer_list, answer_candidates, device)
            
        for idx in range(len(texts)):
            if adv_pred_ans[idx]['answer'] != pred_ans[idx]['answer']:
                acc_list.append(1)
            else:
                acc_list.append(0)
        asr = sum(acc_list) / len(acc_list)
        metric_logger.meters['ASR'].update(asr)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats: %.4f" % (sum(acc_list) / len(acc_list)))
    return format(sum(acc_list) / len(acc_list), '.4f')


def main(args, config):
    device = args.gpu[0]
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    #### Dataset ####
    print("Creating vqa datasets")
    datasets = create_dataset('vqa-albef', config)
    train_loader, test_loader = create_loader(datasets, [None, None],
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])
    print("Creating dataset successfully")

    #### Model ####
    ref_model, ref_tokenizer = load_ref_model(device)
    
    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=ref_tokenizer)
    checkpoint = torch.load('/checkpoints/vqa.pth', map_location='cpu')
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint
    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    test_stats = my_image_text_attack(test_loader, ref_model, ref_tokenizer, model)

    log_stats = {'test_ASR' : test_stats, 'eps': config['epsilon'], 'iters':config['num_iters'], 'sample_num':config['sample_num'], 'alpha':config['alpha']}

    with open(os.path.join(args.output_dir, "log.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA_ALBEF.yaml')
    parser.add_argument('--output_dir', default='./output/VQA')
    parser.add_argument('--text_encoder', default='bert-base-uncased')  
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', type=int, nargs='+',  default=[0])
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
