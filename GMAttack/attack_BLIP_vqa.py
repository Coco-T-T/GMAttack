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

from models.blip_vqa import blip_vqa
from models.blip_pretrain import blip_pretrain
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
    texts = [texts]
    text_input = model.tokenizer(texts, padding='longest', truncation=True, max_length=35, return_tensors="pt").to(device)
    text_output = model.inference_text(text_input)
    text_embeds = text_output['text_embed'][:, 0, :].detach()
    image_output = model.inference_image(images)
    image_embeds = image_output['image_embed'][:, 0, :].detach()

    adv_text_bank = text_attacker.attack(model, texts, image_embeds, sample_num=sample_num)
    if isinstance(adv_text_bank[0], str):
        adv_text_bank = [adv_text_bank]

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


def black_box_predict(model, image, text, answer_list, answer_candidates):
    answer_ids, topk_ids, topk_probs, = model(image, text,
                                              answer_candidates, train=False,
                                              inference='rank',
                                              k_test=128)
    out_v = []
    for answer_id in answer_ids:
        out_v.append({"answer": answer_list[answer_id]})
    return out_v[0]['answer']
            

def my_image_text_attack(test_loader, ref_model, ref_tokenizer, model, correct_list, ans_table):
    #### Image Attacker ####
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
    answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
    answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id
    for i, batch in enumerate(metric_logger.log_every(test_loader, 10, 'Evaluation:')):
        cnt = cnt + 1
        if cnt > 5000:
            break
        if int(batch['question_id'][0]) not in correct_list:
            continue

        images = batch['image'].to(device, non_blocking=True)
        texts = batch['question'][0]
        images = images.to(device)
        pred_ans = black_box_predict(model, images, texts, answer_list, answer_candidates)
        ret = dict()
        ret['preds'] = [ans_table[str(int(batch['question_id'][0]))]]
        if pred_ans != ret['preds'][0]:
            print('wrong answer here', pred_ans, ret['preds'][0])
            continue

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
            adv_pred_ans = black_box_predict(model, images_adv, texts_adv, answer_list, answer_candidates)

        if adv_pred_ans != ans_table[str(int(batch['question_id'][0]))]:
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
    datasets = create_dataset('vqa', config)
    train_loader, test_loader = create_loader(datasets, [None, None],
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[16, 16], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])
    print("Creating dataset successfully")

    #### Model ####
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_res'], vit=config['vit'])
    model = model.to(device)
    print("Creating model successfully")

    f = open(args.correct_idx_store, 'r')
    a = list(f)
    f.close()
    correct_list = [int(l.strip('\n')) for l in a][:5000]
    with open(args.correct_pred_store, 'r') as f:
        answer_list = json.load(f)

    ref_model, ref_tokenizer = load_ref_model(device) 
    
    test_stats = my_image_text_attack(test_loader, ref_model, ref_tokenizer, model, correct_list, answer_list)

    log_stats = {'test_ASR': test_stats, 'eps': config['epsilon'], 'iters':config['num_iters'], 'sample_num':config['sample_num'], 'alpha':config['alpha']}

    with open(os.path.join(args.output_dir, "log_BLIP.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA_BLIP.yaml')
    parser.add_argument('--output_dir', default='./output/VQA')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--correct_idx_store', default='./right_vqa_list.txt')
    parser.add_argument('--correct_pred_store', default='./right_vqa_ans_table.txt')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', type=int, nargs='+',  default=[3])
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
