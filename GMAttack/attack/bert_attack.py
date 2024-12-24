import torch
import torch.nn as nn
import copy
from transformers import BatchEncoding
import torch.nn.functional as F
# from nltk.corpus import stopwords

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)


class myBertAttack():
    def __init__(self, ref_net, tokenizer, cls=True):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.cls = cls

    def attack(self, net, texts, image_embeds, sample_num=10, k=10, num_perturbation=1, threshold_pred_score=0.3, max_length=30, batch_size=32):
        # num: each sentence will have num adversarial samples 
        # k: each word will have the first k substitutions
        # embedding bert-attack
        device = self.ref_net.device
        self.net = net

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)

        # substitutes
        # use Bert to predict the first k substitutions of each word in the original sentence
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, k, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        final_adverse_bank = []
        for i, text in enumerate(texts):
            final_adverse = []
            o_num_perturbation = num_perturbation
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], image_embeds[i], batch_size, max_length)
            # important word list
            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)

            while len(final_adverse) < sample_num:
                adv_list = self.create_adv_text(
                    word_pred_scores_all[i], 
                    word_predictions[i], 
                    origin_embeds[i], 
                    list_of_index,
                    words,
                    keys,
                    num_perturbation=o_num_perturbation)
                
                if len(adv_list)==0:
                    break
            
                if len(final_adverse) + len(adv_list) < sample_num :
                    final_adverse.append(adv_list)
                    o_num_perturbation += 1
                elif len(final_adverse) + len(adv_list) >= sample_num :
                    final_adverse.append(adv_list[:sample_num-len(final_adverse)])
                    break
            
            if len(final_adverse)==0:
                final_adverse = [' '.join(words)]
            final_adverse_bank.extend(final_adverse)

        return final_adverse_bank
    
    def create_adv_text(
            self,
            word_pred_scores_all, 
            word_predictions, 
            origin_embeds, 
            list_of_index,
            words,
            keys,
            num_perturbation,
            max_length=30):

        device = self.ref_net.device
        criterion = torch.nn.KLDivLoss(reduction='none')
        change = 0
        final_words = copy.deepcopy(words)
        final_adverse = []

        #start with the most important word one by one
        for top_index in list_of_index:
            tgt_word = words[top_index[0]] 
            if tgt_word in filter_words:
                continue
            if keys[top_index[0]][0] > max_length - 2:
                continue

            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

            substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores)

            replace_texts = [' '.join(final_words)]
            available_substitutes = [tgt_word]
            for substitute_ in substitutes:
                substitute = substitute_

                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word

                if substitute in filter_words:
                    continue
                    
                temp_replace = copy.deepcopy(final_words)
                temp_replace[top_index[0]] = substitute
                available_substitutes.append(substitute)
                replace_texts.append(' '.join(temp_replace))
            
            replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
            replace_output = self.net.inference_text(replace_text_input)
            if self.cls:
                replace_embeds = replace_output['text_embed'][:, 0, :]
            else:
                replace_embeds = replace_output['text_embed'].flatten(1)

            loss = criterion(replace_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1).repeat(len(replace_embeds), 1))
            loss = loss.sum(dim=-1)
            candidate_idx = loss.argmax()

            if available_substitutes[candidate_idx] != tgt_word: # word is different
                change += 1

            if change:
                if change < num_perturbation:
                    final_words[top_index[0]] = available_substitutes[candidate_idx]
                else:
                    tmp = final_words[top_index[0]]
                    for item in available_substitutes:
                        if item != tgt_word:
                            final_words[top_index[0]] = item
                            final_adverse.append(' '.join(final_words))
                    final_words[top_index[0]] = tmp
                    change -= 1
    
        return final_adverse


    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, image_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i+batch_size], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_embed'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_embed'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))
        import_scores = import_scores.sum(dim=-1)
        
        import_scores_2 = criterion(masked_embeds.log_softmax(dim=-1), image_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))
        import_scores_2 = import_scores_2.sum(dim=-1)

        return import_scores * 1000 + import_scores_2
        # return import_scores


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
