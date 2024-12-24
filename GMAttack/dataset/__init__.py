import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from dataset.caption_dataset import pair_dataset
from dataset.ve_dataset import ve_dataset, my_ve_dataset
from dataset.grounding_dataset import grounding_dataset
from dataset.vqa_dataset import vqa_dataset, vqa_dataset_albef
from dataset.randaugment import RandomAugment
from PIL import Image


def create_dataset(dataset, config, min_scale=0.5):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
         
    if dataset=='vqa': 
        transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
        transform_test = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])  
        train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'],
                                    train_files = config['train_files'], split='train') 
        test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], split='test')
        
    elif dataset=='vqa-albef': 
        train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])
        train_dataset = vqa_dataset_albef(config['train_file'], train_transform, config['vqa_root'], split='train') 
        test_dataset = vqa_dataset_albef(config['test_file'], test_transform, config['vqa_root'], split='test', answer_list=config['answer_list'])       
    
    return train_dataset, test_dataset
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    