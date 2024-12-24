# GMAttack: Gray-Box Adversarial Attack for Multimodal Transformer-based Language Models
# Environment
Git clone our repository, creating a python environment and activate it via the following command: 
```bash
cd GMAttack
conda create -n gmattack python=3.9
conda activate gmattack
pip install -r ./requirements.txt
```
# Preparation
## Download
Download datasets and the corresponding json files. The specific file structure is given in the following section.
### Datasets
- [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/)
- [SNLI-VE](https://github.com/necla-ml/SNLI-VE)
- MSCOCO: [train2014](http://images.cocodataset.org/zips/train2014.zip), [val2014](http://images.cocodataset.org/zips/val2014.zip), [test2015](http://images.cocodataset.org/zips/test2015.zip)
- [Dataset json files](https://github.com/salesforce/ALBEF) for downstream tasks
### Checkpoints
- Finetuned checkpoint for [ALBEF](https://github.com/salesforce/ALBEF)
- Finetuned checkpoint for [TCL](https://github.com/uta-smile/TCL)
- Finetuned checkpoint for [BLIP](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth)
## File Structure
Here is the recommended file structure. You can customize the file structure and modify the corresponding directory information in the files under `/config`.
```bash
/
├── data/	# datasets & json files
│   ├── flickr30k-images/	# Flickr30k
│   │   ├── 36979.jpg
│   │   ├── 65567.jpg
│   │   └── ....
│   ├── mscoco/	# MSCOCO
│   │   ├── train2014
│   │   ├── val2014
│   │   └── test2015
│   ├── refcoco+/	# json files for VG task
│   │   ├── cocos.json
│   │   ├── dets.json
│   │   ├── instance.json
│   │   └── ref(unc).p
│   ├── ve_train.json	# json files for VE task
│   ├── ve_test.json
│   ├── ve_dev.json
│   ├── refcoco+_train.json	# json files for VG task
│   ├── refcoco+_test.json
│   ├── refcoco+_val.json
│   ├── flickr30k_train.json	# json files for VR task on Flickr30k dataset
│   ├── flickr30k_test.json
│   ├── flickr30k_val.json
│   ├── coco_train.json	# json files for VR task on MSCOCO dataset
│   ├── coco_test.json
│   ├── coco_val.json
│   ├── vqa_train.json	# json files for VQA task
│   ├── vqa_test.json
│   ├── vqa_val.json
│   └── answer_list.json
│
├── checkpoints/	# checkpoints of models for different downstream tasks
│   ├── ALBEF_ve.pth
│   ├── ALBEF_refcoco.pth
│   ├── ALBEF_flickr30k.pth
│   ├── ALBEF_mscoco.pth
│   ├── ALBEF_vqa.pth
│   ├── TCL_ve.pth
│   ├── TCL_flickr30k.pth
│   ├── TCL_mscoco.pth
│   └── BLIP_vqa.pth
│
└── GMAttack/	#our repository
```
## Evaluation
Run the program `/GMAttack/attack_MODEL_TASK.py` to evaluate the attack performance of :

```python
python attack_ALBEF_ve.py --gpu 0
```
