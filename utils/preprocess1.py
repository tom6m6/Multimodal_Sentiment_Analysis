import json
import os
import re

import pandas as pd
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from PIL import Image
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torchvision import transforms, models

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from transformers import RobertaModel, RobertaTokenizer

from ClipDataset import ClipDataset
from RobertaDataset import RobertaDataset
from ResNetDataset import ResNetDataset

import clip

img_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_resnet_feats(dloc):
    feats = []

    model = models.__dict__['resnet50'](pretrained=True)
    model.eval().to(device)

    dataset = ResNetDataset(dloc, img_transform=img_transforms)
    dt_loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), num_workers=8)

    for i, batch in enumerate(dt_loader):
        if (i + 1) % 100 == 0 or (i + 1) == len(dt_loader):
            print("processing:\t %d / %d " % (i + 1, len(dt_loader)))
        
        img_inputs = batch[0].to(device)

        with torch.no_grad():
            outputs = model(img_inputs)

        feats.extend(outputs.view(-1, outputs.shape[1]).data.cpu().numpy().tolist())

    return feats



text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def text_transform(tweet):
    proc_tweet = text_processor.pre_process_doc(tweet)
    clean_tweet = [word.strip() for word in proc_tweet if not re.search(r"[^a-z0-9.,\s]+", word)]
    clean_tweet = [word for word in clean_tweet if word not in ['rt', 'http', 'https', 'htt']]

    return " ".join(clean_tweet)

def bert_embeddings(dloc, process_tweet=None):
    txt_feats = []
    fname = []

    tokenizer = RobertaTokenizer.from_pretrained('./tokenizer_roberta')
    model = RobertaModel.from_pretrained('./model_roberta', output_hidden_states=True)
    model.to(device).eval()

    dataset = RobertaDataset(dloc, txt_transform=process_tweet)
    dt_loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), num_workers=8)

    for i, batch in enumerate(dt_loader):
        if (i + 1) % 100 == 0 or (i + 1) == len(dt_loader):
            print("processing:\t %d / %d " % (i + 1, len(dt_loader)))

        txt_emb = batch[0]
        fname.extend(batch[1])
        txt_emb = tokenizer(txt_emb, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**txt_emb)
            text_features = outputs.pooler_output
            txt_feats.extend(text_features.cpu().numpy().tolist())

    return txt_feats, fname



def get_clip_feats(dloc, process_tweet=None):
    img_feats, txt_feats, tags = [], [], []

    model, img_preprocess = clip.load('ViT-B/16', device=device)
    # ViT-B/16 ViT-B/32
    model.eval()

    dataset = ClipDataset(dloc, img_transform=img_preprocess, txt_transform=process_tweet)
    dt_loader = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), num_workers=8)

    for i, batch in enumerate(dt_loader):
        if (i + 1) % 100 == 0 or (i + 1) == len(dt_loader):
            print("processing:\t %d / %d " % (i + 1, len(dt_loader)))

        img_emb, txt_emb, tag = batch[0].to(device), batch[1], batch[2]
        tags.append(str(tag[0]))

        txt_emb = clip.tokenize(txt_emb).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img_emb)
            text_features = model.encode_text(txt_emb)

            img_feats.extend(image_features.cpu().numpy().tolist())
            txt_feats.extend(text_features.cpu().numpy().tolist())

    return img_feats, txt_feats, tags

if __name__ == '__main__':
    trainloc = 'data/train1.txt'
    # ResNet
    image_feats = get_resnet_feats(trainloc)
    json.dump({'img_feats': image_feats}, open('saved/resnet_train.json', 'w'))

    # Roberta
    text_feats, fname = bert_embeddings(trainloc, text_transform)
    json.dump({'txt_feats': text_feats, 'fname': fname}, open('saved/roberta_train.json', 'w'))

    # Clip
    img_feats, text_feats, tags = get_clip_feats(trainloc, text_transform)
    json.dump({'img_feats': img_feats, 'txt_feats': text_feats, 'tags': tags}, open('saved/clip_train.json', 'w'))
    
