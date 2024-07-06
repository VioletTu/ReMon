#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from bs4 import BeautifulSoup
import dist_utils


def read_file(path):
    with open(path, "r",encoding='utf-8') as f:
        lines = f.readlines()
    return "".join(lines)


def get_assets_org_id(path, task):
    with open(os.path.join(path, '{}.json'.format(task)), 'r') as f:
        assets_data = json.load(f)
    with open(os.path.join(path, 'org_index.json'), 'r') as f:
        org2id = json.load(f)

    assets_org_id_list = []
    for org in assets_data.keys():
        for _ in assets_data[org].keys():
            assets_org_id_list.append(org2id[org]) # from:0
    
    return torch.LongTensor(assets_org_id_list)


def split(tokenizer, text, max_token_len=510):
    batch_sentences = []
    raw_token = tokenizer.encode(text, add_special_tokens=False)
    tmp = 1 if len(raw_token) % max_token_len > 0 else 0
    for i in range(len(raw_token) // max_token_len + tmp):
        if len(raw_token) > max_token_len*(i+1):
            batch_sentences.append(tokenizer.decode(raw_token[max_token_len*i:max_token_len*(i+1)]))
        else:
            batch_sentences.append(tokenizer.decode(raw_token[max_token_len*i:]))
    return batch_sentences


class TrainDataset(Dataset):
    def __init__(self, model, train_raw_data, tokenizer, num_orgs, num_train, org2id, asset2id, neg_num=1, max_token_len=512):
        self.model = model
        self.org_assets_list = train_raw_data[0]
        self.assets_info_list = train_raw_data[1]
        self.tokenizer = tokenizer
        self.num_orgs = num_orgs
        self.num_train = num_train
        self.org2id = org2id
        self.asset2id = asset2id
        self.neg_num = neg_num
        self.max_token_len = max_token_len


    def __getitem__(self, index):
        asset_text_list = []
        asset_id = list(self.assets_info_list.keys())[index]
        asset_text_list.append(self.assets_info_list[asset_id]['text'])
        
        org = self.assets_info_list[asset_id]['org']
        org_id = self.org2id[org]
        
        while True:
            asset_id_list = self.org_assets_list[org]
            i = np.random.randint(len(self.org_assets_list[org]))
            if asset_id_list[i] != asset_id:
                pos_id = asset_id_list[i]
                asset_text_list.append(self.assets_info_list[pos_id]['text'])
                pos_idx = self.asset2id[pos_id]
                break
        
        if self.model == "MeOA":
            neg_count = 0
            # while True:
            #     i = np.random.randint(self.num_train)
            #     neg_id = list(self.assets_info_list.keys())[i]
            #     if self.assets_info_list[neg_id]['org'] != org:
            #         asset_text_list.append(self.assets_info_list[neg_id]['text'])
            #         neg_count += 1
            #         if neg_count == self.neg_num:
            #             break
            
            while True:
                i = np.random.randint(self.num_orgs)
                org_n = list(self.org_assets_list.keys())[i]
                if org_n != org:
                    asset_id_list = self.org_assets_list[org_n]
                    j = np.random.randint(len(asset_id_list))
                    neg_id = asset_id_list[j]
                    asset_text_list.append(self.assets_info_list[neg_id]['text'])
                    neg_count += 1
                    if neg_count == self.neg_num:
                        break
                
        return self.tokenizer(asset_text_list, max_length=self.max_token_len, truncation=True, padding='max_length', return_tensors="pt"), org_id, pos_idx


    def __len__(self):
        # return self.num_orgs
        return self.num_train


class TestDataset(Dataset):
    def __init__(self, train_raw_data, test_raw_data, tokenizer, num_orgs, num_test, org2id, max_token_len=512):
        self.train_org_assets_list = train_raw_data[0]
        self.train_assets_info_list = train_raw_data[1]
        self.test_org_assets_list = test_raw_data[0]
        self.test_assets_info_list = test_raw_data[1]
        self.tokenizer = tokenizer
        self.num_orgs = num_orgs
        self.num_test = num_test
        self.org2id = org2id
        self.max_token_len = max_token_len


    def __getitem__(self, index):
        asset_id = list(self.test_assets_info_list.keys())[index]
        org = self.test_assets_info_list[asset_id]['org']
        org_id = self.org2id[org]
        org_label = torch.zeros(self.num_orgs)
        org_label[org_id] = 1
        # print(org_label)
        
        asset_label = []        
        for id in self.train_assets_info_list.keys():
            asset_label.append(1 if org == self.train_assets_info_list[id]['org'] else 0)
        
        return self.test_assets_info_list[asset_id]['text'], org_label, torch.tensor(asset_label)


    def __len__(self):
        return self.num_test


class Datasets():
    def __init__(self, conf, tokenizer):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']
        self.tokenizer = tokenizer
        self.max_token_len = conf['max_token_len'] - 2
        self.mode = conf['mode']
        self.model = conf['model']
        
        self.num_orgs, self.num_assets, self.num_train, self.num_valid, self.num_test = self.get_data_size()
        org2id = self.get_org_id()
        asset2id = self.get_asset_id()

        self.train_raw_data = self.get_data("train")
        self.valid_raw_data= self.get_data("valid")
        self.test_raw_data = self.get_data("test")

        self.train_data = TrainDataset(self.model, self.train_raw_data, tokenizer, self.num_orgs, self.num_train, org2id, asset2id, conf["neg_num"], conf['max_token_len'])
        self.val_data = TestDataset(self.train_raw_data, self.valid_raw_data, tokenizer, self.num_orgs, self.num_valid, org2id, conf['max_token_len'])
        self.test_data = TestDataset(self.train_raw_data, self.test_raw_data, tokenizer, self.num_orgs, self.num_test, org2id, conf['max_token_len'])

        if dist_utils.is_dist_avail_and_initialized():
            num_tasks = dist_utils.get_world_size()  # num of gpus
            global_rank = dist_utils.get_rank()
            # len(sampler): len(train_data) / len(num_tasks)
            self.train_sampler = DistributedSampler(self.train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            self.train_sampler = None
        # len(train_loader): len(train_data) / len(batch_size) or len(sampler) / len(batch_size)
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size_train, shuffle=(self.train_sampler is None), sampler=self.train_sampler, num_workers=10, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size_test, shuffle=False, pin_memory=True, num_workers=10)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size_test, shuffle=False, pin_memory=True, num_workers=10)


    def get_data_size(self):
        with open(os.path.join(self.path, 'data_size.txt'), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:5]


    def get_org_id(self):
        with open(os.path.join(self.path, 'org_index.json'), 'r') as f:
            return json.load(f)


    def get_asset_id(self):
        with open(os.path.join(self.path, 'asset_index.json'), 'r') as f:
            return json.load(f)


    def processing_block(self, html):
        soup = BeautifulSoup(' '.join(html.split('\n')), 'html.parser')
        text_website = soup.get_text(separator=' ')
        text_list = text_website.split()
        
        batch_sentences = []
        token_tmp = []
        for k in range(len(text_list)):
            if len(text_list[k]) > 0 and text_list[k].isspace() == False:
                raw_token = self.tokenizer.encode(text_list[k], add_special_tokens=False)
                if len(token_tmp) + len(raw_token) <= self.max_token_len:
                    if len(token_tmp) == 0:
                        token_tmp = raw_token
                    else:
                        token_tmp = token_tmp + raw_token
                else:
                    batch_sentences.append(self.tokenizer.decode(token_tmp))
                    tmp = 1 if len(raw_token) % self.max_token_len > 0 else 0
                    for i in range(len(raw_token) // self.max_token_len + tmp):
                        if len(raw_token) > self.max_token_len*(i+1):
                            batch_sentences.append(self.tokenizer.decode(raw_token[self.max_token_len*i:self.max_token_len*(i+1)]))
                        else:
                            token_tmp = raw_token[self.max_token_len*i:]
            if k == len(text_list)-1:
                batch_sentences.append(self.tokenizer.decode(token_tmp))
        return batch_sentences


    def processing(self, asset_id, html, task):
        if self.mode == "rewrite":
            text = ' '.join(html.split())
        else:
            soup = BeautifulSoup(' '.join(html.split('\n')), 'html.parser')
            text_website = soup.get_text(separator=' ')
            text = ' '.join(text_website.split())
        
        if task == 'train':
            batch_sentences = split(self.tokenizer, text, self.max_token_len)
            if len(batch_sentences) != 0:
                return asset_id + batch_sentences[np.random.randint(len(batch_sentences))]
            else:
                return asset_id
        else:
            return asset_id + text


    def get_data(self, task):
        # get the data from json file
        with open(os.path.join(self.path, '{}.json'.format(task)), 'r') as f:
            data = json.load(f)
        
        org_assets_list = {}
        assets_info_list = {}
        for org in data.keys():
            org_assets_list[org] = list(data[org].keys())
            for asset_id in data[org].keys():
                if self.mode == 'rewrite':
                    path = os.path.join(self.path + '_remove_rewrite_1500/results', asset_id + '_' + org)
                    assets_info_list[asset_id] = {'text' : self.processing(asset_id, read_file(path), task), 'org' : org}
                else:
                    assets_info_list[asset_id] = {'text' : self.processing(asset_id, data[org][asset_id].replace(org,''), task), 'org' : org}
        
        return org_assets_list, assets_info_list
