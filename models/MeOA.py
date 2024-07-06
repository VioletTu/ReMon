#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class EmbModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.bert_path = conf["bert_path"]
        self.hidden_size = conf["hidden_size"]
        self.embedding_size = conf["embedding_size"]

        self.bert = BertModel.from_pretrained(self.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.embedding_size)


    def forward(self, batch):
        cls_emb = self.bert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])[1]
        output_emb = self.fc(cls_emb)
        return output_emb


class MeOA(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.device = conf["device"]
        self.bert_path = conf["bert_path"]
        self.hidden_size = conf["hidden_size"]
        self.embedding_size = conf["embedding_size"]
        self.margin = conf["margin"]
        self.c_temp = conf["c_temp"]

        self.bert = BertModel.from_pretrained(self.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.embedding_size)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.embedding_size * 2),
        #     nn.Linear(self.embedding_size * 2, self.embedding_size)
        # )


    def embedding(self, batch):
        # anchor---[batch_size, 1]
        # assets---[batch_size, 1+neg_num]
        # print(batch['input_ids'].size())
        # print(batch['attention_mask'].size())
        # print(batch['token_type_ids'].size())
        batch_size = batch['input_ids'].size()[0]
        seq_length = batch['input_ids'].size()[-1]
        cls_emb = self.bert(input_ids=batch['input_ids'].view(-1,seq_length), attention_mask=batch['attention_mask'].view(-1,seq_length), token_type_ids=batch['token_type_ids'].view(-1,seq_length))[1]
        # print(cls_emb.size())
        # return cls_emb.view(batch_size,-1,self.hidden_size)
        embedding = self.fc(cls_emb)
        return embedding.view(batch_size,-1,self.embedding_size)


    def cal_bpr_loss(self, pred):
        # pred---[batch_size, 1+neg_num]
        if pred.shape[1] > 2:
            negs = pred[:, 1:]
            pos = pred[:, 0].unsqueeze(1).expand_as(negs)
        else:
            negs = pred[:, 1].unsqueeze(1)
            pos = pred[:, 0].unsqueeze(1)

        # [batch_size]
        loss = - torch.log(torch.sigmoid(pos - negs))
        loss = torch.mean(loss)
        return loss


    def cal_margin_loss(self, pred):
        # pred---[batch_size, 1+neg_num]
        if pred.shape[1] > 2:
            negs = pred[:, 1:]
            pos = pred[:, 0].unsqueeze(1).expand_as(negs)
        else:
            negs = pred[:, 1].unsqueeze(1)
            pos = pred[:, 0].unsqueeze(1)
        
        loss = torch.mean(F.relu(- pos + negs + self.margin) + self.margin*0.001)
        
        return loss


    def cal_loss(self, anchor_embedding, assets_embedding):
        # pred---[batch_size, 1+neg_num]
        pred = F.cosine_similarity(anchor_embedding, assets_embedding, dim=2)
        # print("---pred")
        # print(pred.size())
        # print(pred)
        
        margin_loss = self.cal_margin_loss(pred)
        
        # bpr_loss = self.cal_bpr_loss(pred)
        
        # batch_size = anchor_embedding.size()[0]
        # ass_len = assets_embedding.size()[1]
        # label_list = []
        # for bs in range(batch_size):
        #     label_list.append(1)
        #     for i in range(ass_len-1):
        #         label_list.append(-1)
        # cos_emb_loss = F.cosine_embedding_loss(anchor_embedding.squeeze(1).repeat(1,ass_len), assets_embedding.view(batch_size,-1), torch.Tensor(label_list).to(self.device), margin=0.25, reduction='mean')
        # loss = F.cross_entropy(pred, torch.Tensor(label_list).view(batch_size,-1).to(self.device))

        return margin_loss


    def forward(self, batch):
        embedding = self.embedding(batch)
        # anchor_embedding---[batch_size, 1, emb_size]
        # assets_embedding---[batch_size, 1+neg_num, emb_size]
        anchor_embedding = embedding[:,0,:].unsqueeze(1)
        assets_embedding = embedding[:,1:,:]
        # print("---anchor")
        # print(anchor_embedding)
        # print("---asset")
        # print(assets_embedding.size())
        # print(assets_embedding)
        margin_loss = self.cal_loss(anchor_embedding, assets_embedding)

        return margin_loss

    @torch.no_grad()
    def mean_assets_emb(self, assets_embedding, assets_org_idx, unique_org_idx):
        # assets_embedding---[num_assets, emb_size]
        # assets_org_idx---[num_assets]
        assets_mean_emb = torch.zeros(unique_org_idx.shape[0], self.embedding_size).to(self.device)
        for i in range(unique_org_idx.shape[0]):
            indices = (assets_org_idx == unique_org_idx[i]).nonzero().reshape(-1)
            assets_mean_emb[i,:] = torch.mean(assets_embedding[indices,:].reshape(-1,self.embedding_size), dim=0)
        return assets_mean_emb

    @torch.no_grad()
    def cal_org_emb(self, train_emb, train_org_idx):
        # train_emb---[train_num, emb_size]
        unique_org_idx = train_org_idx.unique(sorted=True)
        # org_embedding---[org_num, emb_size]
        org_embedding = self.mean_assets_emb(train_emb, train_org_idx, unique_org_idx)
        # print(org_embedding.size())
        return org_embedding

    @torch.no_grad()
    def evaluate(self, test_emb, train_emb):
        # test_emb---[batch_size, emb_size]
        # train_emb---[train_num, emb_size]
        
        scores = F.cosine_similarity(test_emb.unsqueeze(1), train_emb.unsqueeze(0), dim=2)
        # scores---[batch_size, train_num]
        # print(scores.size())
        return scores
