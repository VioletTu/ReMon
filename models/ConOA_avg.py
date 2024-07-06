#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from transformers import BertModel
from math import sqrt


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
        # self.fc = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.embedding_size * 2),
        #     nn.Linear(self.embedding_size * 2, self.embedding_size)
        # )


    def forward(self, batch):
        cls_emb = self.bert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])[1]
        # print(cls_emb.size())
        # return cls_emb
        embedding = self.fc(cls_emb)
        return embedding


class ConOA(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.device = conf["device"]
        self.margin = conf["margin"]
        self.c_temp = conf["c_temp"]
        self.momentum = conf["momentum"]
        self.queue_size = conf["queue_size"]
        self.embedding_size = conf["embedding_size"]
        self.num_asset_sample = conf["num_asset_sample"]
        self.aug_type = conf["aug_type"]
        self.aug_ratio = conf["aug_ratio"]

        self.encoder_q = EmbModel(conf)
        self.encoder_k = EmbModel(conf)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.embedding_size, self.queue_size))
        # org_id
        self.register_buffer("idx_queue", torch.full((1, self.queue_size),-100))
        # org_cluster_id
        self.register_buffer("idx_clusters", torch.full((1, self.queue_size),-100))


    def data_augmentation(self, features):
        if self.aug_type == 'Noise':
            random_noise = torch.rand_like(features).to(self.device)
            features += torch.sign(features) * F.normalize(random_noise, dim=-1) * self.aug_ratio
        elif self.aug_type == 'Dropout':
            features = F.dropout(features, self.aug_ratio)
        return features


    def mean_assets_emb(self, assets_embedding, assets_org_idx, unique_org_idx):
        # assets_embedding---[num_assets, emb_size]
        # assets_org_idx---[num_assets]
        assets_mean_emb = torch.zeros(unique_org_idx.shape[0], self.embedding_size).to(self.device)
        for i in range(unique_org_idx.shape[0]):
            indices = (assets_org_idx == unique_org_idx[i]).nonzero().reshape(-1)
            assets_mean_emb[i,:] = torch.mean(assets_embedding[indices,:].reshape(-1,self.embedding_size), dim=0)
        return assets_mean_emb


    def cal_c_loss(self, pred, sim_targets):
        # pred---[batch_size, batch_size + queue_size]
        # sim_targets---[batch_size, batch_size + queue_size]
        c_loss = - torch.sum(F.log_softmax(pred / self.c_temp, dim=1) * sim_targets, dim=1).mean()
        
        return c_loss


    def cal_loss(self, anchors_embedding, anchors_embedding_m, assets_embedding_m, batch_org_idx):
        ### cross-asset
        # pred1---[batch_size, batch_size(1+negs)]
        pred1 = torch.mm(anchors_embedding, F.normalize(assets_embedding_m, dim=-1).permute(1,0))
        # print("---pred1")
        # print(pred1.size())
        # print(pred1)

        # queue---[emb_size, queue_size]
        queue_assets_embedding = self.queue.clone().detach()
        # pred2---[batch_size, queue_size]
        pred2 = torch.mm(anchors_embedding, F.normalize(queue_assets_embedding, dim=0))
        # print("---pred2")
        # print(pred2.size())
        # print(pred2)
        
        batch_org_idx = batch_org_idx.view(-1,1)
        queue_org_idx = self.idx_queue.clone().detach()
        idx_all = torch.cat([batch_org_idx.permute(1,0), queue_org_idx], dim=1)
        pos_idx = torch.eq(batch_org_idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        
        a_c_loss = self.cal_c_loss(torch.cat((pred1, pred2), dim=1), sim_targets)
        
        ### cal org embedding
        queue_assets_embedding = queue_assets_embedding.permute(1,0)
        queue_org_idx = queue_org_idx.squeeze()
        batch_anchor_org_emb = torch.zeros(anchors_embedding.shape[0], self.embedding_size).to(self.device)
        batch_pos_org_emb = torch.zeros(anchors_embedding.shape[0], self.embedding_size).to(self.device)
        for i in range(batch_org_idx.shape[0]):
            indices = (queue_org_idx == batch_org_idx[i]).nonzero().reshape(-1)
            batch_anchor_org_emb[i,:] = torch.mean(torch.cat((anchors_embedding_m, queue_assets_embedding[indices,:].reshape(-1,self.embedding_size)), dim=0), dim=0)
            batch_pos_org_emb[i,:] = torch.mean(torch.cat((assets_embedding_m, queue_assets_embedding[indices,:].reshape(-1,self.embedding_size)), dim=0), dim=0)
            # batch_pos_org_emb[i,:] = self.data_augmentation(batch_anchor_org_emb[i,:])
        
        # batch_anchor_org_emb---[batch_size, emb_size]
        batch_anchor_org_emb = F.normalize(batch_anchor_org_emb, dim=-1)
        # # batch_pos_org_emb---[batch_size, emb_size]
        batch_pos_org_emb = F.normalize(batch_pos_org_emb, dim=-1)
        
        ### asset-org
        # pred1---[batch_size, batch_size*2]
        pred1 = torch.mm(anchors_embedding, torch.cat([batch_anchor_org_emb, batch_pos_org_emb], dim=0).permute(1,0))
        
        unique_org_idx = queue_org_idx.unique(sorted=True)
        queue_assets_sample_emb = self.mean_assets_emb(queue_assets_embedding, queue_org_idx, unique_org_idx)
        queue_org_emb = F.normalize(queue_assets_sample_emb, dim=-1)
        # pred2---[batch_size, org_size_in_queue]
        pred2 = torch.mm(anchors_embedding, queue_org_emb.permute(1,0))
        idx_all = torch.cat([batch_org_idx.permute(1,0), batch_org_idx.permute(1,0), unique_org_idx.unsqueeze(0)], dim=1)
        pos_idx = torch.eq(batch_org_idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        
        a_o_c_loss = self.cal_c_loss(torch.cat((pred1, pred2), dim=1), sim_targets)
        
        ### org-org
        pred1 = torch.mm(batch_anchor_org_emb, batch_pos_org_emb.permute(1,0))
        pred2 = torch.mm(batch_anchor_org_emb, queue_org_emb.permute(1,0))
        idx_all = torch.cat([batch_org_idx.permute(1,0), unique_org_idx.unsqueeze(0)], dim=1)
        pos_idx = torch.eq(batch_org_idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        
        o_c_loss = self.cal_c_loss(torch.cat((pred1, pred2), dim=1), sim_targets)
        
        return a_c_loss, a_o_c_loss, o_c_loss


    def forward(self, batch, batch_org_idx, batch_pos_idx):
        # anchors---[batch_size, 1]
        anchors_embedding = F.normalize(self.encoder_q({key: values[:,0,:] for key, values in batch.items()}), dim=-1)
        # anchors_embedding---[batch_size, emb_size]
        # print(anchors_embedding.size())

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            anchors_embedding_m = self.encoder_k({key: values[:,0,:] for key, values in batch.items()})
            # assets---[batch_size, 1]
            assets_embedding_m = self.encoder_k({key: values[:,1,:] for key, values in batch.items()})
            # assets_embedding---[batch_size, emb_size]
            # print(assets_embedding.size())

        # a_c_loss, a_o_c_loss = self.cal_loss(anchors_embedding, anchors_embedding_m, assets_embedding_m, batch_org_idx)
        a_c_loss, a_o_c_loss, o_c_loss = self.cal_loss(anchors_embedding, anchors_embedding_m, assets_embedding_m, batch_org_idx)

        # dequeue and enqueue
        self._dequeue_and_enqueue(assets_embedding_m, batch_pos_idx)

        return a_c_loss, a_o_c_loss, o_c_loss


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
        
        scores = torch.mm(F.normalize(test_emb, dim=-1), F.normalize(train_emb, dim=-1).permute(1,0))
        # scores---[batch_size, train_num]
        # print(scores.size())
        return scores


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    @torch.no_grad()
    def init_queue(self, train_sample_emb, train_org_idx, train_clusters):
        """
        Initialization of queue
        """
        self.queue = train_sample_emb.permute(1,0)
        self.idx_queue = train_org_idx.view(1,-1)
        self.idx_clusters = train_clusters.view(1,-1)
    

    @torch.no_grad()
    def update_clusters(self, train_clusters):
        """
        Update the clusters for org embedding
        """
        self.idx_clusters = train_clusters.view(1,-1)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, asset_emb, batch_pos_idx):
        # gather keys before updating queue
        if self.conf["distributed"]:
            asset_embs = concat_all_gather(asset_emb)
            asset_pos_idxs = concat_all_gather(batch_pos_idx)
        else:
            asset_embs = asset_emb
            asset_pos_idxs = batch_pos_idx

        # replace the value
        self.queue[:, asset_pos_idxs] = asset_embs.permute(1,0)



# dist_utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output