#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import json
import argparse
import numpy as np
from tqdm import tqdm
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter
from transformers import BertTokenizer
from contextlib import nullcontext
from data_utils import Datasets
from data_utils import split, get_assets_org_id
from models.ConOA import ConOA
from models.MeOA import MeOA
import dist_utils


def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="1,0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="WOI-a", type=str, help="which dataset to use, options: WOI-a, WOI-b")
    parser.add_argument("-m", "--model", default="ConOA", type=str, help="which model to use, options: ConOA, MeOA")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    return args


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")
    args = get_cmd()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.distributed == True:
        dist_utils.init_distributed_mode(args)
    device = torch.device(args.device)

    dataset_name = args.dataset
    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[0]]
    else:
        conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    conf["device"] = device
    conf["distributed"] = args.distributed and dist_utils.is_dist_avail_and_initialized()
    conf["model"] = args.model
    conf["info"] = args.info

    #### Dataset #### 
    tokenizer = BertTokenizer.from_pretrained(conf["bert_path"])
    dataset = Datasets(conf, tokenizer)
    conf["num_orgs"] = dataset.num_orgs
    conf["num_assets"] = dataset.num_assets
    conf["queue_size"] = dataset.num_train

    print(conf)

    for lr, l2_reg, embedding_size, margin, num_asset_sample, num_head, aug_ratio, c_temp, momentum in product(conf['lrs'], conf['l2_regs'], conf["embedding_sizes"], conf["margins"], conf["num_asset_samples"], conf["num_heads"], conf["aug_ratios"], conf["c_temps"], conf["momentums"]):
        log_path = "./log/%s/%s" %(conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" %(conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" %(conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" %(conf["dataset"], conf["model"])
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)

        conf["lr"] = lr
        conf["l2_reg"] = l2_reg
        conf["embedding_size"] = embedding_size
        conf["margin"] = margin
        conf["num_asset_sample"] = num_asset_sample
        conf["num_head"] = num_head
        conf["aug_ratio"] = aug_ratio
        conf["c_temp"] = c_temp
        conf["momentum"] = momentum

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]
        settings += [conf["mode"], str(conf["batch_size_train"]), str(conf["hidden_size"]), str(embedding_size), str(lr), str(l2_reg)]
        if conf["model"] == "ConOA":
            settings += [str(num_asset_sample), str(num_head), str(aug_ratio), str(conf["queue_size"]), str(c_temp), str(momentum)]
        elif conf["model"] == "MeOA":
            settings += [str(conf["neg_num"]), str(margin)]
        setting = "_".join(settings)

        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting
        run = SummaryWriter(run_path)

        #### Model ####
        if conf["model"] == "ConOA":
            model = ConOA(conf).to(device)
        elif conf["model"] == "MeOA":
            model = MeOA(conf).to(device)
        else:
            raise ValueError("Unimplemented model %s" %(conf["model"]))
        model_without_ddp = model
        if conf["distributed"]:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        optimizer = optim.Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["l2_reg"])
        # optimizer = optim.AdamW(model.parameters(), lr=conf["lr"], weight_decay=conf["l2_reg"])

        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])

        best_metrics, best_perform, best_epoch = init_best_metrics(conf)
        train_org_idx = get_assets_org_id(conf["data_path"], "train").to(device)
        if conf["model"] == "ConOA":
            train_emb = torch.zeros(conf["queue_size"], embedding_size).to(device)
            train_clusters = torch.zeros(conf["queue_size"], dtype=torch.int).to(device)
        if not conf["distributed"] or (conf["distributed"] and dist_utils.is_main_process()):
            metrics = {}
            sim_metric = {}
            train_emb = cal_asset_emb(model_without_ddp, dataset.train_raw_data[1], tokenizer, conf)
            metrics["val"], sim_metric["val"] = test(model_without_ddp, train_emb, train_org_idx, dataset.val_loader, tokenizer, conf)
            metrics["test"], sim_metric["test"] = test(model_without_ddp, train_emb, train_org_idx, dataset.test_loader, tokenizer, conf)
            best_metrics, best_perform, best_epoch = log_metrics(conf, model_without_ddp, metrics, sim_metric, run, log_path, checkpoint_model_path, checkpoint_conf_path, 0, 0, best_metrics, best_perform, best_epoch)
        if conf["distributed"]:
            dist.barrier()
            if conf["model"] == "ConOA":
                dist.broadcast(train_emb, src=0)
                # dist.broadcast(train_clusters, src=0)
        if conf["model"] == "ConOA":
            model_without_ddp.init_queue(train_emb, train_org_idx, train_clusters)

        for epoch in range(1, conf["epochs"] + 1):
            if conf["distributed"]:
                dataset.train_sampler.set_epoch(epoch)
            
            epoch_anchor = (epoch - 1) * batch_cnt
            model.train(True)
            optimizer.zero_grad()
            pbar = tqdm(enumerate(dataset.train_loader), total=batch_cnt, desc='Training')

            for batch_i, (batch, batch_org_idx, batch_pos_idx) in pbar:
                batch_anchor = epoch_anchor + batch_i + 1
                # print(batch_anchor)

                model.train(True)
                #### Gradient Accumulation ####
                my_context = model.no_sync if conf["distributed"] and (batch_i+1) % conf["opt_interval"] != 0 else nullcontext
                # print(torch.cuda.memory_allocated())
                with my_context():
                    batch = batch.to(device)
                    batch_org_idx = batch_org_idx.to(device)
                    batch_pos_idx = batch_pos_idx.to(device)
                    # print('---batch---')
                    # print(torch.cuda.memory_allocated())
                    if conf["model"] == "ConOA":
                        # loss = model(batch, batch_org_idx, batch_pos_idx) / conf["opt_interval"]
                        # a_c_loss, a_o_c_loss = model(batch, batch_org_idx, batch_pos_idx)
                        # loss = (0.6*a_c_loss + 0.4*a_o_c_loss) / conf["opt_interval"]
                        a_c_loss, a_o_c_loss, o_c_loss= model(batch, batch_org_idx, batch_pos_idx)
                        loss = (0.6*a_c_loss + 0.2*a_o_c_loss + 0.2*o_c_loss) / conf["opt_interval"]
                    else:
                        loss = model(batch) / conf["opt_interval"]
                    # print('---loss_cal---')
                    # print(torch.cuda.memory_allocated())
                    
                    loss.backward()
                    # print('---loss_back---')
                    # print(torch.cuda.memory_allocated())
                
                if (batch_i+1) % conf["opt_interval"] == 0:
                    optimizer.step()
                    # print('---optimize---')
                    # print(torch.cuda.memory_allocated())
                    optimizer.zero_grad()
                    # print('---optimize_zero_grad---')
                    # print(torch.cuda.memory_allocated())

                if conf["distributed"]:
                    loss_scalar = dist_utils.reduce_mean(loss, dist.get_world_size()).item()
                    a_c_loss_scalar = dist_utils.reduce_mean(a_c_loss, dist.get_world_size()).item()
                    a_o_c_loss_scalar = dist_utils.reduce_mean(a_o_c_loss, dist.get_world_size()).item()
                    o_c_loss_scalar = dist_utils.reduce_mean(o_c_loss, dist.get_world_size()).item()
                else:
                    loss_scalar = loss.item()
                    a_c_loss_scalar = a_c_loss.item()
                    a_o_c_loss_scalar = a_o_c_loss.item()
                    o_c_loss_scalar = o_c_loss.item()
                run.add_scalar("loss", loss_scalar, batch_anchor)
                run.add_scalar("a_c_loss", a_c_loss_scalar, batch_anchor)
                run.add_scalar("a_o_c_loss", a_o_c_loss_scalar, batch_anchor)
                run.add_scalar("o_c_loss", o_c_loss_scalar, batch_anchor)

                # pbar.set_description("epoch: %d, loss: %.4f" %(epoch, loss_scalar))
                # pbar.set_description("epoch: %d, loss: %.4f, a_c_loss: %.4f, a_o_c_loss: %.4f" %(epoch, loss_scalar, a_c_loss_scalar, a_o_c_loss_scalar))
                pbar.set_description("epoch: %d, loss: %.4f, a_c_loss: %.4f, a_o_c_loss: %.4f, o_c_loss: %.4f" %(epoch, loss_scalar, a_c_loss_scalar, a_o_c_loss_scalar, o_c_loss_scalar))

                if (not conf["distributed"] or (conf["distributed"] and dist_utils.is_main_process())) and batch_anchor % test_interval_bs == 0:  
                    metrics = {}
                    sim_metric = {}
                    train_emb = cal_asset_emb(model_without_ddp, dataset.train_raw_data[1], tokenizer, conf)
                    # train_clusters = clustering(train_emb, train_org_idx, conf)
                    metrics["val"], sim_metric["val"] = test(model_without_ddp, train_emb, train_org_idx, dataset.val_loader, tokenizer, conf)
                    metrics["test"], sim_metric["test"] = test(model_without_ddp, train_emb, train_org_idx, dataset.test_loader, tokenizer, conf)
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model_without_ddp , metrics, sim_metric, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)   

                if conf["distributed"]:
                    dist.barrier()


@torch.no_grad()
def cal_asset_emb(model, train_assets_info, tokenizer, conf):
    device = conf["device"]
    train_emb = torch.tensor([]).to(device)
    train_text_list = [train_assets_info[key]['text'] for key in train_assets_info]
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(train_text_list)), desc='Getting candidates\' embedding'):
            train_sentences = split(tokenizer, train_text_list[i], conf['max_token_len']-2)
            if conf["model"] == "ConOA":
                embedding = model.encoder_k(tokenizer(train_sentences, max_length=conf['max_token_len'], truncation=True, padding='max_length', return_tensors="pt").to(device))
                train_emb = torch.cat((train_emb, torch.mean(embedding, keepdim=True, dim=0)), 0)
            else:
                embedding = model.embedding(tokenizer(train_sentences, max_length=conf['max_token_len'], truncation=True, padding='max_length', return_tensors="pt").to(device))
                train_emb = torch.cat((train_emb, torch.mean(embedding, dim=0)), 0)
        # train_emb---[train_num, emb_size]
    return train_emb


@torch.no_grad()
def test(model, train_emb, train_org_idx, dataloader, tokenizer, conf):
    tmp_metrics = {}
    for m in ["hit_rate_o", "ndcg_o", "hit_rate_a", "recall_a", "ndcg_a"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]
    for m in [ "mrr_o", "map_a"]:
        tmp_metrics[m] = [0, 0]

    device = conf["device"]
    sim_metric = torch.tensor([]).to(device)
    global_sim_metric = torch.tensor([]).to(device)
    model.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Testing')
    with torch.no_grad():
        for _, (test_asset_text, org_label, asset_label) in pbar:
            test_emb = torch.tensor([]).to(device)
            for text in test_asset_text:
                test_sentences = split(tokenizer, text, conf['max_token_len']-2)
                if len(test_sentences) > 40:
                    test_sentences = test_sentences[:40]
                if conf["model"] == "ConOA":
                    embedding = model.encoder_q(tokenizer(test_sentences, max_length=conf['max_token_len'], truncation=True, padding='max_length', return_tensors="pt").to(device))
                    test_emb = torch.cat((test_emb, torch.mean(embedding, keepdim=True, dim=0)), 0)
                else:
                    embedding = model.embedding(tokenizer(test_sentences, max_length=conf['max_token_len'], truncation=True, padding='max_length', return_tensors="pt").to(device))
                    test_emb = torch.cat((test_emb, torch.mean(embedding, dim=0)), 0)
            # test_emb---[batch_size, emb_size]
            
            # asset_scores---[batch_size, train_num]
            asset_scores = model.evaluate(test_emb, train_emb)
            assert asset_scores.shape[1] == train_emb.shape[0]

            mean_scores = scatter(asset_scores, train_org_idx, dim=-1, reduce='mean')
            assert mean_scores.size() == org_label.size()
            
            ### 1org
            train_org_emb = model.cal_org_emb(train_emb, train_org_idx)
            org_scores = model.evaluate(test_emb, train_org_emb)
            # if conf["model"] == "ConOA":
                ### norg
                # train_sample_org_emb, train_sample_org_idx = model.cal_org_emb(train_emb, train_org_idx)
                # org_scores,_ = scatter_max(model.evaluate(test_emb, train_sample_org_emb), train_sample_org_idx)
            assert org_scores.size() == org_label.size()
            
            tmp_metrics, tmp_sim_metric, tmp_global_sim_metric = get_metrics(tmp_metrics, asset_label.to(device), asset_scores, org_label.to(device), org_scores, mean_scores, conf["topk"])
            sim_metric = torch.cat((sim_metric, tmp_sim_metric), dim=0)
            global_sim_metric = torch.cat((global_sim_metric, tmp_global_sim_metric), dim=0)

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        if m not in ["mrr_o", "map_a"]:
            metrics[m] = {}
            for topk, res in topk_res.items():
                metrics[m][topk] = res[0] / res[1]
        else:
            metrics[m] = topk_res[0] / topk_res[1]

    sim_metric = torch.mean(sim_metric,dim=0).tolist()
    global_sim_metric = torch.mean(global_sim_metric, dim=0)
    sim_metric.append(global_sim_metric.item())
    # print(sim_metric)
    return metrics, sim_metric


def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key] = {}
        best_metrics[key]["hit_rate_o"] = {}
        best_metrics[key]["ndcg_o"] = {}
        best_metrics[key]["hit_rate_a"] = {}
        best_metrics[key]["recall_a"] = {}
        best_metrics[key]["ndcg_a"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    for key in best_metrics:
        best_metrics[key]["mrr_o"] = 0
        best_metrics[key]["map_a"] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}
    best_epoch = 0

    return best_metrics, best_perform, best_epoch


def get_metrics(metrics, asset_grd, asset_pred, org_grd, org_pred, mean_scores, topks):
    tmp = {"hit_rate_o": {}, "ndcg_o": {}, "hit_rate_a": {}, "recall_a": {}, "ndcg_a": {}, "mrr_o":[], "map_a": []}
    for topk in topks:
        # org_pred---[batch_size, org_num]
        # col_indice---[batch_size, top_k]
        _, col_indice = torch.topk(org_pred, topk)
        # row_indice---[batch_size, top_k]
        row_indice = torch.zeros_like(col_indice) + torch.arange(org_pred.shape[0], device=org_pred.device, dtype=torch.long).view(-1, 1)
        is_hit = org_grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["hit_rate_o"][topk] = get_recall(org_pred, org_grd, is_hit)
        tmp["ndcg_o"][topk] = get_ndcg(org_pred, org_grd, is_hit, topk)

        # asset_pred---[batch_size, train_num]
        # col_indice---[batch_size, top_k]
        _, col_indice = torch.topk(asset_pred, topk)
        # row_indice---[batch_size, top_k]
        row_indice = torch.zeros_like(col_indice) + torch.arange(asset_pred.shape[0], device=asset_pred.device, dtype=torch.long).view(-1, 1)
        is_hit = asset_grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["hit_rate_a"][topk] = get_hit_rate(asset_pred, asset_grd, is_hit)
        tmp["recall_a"][topk] = get_recall(asset_pred, asset_grd, is_hit)
        tmp["ndcg_a"][topk] = get_ndcg(asset_pred, asset_grd, is_hit, topk)

    tmp["mrr_o"] = get_mrr(org_pred, org_grd)
    tmp["map_a"] = get_map(asset_pred, asset_grd)

    for m, topk_res in tmp.items():
        if m not in ["mrr_o", "map_a"]:
            for topk, res in topk_res.items():
                for i, x in enumerate(res):
                    metrics[m][topk][i] += x
        else:
            for i, x in enumerate(topk_res):
                metrics[m][i] += x

    # sim_metric---[batch_size, 2]
    sim_metric = scatter(mean_scores, org_grd.long(), dim=-1, reduce='mean')
    tmp_global_sim_metric = torch.mean(asset_pred, dim=1)
    return metrics, sim_metric, tmp_global_sim_metric


def get_hit_rate(pred, grd, is_hit):
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt > 0).sum().item()

    return [nomina, denorm]


def get_recall(pred, grd, is_hit):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, device=device, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float).to(device)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


def get_mrr(pred, grd):
    _, sorted_indices = torch.sort(pred, descending=True, dim=-1)
    rank_reciprocal = 1.0 / (torch.arange(pred.size(1)) + 1).to(pred.device)

    num_pos = grd.sum(dim=1)
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (rank_reciprocal * torch.gather(grd, 1, sorted_indices)).sum().item()

    return [nomina, denorm]


def get_map(pred, grd):
    prediction_k = torch.zeros_like(pred, dtype=torch.float).to(pred.device)
    
    for k in range(pred.shape[1]):
        _, col_indice = torch.topk(pred, k+1)
        # row_indice---[batch_size, top_k]
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, k+1)
        prediction_k[:, k] = hit.sum(dim=1) / hit.shape[1]

    epsilon = 1e-8
    num_pos = grd.sum(dim=1)
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    _, sorted_indices = torch.sort(pred, descending=True, dim=-1)
    nomina = ((prediction_k * torch.gather(grd, 1, sorted_indices)).sum(dim=-1) / (num_pos + epsilon)).sum().item()

    return [nomina, denorm]


def log_metrics(conf, model, metrics, sim_metric, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    write_log(run, log_path, conf["topk"], batch_anchor, metrics, sim_metric)

    topk_ = conf["topk"][0]
    print("top%d as the final evaluation standard" %(topk_))
    
    if epoch > 0:
        log = open(log_path, "a")
        if metrics["val"]["hit_rate_o"][topk_] > best_metrics["val"]["hit_rate_o"][topk_] and metrics["val"]["ndcg_o"][topk_] > best_metrics["val"]["ndcg_o"][topk_]:
            torch.save(model.state_dict(), checkpoint_model_path)
            dump_conf = dict(conf)
            del dump_conf["device"]
            json.dump(dump_conf, open(checkpoint_conf_path, "w"))
            best_epoch = epoch
            curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for topk in conf['topk']:
                for key, res in best_metrics.items():
                    for metric in res:
                        if metric not in ["mrr_o", "map_a"]:
                            best_metrics[key][metric][topk] = metrics[key][metric][topk]
                best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: HR_o_T=%.5f, NDCG_o_T=%.5f, HR_a_T=%.5f, REC_a_T=%.5f, NDCG_a_T=%.5f" %(curr_time, best_epoch, topk, best_metrics["test"]["hit_rate_o"][topk], best_metrics["test"]["ndcg_o"][topk], best_metrics["test"]["hit_rate_a"][topk], best_metrics["test"]["recall_a"][topk], best_metrics["test"]["ndcg_a"][topk])
                best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: HR_o_V=%.5f, NDCG_o_V=%.5f, HR_a_V=%.5f, REC_a_V=%.5f, NDCG_a_V=%.5f" %(curr_time, best_epoch, topk, best_metrics["val"]["hit_rate_o"][topk], best_metrics["val"]["ndcg_o"][topk], best_metrics["val"]["hit_rate_a"][topk], best_metrics["val"]["recall_a"][topk], best_metrics["val"]["ndcg_a"][topk])
                print(best_perform["val"][topk])
                print(best_perform["test"][topk])
                log.write(best_perform["val"][topk] + "\n")
                log.write(best_perform["test"][topk] + "\n")
            
            for metric in ["mrr_o", "map_a"]:
                best_metrics["val"][metric]= metrics["val"][metric]
                best_metrics["test"][metric]= metrics["test"][metric]
            
            best_perform["test"][0] = "%s, Best in epoch %d: MRR_o_T=%.5f, MAP_a_T=%.5f" %(curr_time, best_epoch, best_metrics["test"]["mrr_o"], best_metrics["test"]["map_a"])
            best_perform["val"][0] = "%s, Best in epoch %d: MRR_o_V=%.5f, MAP_a_V=%.5f" %(curr_time, best_epoch, best_metrics["val"]["mrr_o"], best_metrics["val"]["map_a"])
            print(best_perform["val"][0])
            print(best_perform["test"][0])
            log.write(best_perform["val"][0] + "\n")
            log.write(best_perform["test"][0] + "\n")
        log.close()

    return best_metrics, best_perform, best_epoch


def write_log(run, log_path, topks, step, metrics, sim_metric):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for topk in topks:
        for m, val_score in val_scores.items():
            if m not in ["mrr_o", "map_a"]:
                test_score = test_scores[m]
                run.add_scalar("%s_%d/Val" %(m, topk), val_score[topk], step)
                run.add_scalar("%s_%d/Test" %(m, topk), test_score[topk], step)
        val_str = "%s, Top_%d, Val: hit_rate_o: %f, ndcg_o: %f, hit_rate_a: %f, recall_a: %f, ndcg_a: %f" %(curr_time, topk, val_scores["hit_rate_o"][topk], val_scores["ndcg_o"][topk], val_scores["hit_rate_a"][topk], val_scores["recall_a"][topk], val_scores["ndcg_a"][topk])
        test_str = "%s, Top_%d, Test: hit_rate_o: %f, ndcg_o: %f, hit_rate_a: %f, recall_a: %f, ndcg_a: %f" %(curr_time, topk, test_scores["hit_rate_o"][topk], test_scores["ndcg_o"][topk], test_scores["hit_rate_a"][topk], test_scores["recall_a"][topk], test_scores["ndcg_a"][topk])
        log = open(log_path, "a")
        log.write("%s\n" %(val_str))
        log.write("%s\n" %(test_str))
        log.close()
        print(val_str)
        print(test_str)

    run.add_scalar("mrr_o/Val", val_scores["mrr_o"], step)
    run.add_scalar("map_a/Val", val_scores["map_a"], step)
    run.add_scalar("mrr_o/Test", test_scores["mrr_o"], step)
    run.add_scalar("map_a/Test", test_scores["map_a"], step)

    val_str = "%s, Val: mrr_o: %f, map_a: %f" %(curr_time, val_scores["mrr_o"], val_scores["map_a"])
    test_str = "%s, Test: mrr_o: %f, map_a: %f" %(curr_time, test_scores["mrr_o"],  test_scores["map_a"])
    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()
    print(val_str)
    print(test_str)

    run.add_scalar("Global_cos_sim/Val", sim_metric["val"][2], step)
    run.add_scalar("Intra_cos_sim/Val", sim_metric["val"][1], step)
    run.add_scalar("Extra_cos_sim/Val", sim_metric["val"][0], step)
    run.add_scalar("Global_cos_sim/Test", sim_metric["test"][2], step)
    run.add_scalar("Intra_cos_sim/Test", sim_metric["test"][1], step)
    run.add_scalar("Extra_cos_sim/Test", sim_metric["test"][0], step)

    val_str = "%s, Val: global_sim: %f, intra_sim: %f, extra_sim: %f" %(curr_time, sim_metric["val"][2], sim_metric["val"][1], sim_metric["val"][0])
    test_str = "%s, Test: global_sim: %f, intra_sim: %f, extra_sim: %f" %(curr_time, sim_metric["test"][2], sim_metric["test"][1], sim_metric["test"][0])
    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()
    print(val_str)
    print(test_str)


if __name__ == "__main__":
    main()
