import os
import yaml
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy import stats
from transformers import BertTokenizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from data_utils import Datasets
from data_utils import split, get_assets_org_id
from models.ConOA import ConOA
from models.MeOA import MeOA

def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0,1", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="WOI_a", type=str, help="which dataset to use, options: WOI_a, WOI_b")
    parser.add_argument("-e", "--embedding_model", default="ConOA", type=str, help="which embedding model to use, options: ConOA, MeOA")
    # parser.add_argument("-n", "--embedding_model_name", default="", type=str)
    # parser.add_argument("-c", "--clustering_model", default="hac", type=str, help="which clustering model to use, options: kmeans,dbscan,hac")
    args = parser.parse_args()

    return args


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")
    args = get_cmd()
    dataset_name = args.dataset
    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[-1]]
    else:
        conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    conf["model"] = args.embedding_model
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["model"] = device
    
    embedding_model_names = ['allrm_ES_6ac_2aocd_2oc_ave_dropout_rewrite_8_768_128_1e-05_1e-07_2_1_0.1_9775_0.04_0.99']
    
    for embedding_model_name in embedding_model_names:
        #### Dataset #### 
        conf["mode"] = 'html' if 'html' in embedding_model_name else 'rewrite'
        
        with open(os.path.join(conf["model_path"], args.embedding_model, "conf", embedding_model_name), 'r', encoding='utf-8') as f:
            model_conf = json.loads(f.read())
        model_conf["device"] = device
        tokenizer = BertTokenizer.from_pretrained(model_conf["bert_path"])
        dataset = Datasets(conf, tokenizer)
        
        #### Embedding model ####
        if args.embedding_model == "ConOA":
            emb_model = ConOA(model_conf).to(device)
        elif args.embedding_model == "MeOA":
            emb_model = MeOA(model_conf).to(device)
        else:
            raise ValueError("Unimplemented model %s" %(args.embedding_model))
        model_path = os.path.join(conf["model_path"], args.embedding_model, "model", embedding_model_name)
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(embedding_model_name))
            checkpoint = torch.load(model_path, map_location=device)
            emb_model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'".format(embedding_model_name))
        else:
            print("=> no checkpoint found at '{}'".format(embedding_model_name))
        
        #### Get embedding ####
        test_assets_info = dataset.test_raw_data[1]
        test_emb = torch.tensor([]).to(device)
        test_text_list = [test_assets_info[key]['text'] for key in test_assets_info]
        emb_model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(test_text_list)), desc='Getting test assets\' embedding'):
                sentences = split(tokenizer, test_text_list[i], conf['max_token_len']-2)
                if len(sentences) > 40:
                    sentences = sentences[:40]
                if args.embedding_model == "ConOA":
                    embedding = emb_model.encoder_q(tokenizer(sentences, max_length=conf['max_token_len'], truncation=True, padding='max_length', return_tensors="pt").to(device))
                    test_emb = torch.cat((test_emb, torch.mean(embedding, keepdim=True, dim=0)), 0)
                else:
                    embedding = emb_model.embedding(tokenizer(sentences, max_length=conf['max_token_len'], truncation=True, padding='max_length', return_tensors="pt").to(device))
                    test_emb = torch.cat((test_emb, torch.mean(embedding, dim=0)), 0)
            # test_emb---[test_num, emb_size]
        test_emb = test_emb.cpu().numpy()
        
        #### Get label ####
        true_label = get_assets_org_id(conf["data_path"], "test").numpy()
        
        f = open('./output/clustering/res-%s-%s.txt' % (dataset_name, embedding_model_name), 'a', encoding = 'utf-8')
        #### Clustering ####
        methods = ["kmeans", "dbscan", "hac"]
        for method in methods:
            if method == "kmeans":
                KMeans_model = KMeans(n_clusters=dataset.num_orgs, n_init='auto')
                result = KMeans_model.fit_predict(test_emb)
            elif method == "dbscan":
                DBSCAN_model = DBSCAN(eps=0.3, min_samples=4)
                result = DBSCAN_model.fit_predict(test_emb)
            elif method == "hac":
                HAC_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8)
                result = HAC_model.fit_predict(test_emb)
            else:
                raise ValueError("Unimplemented model %s" %(method))
            
            n_clusters_ = len(set(result)) - (1 if -1 in result else 0)
            pred_label = np.zeros_like(result)
            for i in range(n_clusters_):
                mask = (result == i)
                pred_label[mask] = stats.mode(true_label[mask])[0]
            accuracy = np.sum(pred_label == true_label) / len(true_label)
            print(f"{method} accuracy: {accuracy}")
            f.write("----%s----\n" %(dataset_name))
            f.write("%s accuracy: %f\n" %(method, accuracy))
            
            true_label_dict = {}
            for idx, true_lbl in enumerate(true_label):
                if true_lbl not in true_label_dict:
                    true_label_dict[true_lbl] = [idx]
                else:
                    true_label_dict[true_lbl].append(idx)
            
            pred_label_dict = {}
            for idx, pred_lbl in enumerate(pred_label):
                if pred_lbl not in pred_label_dict:
                    pred_label_dict[pred_lbl] = [idx]
                else:
                    pred_label_dict[pred_lbl].append(idx)
            
            # compute cluster-level F1
            # let's denote C(r) as clustering result and T(k) as partition (ground-truth)
            # construct r * k contingency table for clustering purpose
            r_k_table = []
            for v1 in pred_label_dict.values():
                k_list = []
                for v2 in true_label_dict.values():
                    N_ij = len(set(v1).intersection(v2))
                    k_list.append(N_ij)
                r_k_table.append(k_list)
            r_k_matrix = np.array(r_k_table)
            r_num = int(r_k_matrix.shape[0])
            # compute F1 for each row C_i
            sum_pre = 0.0
            sum_rec = 0.0
            sum_f1 = 0.0
            for row in range(0, r_num):
                row_sum = np.sum(r_k_matrix[row, :])
                if row_sum != 0:
                    max_col_index = np.argmax(r_k_matrix[row, :])
                    row_max_value = r_k_matrix[row, max_col_index]
                    prec = float(row_max_value) / row_sum
                    col_sum = np.sum(r_k_matrix[:, max_col_index])
                    rec = float(row_max_value) / col_sum
                    row_f1 = float(2 * prec * rec) / (prec + rec)
                    sum_pre += prec
                    sum_rec += rec
                    sum_f1 += row_f1
            average_pre = float(sum_pre) / r_num
            print("%s average_pre: %f" %(method, average_pre))
            f.write("%s average_pre: %f\n" %(method, average_pre))

            average_rec = float(sum_rec) / r_num
            print("%s average_rec: %f" %(method, average_rec))
            f.write("%s average_rec: %f\n" %(method, average_rec))
            
            average_f1 = float(sum_f1) / r_num
            print("%s average_f1: %f" %(method, average_f1))
            f.write("%s average_f1: %f\n" %(method, average_f1))
        f.close()


if __name__ == "__main__":
    main()
