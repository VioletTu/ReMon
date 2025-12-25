# Website Owner Identification through Multi-level Contrastive Representation Learning

## Introduction
ReMon is a novel framework for website owner identification that formulates the task as webpage representation learning. It leverages LLM-based text rewriting to reduce noise and multi-level contrastive learning to capture website–owner relations. ReMon achieves state-of-the-art performance on real-world datasets, especially in challenging scenarios where WHOIS records are incomplete or webpages do not explicitly reveal owner names.

## Requirements
```
python3.10.13
cuda12.1
pytorch2.1.0
numpy 1.26.0
transformers 4.35.0
scipy 1.11.3
scikit-learn 1.3.2
```

## How to run the code
```
python -m torch.distributed.run --nproc_per_node=2 train.py -m ConOA -d WOI_a
python clustering.py -d WOI_a
```

If you find this code useful for your research or work, please cite the following paper:
```
@article{tu2025website,
  title={Website Owner Identification through Multi-level Contrastive Representation Learning},
  author={Tu, Cheng and Ma, Yunshan and Li, Yang and Zhang, Min and Hu, Miao and Shi, Fan and Wang, Xiang},
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={19},
  number={9},
  pages={1--39},
  year={2025},
  publisher={ACM New York, NY}
}
```
