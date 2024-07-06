# Website Owner Identification through Multi-level Contrastive Representation Learning

## Enviromment

```
python3.10.13
cuda12.1
pytorch2.1.0
numpy 1.26.0
transformers 4.35.0
scipy 1.11.3
scikit-learn 1.3.2
```

## Run the code
```
python -m torch.distributed.run --nproc_per_node=2 train.py -m ConOA -d WOI-a
python clustering.py -d WOI-a
```
