preprocess:
	python src/bsc_relish/pipeline.py --config configs/preprocess.yaml

train:
	python src/bsc_relish/train.py --config configs/logreg.yaml

train-bert:
	python src/bsc_relish/train_bert.py --config configs/bert_config.yaml

pipe:
	python src/bsc_relish/pipeline.py --config configs/pipeline.yaml

train-roberta:
	python src/bsc_relish/train_roberta.py --config configs/roberta_config.yaml

train-distilbert:
	python src/bsc_relish/train_distilbert.py --config configs/distilbert_config.yaml