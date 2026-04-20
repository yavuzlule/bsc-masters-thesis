preprocess:
	python src/bsc_relish/pipeline.py --config configs/preprocess.yaml

train:
	python src/bsc_relish/train.py --config configs/logreg.yaml

train-bert:

pipe:
	python src/bsc_relish/pipeline.py --config configs/pipeline.yaml
