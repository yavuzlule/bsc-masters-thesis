
# Makefile for BSC Relish project

# PIPELINE
pipe:
	python src/bsc_relish/pipeline.py --config configs/pipeline.yaml




# PREPROCESSING
preprocess:
	python /Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/preprocess/pipeline/run_pipeline.py --config configs/preprocess.yaml




# TRAINING
train:
	python src/bsc_relish/train_logreg.py --config configs/logreg.yaml

train-bert:
	python src/bsc_relish/train_bert.py --config configs/bert_config.yaml

train-roberta:
	python src/bsc_relish/train_roberta.py --config configs/roberta_config.yaml

train-distilbert:
	python src/bsc_relish/train_distilbert.py --config configs/distilbert_config.yaml




# VISUALIZATION
visualize:
	python src/bsc_relish/visualize_report.py --run_dir /Users/yavuzlule/Desktop/bsc-relish/results/bert-base-uncased/2026-04-30_14-20-28

# INFERENCE
infer-bert:
	python src/bsc_relish/infer_bert.py

infer-roberta:
	python src/bsc_relish/infer_roberta.py

infer-distilbert:
	python src/bsc_relish/infer_distilbert.py

infer-xlmroberta:
	python src/bsc_relish/sequence_classification/infer/infer_xlmroberta.py
# TRANSLATION

translate-folder:

	python src/bsc_relish/translate/translate_llm.py \
		--input_dir /Users/yavuzlule/Desktop/bsc-relish/notebooks/chunked_recipes_2k/de \
		--langs de \
		--outdir /Users/yavuzlule/Desktop/bsc-relish/notebooks/chunked_recipes_2k/de/

translate-df:
	python src/bsc_relish/translate/translate_df_with_llm.py \
		--input_folder /Users/yavuzlule/Desktop/bsc-relish/data/corpus/20260529_163813 \
		--langs es de it nl fr \
		--output_folder /Users/yavuzlule/Desktop/bsc-relish/data/corpus/20260529_160341 \


# MoE
moe:
	python src/bsc_relish/MoE/main.py --config configs/moe.yaml


gepeto:
	python src/bsc_relish/agentic/main_gepeto.py --num_recipes 10 --model_name qwen2.5

agentic:
	python src/bsc_relish/agentic/main.py --num_recipes 10 --model_name phi4-mini-reasoning:3.8b

archive:
	python src/bsc_relish/MoE/internet_archive_moe/main.py --config configs/moe.yaml