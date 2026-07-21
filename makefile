
# Makefile for BSC Relish project

# PREPROCESSING
preprocess:
	python /Users/yavuzlule/Desktop/bsc-relish/src/bsc_relish/preprocess/pipeline/run_pipeline.py --config configs/preprocess.yaml

# TRAINING
train-logreg:
	python src/bsc_relish/train_logreg.py --config configs/logreg.yaml

train-svm:
	python src/bsc_relish/train_svm.py --config configs/svm.yaml

train-xgb:
	python src/bsc_relish/train_xgb.py --config configs/xgb.yaml

train-distilbert:
	python src/bsc_relish/sequence_classification/train/train_transformer_model.py --config configs/distilbert.yaml

train-bert:
	python src/bsc_relish/sequence_classification/train/train_transformer_model.py --config configs/bert.yaml

train-roberta:
	python src/bsc_relish/sequence_classification/train/train_transformer_model.py --config configs/roberta.yaml

train-xlmroberta:
	python src/bsc_relish/sequence_classification/train/train_transformer_model.py --config configs/xlmroberta.yaml

train-mmbert:
	python src/bsc_relish/sequence_classification/train/train_transformer_model.py --config configs/mmbert.yaml

train-language-aware-distilbert:
	python src/bsc_relish/sequence_classification/train/train_language_aware_model.py --config configs/distilbert.yaml

train-language-aware-bert:
	python src/bsc_relish/sequence_classification/train/train_language_aware_model.py --config configs/bert.yaml

train-language-aware-roberta:
	python src/bsc_relish/sequence_classification/train/train_language_aware_model.py --config configs/roberta.yaml

train-language-aware-xlmroberta:
	python src/bsc_relish/sequence_classification/train/train_language_aware_model.py --config configs/xlmroberta.yaml

train-language-aware-mmbert:
	python src/bsc_relish/sequence_classification/train/train_language_aware_model.py --config configs/mmbert.yaml


# INFERENCE

infer-logreg:
	python src/bsc_relish/infer/infer_classical_model.py --config configs/logreg.yaml

infer-svm:
	python src/bsc_relish/infer/infer_classical_model.py --config configs/svm.yaml

infer-xgb:
	python src/bsc_relish/infer/infer_classical_model.py --config configs/xgb.yaml

infer-bert:
	python src/bsc_relish/infer/infer_bert.py --config configs/bert.yaml

infer-roberta:
	python src/bsc_relish/infer/infer_roberta.py --config configs/roberta.yaml

infer-distilbert:
	python src/bsc_relish/infer/infer_distilbert.py --config configs/distilbert.yaml

infer-xlmroberta:
	python src/bsc_relish/infer/infer_xlmroberta.py --config configs/xlmroberta.yaml

infer-mmbert:
	python src/bsc_relish/infer/infer_mmbert.py --config configs/mmbert.yaml

infer-language-aware-bert:
	python src/bsc_relish/sequence_classification/infer/infer_language_aware_model.py --config configs/bert.yaml

infer-language-aware-roberta:
	python src/bsc_relish/sequence_classification/infer/infer_language_aware_model.py --config configs/roberta.yaml

infer-language-aware-distilbert:
	python src/bsc_relish/sequence_classification/infer/infer_language_aware_model.py --config configs/distilbert.yaml

infer-language-aware-xlmroberta:
	python src/bsc_relish/sequence_classification/infer/infer_language_aware_model.py --config configs/xlmroberta.yaml

infer-language-aware-mmbert:
	python src/bsc_relish/sequence_classification/infer/infer_language_aware_model.py --config configs/mmbert.yaml

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