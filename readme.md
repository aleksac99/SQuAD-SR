# Synthetic Dataset Creation and Fine-Tuning of Transformer Models for Question Answering in Serbian

The paper is publiclly available at [arXiv](https://arxiv.org/abs/2404.08617).
The best performing model (BERTić finetuned on Latin version of the dataset) is publicly available at [HuggingFace](https://huggingface.co/aleksahet/BERTic-squad-sr-lat).
The synthetic dataset is available at [Kaggle](https://www.kaggle.com/datasets/aleksacvetanovic/squad-sr).

## Abstract
In this paper, we focus on generating a synthetic question answering (QA) dataset using an adapted Translate-Align-Retrieve method. Using this method, we created the largest Serbian QA dataset of more than 87K samples, which we name SQuAD-sr. To acknowledge the script duality in Serbian, we generated both Cyrillic and Latin versions of the dataset. We investigate the dataset quality and use it to fine-tune several pre-trained QA models. Best results were obtained by fine-tuning the BERTić model on our Latin SQuAD-sr dataset, achieving 73.91% Exact Match and 82.97% F1 score on the benchmark XQuAD dataset, which we translated into Serbian for the purpose of evaluation. The results show that our model exceeds zero-shot baselines, but fails to go beyond human performance. We note the advantage of using a monolingual pre-trained model over multilingual, as well as the performance increase gained by using Latin over Cyrillic. By performing additional analysis, we show that questions about numeric values or dates are more likely to be answered correctly than other types of questions. Finally, we conclude that SQuAD-sr is of sufficient quality for fine-tuning a Serbian QA model, in the absence of a manually crafted and annotated dataset. 

## Dependencies
Pipeline utilizes publicly available libraries. Python libraries are listed in `requirements.txt`. Notable projects and libraries are:
* [HuggingFace](https://github.com/huggingface): Ecosystem utilized for original dataset translation, as well as model finetuning and evaluation
* [Eflomal](https://github.com/robertostling/eflomal): Statistical Word Alignment tool
* [FastAlign](https://github.com/clab/fast_align): Statistical Word Alignment tool, utilized for symmetrization

To install Eflomal and FastAlign, please refer to instructions in their repositories.

## Usage
Source code is divided into 3 parts:
* Synthesis - contains a pipeline for synthetic translation of SQuAD Dataset
* Finetuning - contains Model finetuning notebook and loading script
* Analysis - contains notebooks in which dataset and model analysis, as well as model evaluation and error extraction is performed

Before running any part of the code, configuration parameters must be set. Configuration example is given in `config.json` file. Please use this file as a template to your own desired configuration parameters.

### Synthesis
Synthesis creation pipeline is divided into 4 scripts (excluding `utils`):

1. `translate.py` - for original SQuAD Dataset translation
2. `transliterate.py` - for transliteration from cyrillic to latin script (optional)
3. `align.py` - for calculating Word Alignments
4. `retrieve.py` - for final SQuAD-sr Dataset creation

To run full pipeline, please use shell script `run_all.sh` using the following command:
```
bash src/synthesis/run_all.sh <config_path> <fast_align_dir>
```
Also, scripts can be run one at a time, but don't forget to perform symmetrization between running `align.py` and `retrieve.py`. To find instructions on how to perform symmetrization, refer to FastAlign repository or to `run_all.sh` script.

### Finetuning
Model is finetuned in `finetune.ipynb`, which uses HuggingFace ecosystem to perform finetuning. To load synthetically crafted dataset, `loading_script.py` is used, which is adapted from HuggingFace examples.

### Analysis
In this section three notebooks are provided:
* `dataset_analysis.ipynb` - for synthetic dataset analysis
* `model_evaluation.ipynb` - for finetuned model evaluation and extraction of incorrect samples
* `model_error_analysis.ipynb` - which provides analysis of extracted samples

To use any of these notebooks, please change file paths in order to sucessfully load model and required data.
