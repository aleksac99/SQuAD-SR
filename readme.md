# Neural Network for Extractive Question Answering finetuned on synthetically translated dataset to Serbian Language

This repository contains code that accompanies research for my thesis. It contains scripts for crafting synthetically translated SQuAD Dataset to Serbian language, notebook for finetuning an Extractive Question Answering models, and notebooks for dataset analysis, model evaluation and model error analysis.

## Abstract
When it comes to any Natural Language Processing task, there’s a huge gap between the research done about English and the research done about low-resource languages, including Serbian. This thesis tries to bridge this gap by tackling the Extractive Question Answering task in Serbian. To acquire necessary training data, it examines a method for synthetic dataset creation, based on Stanford Question Answering Dataset. Contributions of this work are the following: two synthetically crafted Question Answering datasets are obtained (both in Cyrillic and Latin script), their quality and usability is briefly analyzed, four finetuned models are obtained (two for each script), from which the best one is released, and finally the evaluation and brief error analysis of the model is performed. Acquired results are compared to existing similar research, English state-of-the-art and human baseline, and future improvements are proposed.

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

## Results
This work reports the following results:

| Model               | Exact Match | F1 Score |
|---------------------|-------------|----------|
| mBERT-SQuAD-sr-cyrl | 51.46       | 67.28    |
| XLM-R-SQuAD-sr-cyrl | 53.73       | 69.45    |
| mBERT-SQuAD-sr-lat  | 69.32       | 80.11    |
| XLM-R-SQuAD-sr-lat  | 71.04       | 81.62    |

## Future Work

In the future, the plan is to perform much more in-depth analysis of synthetic dataset, and modify synthesis pipeline to obtain dataset which has higher quality. The plan is to open-source that dataset, and finetune and evaluate models to hopefully achieve even better results than reported now.

## References
The project is inspired by Translate-Align-Retrieve method proposed [here](https://arxiv.org/pdf/1912.05200.pdf) and adapts some parts of the code and naming conventions from [here](https://github.com/ccasimiro88/TranslateAlignRetrieve). Translation model used is published as a part of [NLLB](https://ai.facebook.com/research/no-language-left-behind/) project. For finetuning, considered models are [mBERT](https://arxiv.org/abs/1810.04805) and [XLM-R](https://arxiv.org/abs/1911.02116).