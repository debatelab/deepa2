<p align="left">
    <a href="https://github.com/debatelab/deepa2/actions/workflows/run_pytest.yml">
        <img alt="unit tests" src="https://github.com/debatelab/deepa2-datasets/actions/workflows/run_pytest.yml/badge.svg?branch=main">
    </a>
    <a href="https://github.com/debatelab/deepa2/actions/workflows/code_quality_checks.yml">
        <img alt="code quality" src="https://github.com/debatelab/deepa2-datasets/actions/workflows/code_quality_checks.yml/badge.svg?branch=main">
    </a>
</p>

# Deep Argument Analysis (`deepa2`)</p>

This project provides `deepa2`, which

* 🥚 takes NLP data (e.g. NLI, argument mining) as ingredients;
* 🎂 bakes DeepA2 datatsets conforming to the [Deep Argument Analysis Framework](https://arxiv.org/abs/2110.01509);
* 🍰 serves DeepA2 data as text2text datasets suitable for training language models.

There's a public collection of 🎂 DeepA2 datatsets baked with `deepa2` at the [HF hub](https://huggingface.co/datasets/debatelab/deepa2).

The [Documentation](docs/) describes usage options and gives background info on the Deep Argument Analysis Framework.


## Quickstart

### Integrating `deepa2` into Your Training Pipeline

1. Install `deepa2` into your ML project's virtual environment, e.g.:

```bash
source my-projects-venv/bin/activate 
python --version  # should be ^3.8
python -m pip install deepa2
```

2. Add `deepa2` preprocessor to your training pipeline. Your training script may look like, for example:

```sh
#!/bin/bash

# configure and activate environment
...

# download deepa2 datasets and 
# prepare for text2text training
deepa2 serve \
    --path some-deepa2-dataset \    # <<< 🎂
    --export_format csv \
    --export_path t2t \             # >>> 🍰

# run default training script, 
# e.g., with 🤗 Transformers
python .../run_summarization.py \
    --train_file t2t/train.csv \    # <<< 🍰
    --text_column "text" \
    --summary_column "target" \
    --...

# clean-up
rm -r t2t
```

3. That's it.


### Create DeepA2 datasets with `deepa2` from existing NLP data

Install [poetry](https://python-poetry.org/docs/#installation). 

Clone the repository:
```bash
git clone https://github.com/debatelab/deepa2-datasets.git
```

Install this package from within the repo's root folder:
```bash
poetry install
```

Bake a DeepA2 dataset, e.g.:
```bash
poetry run deepa2 bake \\
  --name esnli \\                   # <<< 🥚
  --debug-size 100 \\
  --export-path ./data/processed    # >>> 🎂  
```

## Contribute a DeepA2Builder for another Dataset

We welcome contributions to this repository, especially scripts that port existing datasets to the DeepA2 Framework. Within this repo, a code module that transforms data into the DeepA2 format contains

1. a Builder class that describes how DeepA2 examples will be constructed and that implements the abstract `builder.Builder` interface (such as, e.g., `builder.entailmentbank_builder.EnBankBuilder`);
2. a DataLoader which provides a method for loading the raw data as a 🤗 Dataset object (such as, for example, `builder.entailmentbank_builder.EnBankLoader`) -- you may use `deepa2.DataLoader` as is in case the data is available in a way compatible with 🤗 Dataset;
3. dataclasses which describe the features of the raw data and the preprocessed data, and which extend the dummy classes `deepa2.RawExample` and `deepa2.PreprocessedExample`;
4. a collection of unit tests that check the concrete Builder's methods (such as, e.g., `tests/test_enbank.py`);
5. a documentation of the pipeline (as for example in `docs/esnli.md`).

Consider **suggesting** to collaboratively construct such a pipeline by opening a [new issue](https://github.com/debatelab/deepa2/issues/new?assignees=&labels=enhancement&template=new_dataset.md&title=%5BDATASET+NAME%5D).

## Citation

This repository builds on and extends the DeepA2 Framework originally presented in:

```bibtex
@article{betz2021deepa2,
      title={DeepA2: A Modular Framework for Deep Argument Analysis with Pretrained Neural Text2Text Language Models}, 
      author={Gregor Betz and Kyle Richardson},
      year={2021},
      eprint={2110.01509},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```