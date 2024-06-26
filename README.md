# The Lou Dataset - Exploring the Impact of Gender-Fair Language in German Text Classification

![example](example_entry.png)
*Figure 1, example entry of the `Lou` dataset with the `original` instance of the engaging detection task from the GermEval-2021 dataset and its six reformulations.*

This work explores the impact of gender-fair language for German text classification tasks. 
We provide in this repository:
* The `Lou` dataset including the gender-inclusive and gender-neutral reformulations according to six gender-fair strategies.
* Source code to run the experiments reported in our work.

If there are any issues or questions, we are happy to help you. Just open an issue or e-mail us. 



> **Abstract:** Gender-fair language, an evolving linguistic variation in German, fosters inclusion by addressing all genders or using neutral forms. However, there is a notable lack of resources to assess the impact of this language shift on language models (LMs) might not been trained on examples of this variation. Addressing this gap, we present Lou, the first dataset providing high-quality reformulations for German text classification covering seven tasks, like stance detection and toxicity classification. We evaluate 16 mono- and multi-lingual LMs and find substantial label flips, reduced prediction certainty, and significantly altered attention patterns. However, existing evaluations remain valid, as LM rankings are consistent across original and reformulated instances. Our study provides initial insights into the impact of gender-fair language on classification for German. However, these findings are likely transferable to other languages, as we found consistent patterns in multi-lingual and English LMs.
> 
Contact person: Andreas Waldis, andreas.waldis@live.com

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/




> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


## The Lou Dataset

The `Lou` dataset provides gender-fair reformulation of instances from seven German classification task.

### Tasks and Data
We include seven tasks (*sentiment analysis* and *stance-*, *fact-claiming-*, *engaging-*, *hate-speech-*, and *toxicity-detection*) from the [X-Stance](https://huggingface.co/datasets/ZurichNLP/x_stance), [GermEval-2021](https://germeval2021toxic.github.io/SharedTask/), and [DeTox](https://github.com/hdaSprachtechnologie/detox) datasets.
For `Lou`, you need access to the full version of DeTox.
Therefore, we excluded it from this public repository. 
However, we are happy to this part of `Lou` with you when you provide us the approval which you can request [here](https://github.com/hdaSprachtechnologie/detox)

### Reformulation Strategies
With `Lou`, we follow six strategies to reformulate a text with a masculine formulation (like *Politiker*) gender-fair. *Figure 1* shows one example.
* **Binary Gender Inclusion** (`Doppelnennung`) explicitly mentions the feminine and masculine but ignores others like agender.
For example, *Politiker* (politician.MASC.PL) is transformed into *Politikerinnen und Politiker* (*politician.FEM.PL and politician.MASC.PL*).
* **All Gender Inclusion** explicitly addresses every gender, including agender, non-binary, or demi-gender, using a gender gap character pronounced with a small pause.
We consider three commonly used strategies with different gender characters: `GenderStern` (*), `GenderDoppelpunkt` (:), and  `GenderGap` (_).
For example, *Politiker* (politician.MASC.PL) is turned into *Politiker*innen*, *Politiker:innen*, or *Politiker_innen* (*politician.FEM.MASC.NEUT.PL*).
* **Gender Neutralization** avoids naming a particular gender. For this strategy (`Neutral`), we use neutral terms like *ärztliche Fachperson* (*medical professional*).
* **Neosystem** (`De-e`) is a well-specified system that emerged from a significant [community-driven effort](https://geschlechtsneutral.net).
This strategy use a fourth gender, including new pronouns, articles, and suffixes to avoid naming a particular gender.
For example, *Politiker* (*politician.MASC.PL*) is changed to *Politikerne* (*politician.FEM.MASC.NEUT.PL*).


## Experiments
With the following steps, you can run the experiments of the paper on the `Lou` dataset.

This repository requires Python3.7 or higher; further requirements can be found in the requirements.txt.
Install them with the following command:

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Next, you need to setup the `.env`.
Either copy `.env_dev` (development) or `.env_prod` (production) to `.env` and set your OpenAI (`OPENAI_KEY`) key, if you would like to run the in-context learning (ICL) experiments with OpenAI models.

```
$ cp .env_dev .env #development
$ cp .env_prod .env #production
```

Finally, you need to log in with your wandb account for performance reporting.

```
$ wandb login
```



## Tasks

This work relies on the following different datasets:
*  Argument Quality (`arg-qua`), available [here](https://research.ibm.com/haifa/dept/vst/debating_data.shtml)
*  Argument Similarity (`arg-sim`), available [here](https://huggingface.co/datasets/UKPLab/UKP_ASPECT)
*  Argument Classification (`arg-cls`), available [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345)
*  Evidence Classification (`evi-cls`), available [here](https://research.ibm.com/haifa/dept/vst/debating_data.shtml)
*  Entailment (`entail`), RTE is available [here](https://huggingface.co/datasets/nyu-mll/glue), SCITAIL [here](https://huggingface.co/datasets/allenai/scitail), and HANS [here](https://huggingface.co/datasets/jhu-cogsci/hans)
*  Sentiment Analysis (`review`), available [here](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
*  Stance Detection (`stance`), SemEval2016Task6 is available [here](http://alt.qcri.org/semeval2016/task6/), EMERGENT [here](https://github.com/willferreira/mscproject), and IAC [here](https://nlds.soe.ucsc.edu/iac)
*  Multi-Lingual Stance Detection (`x-stance`), available [here](https://huggingface.co/datasets/ZurichNLP/x_stance)
*  Multi-Lingual Sentiment Analysis (`x-review`), available [here](https://zenodo.org/record/3251672/files/cls-acl10-unprocessed.tar.gz)

Some of these datasets requires accepting conditions, but we are happy to share our parsed version with you.
Send us these datasets as zip to proof that you have access.
Afterward, we share our splits and you can put them into the `tasks` folder.

## Running the experiments

Please use the following scripts to run the different experiments.
For all of them take the following general parameters:
* `task`, specific task to run, for example `arg-cls`
* `model_name`, specific model (huggingface tag) to run, for example `bert-base-uncased"`
* `fold`, specific fold to run, for example `0`
* `setup`, specific setup to run, depends on the task, either cross-topic (`ct`), in-topic (`it`), cross-domain (`cd`), in-domain (`id`), cross-language (`cl`), or in-language (`il`)


### Probing (LP) Experiments (`run_frozen.py`)
Mono- and multi-lingual linear probing experiments, additional parameters:

* `seed`, specific random seed
* `pooling`, pooling method, for example `mean`
* `batch_size`, batch size for training, for example `16`
* `learning_rate`, learning rate for training, for example `0.0005`


### Prompting (P) Experiments (`run_prompt.py` and `run_x_prompt.py`)
Mono- or multi-lingual prompting experiments, additional parameters:

* `seed`, specific random seed
* `batch_size`, batch size for inference, for example `16`
* `verbalizing_mode`, verbalizing mode, either `static` (label specific tokens) or `automatic` (50 optimized tokens per label)


### Fine-Tuning (FT) Experiments (`run_fine-tuning.py`)
Vanilla mono- and multi-lingual fine-tuning experiments, additional parameters:

* `seed`, specific random seed
* `pooling`, pooling method, for example `cls`
* `batch_size`, batch size for training, for example `16`
* `learning_rate`, learning rate for training, for example `0.0005`
* `epochs`, specific number of epochs to train, for example `5`
* `dropout_rate`, dropout rate during training, for example `0.1`

### Prompt-Based Fine-Tuning (P+FT) Experiments (`run_prompt_tuning.py` and `run_x_prompt_tuning.py`)
Prompt-based mono- and multi-lingual fine-tuning experiments, additional parameters:
* `seed`, specific random seed
* `pooling`, pooling method, for example `cls`
* `batch_size`, batch size for training, for example `16`
* `learning_rate`, learning rate for training, for example `0.0005`
* `epochs`, specific number of epochs to train, for example `5`
* `verbalizing_mode`, verbalizing mode, either `static` (label specific tokens) or `automatic` (50 optimized tokens per label)


### Parameter-Efficient Prompt-Based Fine-Tuning (P+FT+LoRA) Experiments (`run_peft_prompt_tuning.py`)
Vanilla mono- and multi-lingual fine-tuning experiments, additional parameters:

* `seed`, specific random seed
* `pooling`, pooling method, for example `cls`
* `batch_size`, batch size for training, for example `16`
* `learning_rate`, learning rate for training, for example `0.0005`
* `verbalizing_mode`, verbalizing mode, either `static` (label specific tokens) or `automatic` (50 optimized tokens per label)
* `peft_mode`, specific efficient method, for example `LORA`
* `lora_r`, if using LORA the specific `r`, for example `4`

### In-Context Learning Experiments (`run_icl.py`)
In-context learning experiments

* `seed`, specific random seed
* `k`, number of demonstration examples, for example `4`
* `bm25_retrieval`, whether to use BM25 to retrieve most similar examples, for example `True`


## Citation

```
@misc{waldis2024handle,
      title={How to Handle Different Types of Out-of-Distribution Scenarios in Computational Argumentation? A Comprehensive and Fine-Grained Field Study}, 
      author={Andreas Waldis and Yufang Hou and Iryna Gurevych},
      year={2024},
      eprint={2309.08316},
      archivePrefix={arXiv},
}
```
