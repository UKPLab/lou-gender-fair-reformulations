import glob
import os
import uuid

import click
import pandas
import torch
import wandb
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from hyperparams import HYPERPARAMS
from models.classification import CLSFineTuningClassification
from utils.composition import compose_samples
from utils.dataset import FineTuningDataset
from utils.training import check_run_done


def truncate_sentence(sentence, truncation_length, tokenizer):
    if sentence == None:
        sentence = "None"
    tokens = tokenizer.encode(text=sentence, max_length=truncation_length, truncation=True, add_special_tokens=False)
    return tokenizer.decode(tokens)

@click.command()
@click.option('--task', type=str, default="x-stance-fr")
@click.option('--model_name', type=str, default="deepset/gbert-base")
@click.option('--fold', type=int, default=0)
@click.option('--setup', type=str, default="it")
@click.option('--seed', type=int, default=20)
@click.option('--batch_size', type=int, default=16)
@click.option('--epochs', type=int, default=5)
@click.option('--dropout_rate', type=float, default=0.1)
@click.option('--learning_rate', type=float, default=0.00002)
def main(task, model_name, fold, setup, seed, batch_size, epochs, dropout_rate, learning_rate):

    if task in HYPERPARAMS and model_name in HYPERPARAMS[task]:
        learning_rate = HYPERPARAMS[task][model_name]["learning_rate"]
        batch_size = HYPERPARAMS[task][model_name]["batch_size"]

    load_dotenv()
    task_id = task + "-" + setup + "-fold-" + str(fold)
    gpu = "cuda" if os.getenv('MODE') == "prod" else "cpu"
    training = "FINE_TUNING"

    train_samples = pandas.read_json("../tasks/" + task_id + "/train.jsonl", lines=True).sort_index()
    dev_samples = pandas.read_json("../tasks/" + task_id + "/dev.jsonl", lines=True).sort_index()
    other_test_samples = {
        file.replace("../tasks/" + task_id + "/test_", "").replace(".jsonl", ""): pandas.read_json(file, lines=True).sort_index()
        for file in glob.glob("../tasks/" + task_id + "/test_*.jsonl")
    }

    if "text" in dev_samples.columns:
        dev_samples = dev_samples.sort_values('text',key=lambda x:x.str.len())
        other_test_samples = {
            test_set:samples.sort_values('text',key=lambda x:x.str.len())
            for test_set, samples in other_test_samples.items()
        }

    num_classes = len(train_samples["label"].unique())

    hyperparameter = {
        "model_name": model_name,
        "fold": fold,
        "setup": setup,
        "training": training,
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "seed": seed,
    }

    hyperparameter["warmup_steps"] = int(train_samples.shape[0] * epochs / batch_size * 0.1)
    hyperparameter["training_steps"] = int(train_samples.shape[0] * epochs / batch_size)

    is_run_done = check_run_done(task, hyperparameter)

    if not is_run_done:

        model = CLSFineTuningClassification(hyperparameter=hyperparameter, num_classes=num_classes)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def shorten_text(samples):
            samples["text"] = samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))
            return samples

        if "sentiment" in task or "germeval" in task:
            train_samples = shorten_text(train_samples)
            dev_samples = shorten_text(dev_samples)
            other_test_samples = {
                test_set:shorten_text(samples)
                for test_set, samples in other_test_samples.items()
            }

        def tokenize_function(samples):
            composed_samples = compose_samples(samples, task=task, sep_token=tokenizer.sep_token)
            samples["input_ids"] = [
                tokenizer.encode(composed_sample, truncation=True)
                for composed_sample in composed_samples
            ]
            return samples

        train_samples = tokenize_function(train_samples)
        dev_samples = tokenize_function(dev_samples)
        other_test_samples = {
            test_set:tokenize_function(samples)
            for test_set, samples in other_test_samples.items()
        }

        train_dataset = FineTuningDataset(train_samples)
        dev_dataset = FineTuningDataset(dev_samples)

        other_test_datasets = {
            test_set:FineTuningDataset(samples)
            for test_set, samples in other_test_samples.items()
        }

        run_id = str(uuid.uuid4())

        wandb_logger = WandbLogger(project=task, id=run_id)


        if "germeval" in task:
            batch_size = int(batch_size/2)
            accumulate_grad_batches = 2
        else:
            accumulate_grad_batches = 1

        trainer = Trainer(
            max_epochs=epochs, gradient_clip_val=1.0, logger=wandb_logger, accelerator=gpu, num_sanity_val_steps=0, accumulate_grad_batches=accumulate_grad_batches,
            callbacks=[RichProgressBar(), ModelCheckpoint(monitor="eval/f1-macro",  mode="max", dirpath="./" + run_id + "-checkpoints")]
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=512, padding="longest")

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

        other_test_dataloader = {
            test_set:DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
            for test_set, dataset in other_test_datasets.items()
        }

        trainer.fit(model=model, train_dataloaders=[train_dataloader], val_dataloaders=[dev_dataloader])

        other_test_predictions = {
            test_set: trainer.predict(ckpt_path="best", dataloaders=[dataloader], return_predictions=True)
            for test_set, dataloader in other_test_dataloader.items()
        }


        other_test_predictions = {
            test_set: torch.concat(predictions).numpy()
            for test_set, predictions in other_test_predictions.items()
        }

        def add_predictions(samples, predictions):
            samples["pred"] = predictions

            if "input_ids" in samples.columns:
                del samples["input_ids"]

            return samples

        other_test_samples = {
            test_set: add_predictions(other_test_samples[test_set], predictions)
            for test_set, predictions in other_test_predictions.items()
        }

        wandb.config["strategy"] = "full"
        wandb.config["status"] = "done"
        wandb.config.update(
            {
                k: str(v)
                for k, v in hyperparameter.items()
            },
            allow_val_change=True
        )
        wandb.join()

        for test_set, samples in other_test_samples.items():
            run_id = str(uuid.uuid4())
            wandb.init(id=run_id, project=task)

            samples_table = wandb.Table(dataframe=samples)

            wandb.log({
                test_set + "_test_predictions": samples_table
            })

            if num_classes > 2:
                wandb.log({
                    "test/f1-macro": f1_score(samples["label"], samples["pred"], average="macro"),
                    "test/accuracy": accuracy_score(samples["label"], samples["pred"]),
                })
            else:
                wandb.log({
                    "test/f1-macro": f1_score(samples["label"], samples["pred"]),
                    "test/accuracy": accuracy_score(samples["label"], samples["pred"]),
                })

            wandb.config["strategy"] = test_set
            wandb.config["status"] = "done"
            wandb.config.update(
                {
                    k: str(v)
                    for k, v in hyperparameter.items()
                },
                allow_val_change=True
            )
            wandb.join()

        os.system("rm -rf ./" + run_id + "-checkpoints")

    else:
        print("Run already done")



if __name__ == "__main__":
    main()