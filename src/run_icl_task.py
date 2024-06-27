import glob
import os
import uuid

import click
import pandas
import tiktoken
import wandb
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer

from utils.composition import LABEL_MAPPING, compose_chat_instructions
from utils.prompting import get_icl_preds
from utils.training import check_run_done, get_metrics


def truncate_sentence(sentence, truncation_length, tokenizer):
    if sentence == None:
        sentence = "None"
    if "Encoding" not in str(tokenizer):
        tokens = tokenizer.encode(text=sentence, max_length=truncation_length, truncation=True, add_special_tokens=False)
    else:
        tokens = tokenizer.encode(text=sentence)[:truncation_length]
    return tokenizer.decode(tokens)

@click.command()
@click.option('--task', type=str, default="germeval-toxic")
@click.option('--model_name', type=str, default="gpt-3.5-turbo")
@click.option('--k', type=int, default=0)
@click.option('--seed', type=int, default=0)
@click.option('--template_indices', type=str, default="0")
@click.option('--endpoint', type=str)
def main(task, model_name, k, seed, template_indices, endpoint):
    load_dotenv()
    training = "INSTRUCTION_CHAT"
    api_key = os.getenv('OPENAI_KEY')

    if endpoint:
        client = OpenAI(
            api_key="empty",
            base_url=endpoint,
        )
    else:
        client = OpenAI(
            api_key=api_key,
        )


    train_samples = pandas.read_json("../tasks/" + task + "/train.jsonl", lines=True).sort_index()

    template_indices = [int(i) for i in template_indices.split(",")]

    for i in template_indices:

        for file in glob.glob("../tasks/" + task + "/test_*.jsonl"):
            subset = file.replace("../tasks/" + task + "/test_", "").replace(".jsonl", "")
            test_samples = pandas.read_json(file, lines=True).sort_index()

            def shorten_text(samples):
                if endpoint:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                else:
                    tokenizer = tiktoken.get_encoding("cl100k_base")

                samples["text"] = samples["text"].apply(lambda sentence: truncate_sentence(sentence, 300, tokenizer))
                return samples

            if "review" in task or "sentiment" in task or "germeval" in task:
                train_samples = shorten_text(train_samples)
                test_samples = shorten_text(test_samples)

            hyperparameter = {
                "model_name": model_name,
                "training": training,
                "k": k,
                "seed": seed,
                "template": i,
                "strategy": subset
            }

            is_run_done = check_run_done(task, hyperparameter)

            if not is_run_done:


                token_label_mapping = dict([
                    (token, label)
                    for label, token in LABEL_MAPPING[task][i].items()
                ])

                test_instructions = compose_chat_instructions(train_samples, test_samples, task, seed, k, i)

                test_predictions, test_answers, test_pred_tokens, test_prediction_categories = get_icl_preds(test_instructions, token_label_mapping, model_name=model_name, client=client)

                test_samples["pred"] = test_predictions
                test_samples["pred_token"] = test_pred_tokens
                test_samples["response_text"] = test_answers
                test_samples["response_category"] = test_prediction_categories

                test_samples_table = wandb.Table(dataframe=test_samples)

                run_id = str(uuid.uuid4())

                wandb.init(
                    project=task,
                    id=run_id,
                    config={
                        k: str(v)
                        for k, v in hyperparameter.items()
                    },
                    tags=[training]
                )

                wandb.log({
                    "test_predictions": test_samples_table
                })

                test_f1, test_acc = get_metrics(test_samples["label"], test_predictions)

                metrics = {
                    "test/f1-macro": test_f1,
                    "test/accuracy": test_acc,
                }

                wandb.log(metrics)

                wandb.config["template"] = str(i)
                wandb.config.update(
                    {
                        k: str(v)
                        for k, v in hyperparameter.items()
                    },
                    allow_val_change=True
                )

                wandb.config["status"] = "done"
                wandb.join()

    else:
        print("Run already done")



if __name__ == "__main__":
    main()