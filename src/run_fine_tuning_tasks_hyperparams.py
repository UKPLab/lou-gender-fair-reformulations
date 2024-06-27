import os
import click


@click.command()
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--tasks', type=str)
@click.option('--seeds', type=str, default="0,1,2")
@click.option('--batch_sizes', type=str, default="8,16,32")
@click.option('--learning_rates', type=str, default="0.00005,0.00002,0.00001")
def main(model_name, tasks, seeds, batch_sizes, learning_rates):
    if tasks is None:
        tasks = [
            "detox-toxic", "detox-sentiment", "detox-hate-speech",
            "germeval-engaging", "germeval-factclaiming", "germeval-toxic",
            "x-stance-de"
        ]
    else:
        tasks = tasks.split(",")

    if seeds is None:
        seeds = seeds.split(",")

    if batch_sizes is None:
        batch_sizes = batch_sizes.split(",")

    if learning_rates is None:
        learning_rates = learning_rates.split(",")

    for seed in seeds:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for task in tasks:
                    os.system(f"python3 run_fine_tuning_task.py --learning_rate {learning_rate} --batch_size {batch_size} --model_name {model_name} --task {task} --seed {seed}")



if __name__ == "__main__":
    main()