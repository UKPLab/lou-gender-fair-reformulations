import os
import click


@click.command()
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--tasks', type=str)
@click.option('--seeds', type=str, default="0,1,2,3,4,5,6,7,8,9")
def main(model_name, tasks, seeds):
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

    for seed in seeds:
        for task in tasks:
            os.system(f"python3 run_fine_tuning_task.py --model_name {model_name} --seed {seed} --task {task}")


if __name__ == "__main__":
    main()
