import os
import click


@click.command()
@click.option('--model_name', type=str, default="bert-base-uncased")
@click.option('--task', type=str)
def main(model_name, task):
    if task is None:
        tasks = [
            "detox-toxic", "detox-sentiment", "detox-hate-speech",
            "germeval-engaging", "germeval-factclaiming", "germeval-toxic",
            "x-stance-de"
        ]
    else:
        tasks = [task]

    for seed in [0,1,2,3,4,5,6,7,8,9]:
        for fold in [0]:
            for task in tasks:
                os.system("python3 run_fine_tuning_model.py --model_name " + model_name + " --fold " + str(fold) + " --seed " + str(seed) + " --setup it --pooling cls --task " + task)


if __name__ == "__main__":
    main()
