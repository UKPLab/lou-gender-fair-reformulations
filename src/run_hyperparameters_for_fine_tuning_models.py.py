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

    for seed in [0,1,2]:
        for fold in [0]:
            for batch_size in [8, 16, 32]:
                for learning_rate in [0.00005,0.00002,0.00001]:
                    for task in tasks:
                        os.system(f"python3 run_fine-tuning_multi_test.py --learning_rate {learning_rate} --batch_size " + str(batch_size) + " --model_name " + model_name + " --fold " + str(fold) + " --seed " + str(seed) + " --setup it --pooling cls --task " + task)



if __name__ == "__main__":
    main()