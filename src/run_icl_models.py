import os
import click


@click.command()
@click.option('--model_name', type=str, default="gpt-3.5-turbo")
@click.option('--endpoint', type=str)
@click.option('--task', type=str)
@click.option('--k', type=str)
def main(model_name, endpoint, task, k):
    if task is None:
        tasks = [
            "detox-toxic", "detox-sentiment", "detox-hate-speech",
            "germeval-engaging", "germeval-factclaiming", "germeval-toxic",
            "x-stance-de"
        ]
    else:
        tasks = [task]


    if k is None:
        k = [0, 1, 2, 4, 8]
    else:
        k = [int(ele) for ele in k.split(",")]

    for k in k:
        for fold in [0]:
            if k == 0:
                seeds = [0]
            else:
                seeds = [0, 1, 2, 4, 5]

            for seed in seeds:
                for task in tasks:
                    os.system("python3 run_icl_model.py --template_indices 0,1,2,3 --seed " + str(seed) + " --endpoint " + endpoint + " --model_name " + model_name + " --fold " + str(fold) + " --bm25_retrieval False --setup it --task " + task + " --k " + str(k))


if __name__ == "__main__":
    main()