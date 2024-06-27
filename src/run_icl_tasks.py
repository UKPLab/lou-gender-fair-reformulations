import os
import click


@click.command()
@click.option('--model_name', type=str, default="gpt-3.5-turbo")
@click.option('--endpoint', type=str)
@click.option('--tasks', type=str)
@click.option('--ks', type=str, default="0,1,2,4,8")
@click.option('--seeds', type=str, default="0,1,2,3,4")
@click.option('--template_indices', type=str, default="0,1,2,3,4")
def main(model_name, endpoint, tasks, ks, seeds, template_indices):
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


    if ks is None:
        ks = [0, 1, 2, 4, 8]
    else:
        ks = ks.split(",")

    for k in ks:
        if k == 0:
            selected_seeds = [0]
        else:
            selected_seeds = seeds

        for seed in selected_seeds:
            for task in tasks:
                os.system(f"python3 run_icl_task.py --model_name {model_name} --template_indices {template_indices} --seed {seed} --endpoint {endpoint} --model_name --task {task} --k {k}")


if __name__ == "__main__":
    main()