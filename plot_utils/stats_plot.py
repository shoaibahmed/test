#!/bin/python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn theme
sns.set(style="whitegrid")


config_name = "20k_8gpu_4heads"
if config_name == "20k_8gpu_1head":
    eval_results = {'mmlu': [0.25324027916251246], 'gsm8k': [0.0], 'hellaswag': [0.29575781716789484], 'truthfulqa_mc2': [0.4432557351549844], 'winogrande': [0.5272296764009471], 'arc_easy': [0.4145622895622896], 'arc_challenge': [0.23122866894197952], 'piqa': [0.5984766050054406], 'boolq': [0.5691131498470948], 'lambada_standard': [0.1009120900446342, 2460.5758382823415], 'toxigen': [0.42659574468085104]}
elif config_name == "20k_8gpu_4heads":
    eval_results = {'mmlu': [0.25801167924797036], 'gsm8k': [0.0], 'hellaswag': [0.2975502887870942], 'truthfulqa_mc2': [0.43635761257616407], 'winogrande': [0.510655090765588], 'arc_easy': [0.43434343434343436], 'arc_challenge': [0.22610921501706485], 'piqa': [0.6088139281828074], 'boolq': [0.5483180428134556], 'lambada_standard': [0.09916553464001553, 2849.5358978465097], 'toxigen': [0.4329787234042553]}
else:
    raise RuntimeError(f"Unknown config: {config_name}")

only_plot_acc = True
harness_eval_results = {}
for dataset_name in eval_results:
    results = eval_results[dataset_name]
    if dataset_name == "lambada_standard":
        results = [results[0]]  # select the first element
    assert len(results) == 1, results
    harness_eval_results[f"{dataset_name}"] = results[0]

# Plot the harness eval results
plt.figure(figsize=(8, 6))
fontsize = 12

label_list = list(harness_eval_results.keys())
data_list = [harness_eval_results[k] for k in label_list]

# Sort w.r.t. accuracy
sorted_idx = np.argsort(data_list)  # sort descending
label_list = [label_list[i] for i in sorted_idx]
data_list = [data_list[i] * 100. for i in sorted_idx]  # convert to percentage

colors = sns.color_palette("viridis", len(label_list))
bars = plt.barh(range(len(label_list)), data_list, color=colors)
plt.yticks(range(len(label_list)), label_list, fontsize=fontsize)

# Adding value labels
for bar in bars:
    plt.gca().text(bar.get_width() + 4,  # x-coordinate for label
                bar.get_y() + bar.get_height() / 2,  # y-coordinate for label
                f'{bar.get_width():.2f}%',  # label text
                ha='center', va='center', fontsize=fontsize, color='black')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.xlabel('Accuracy', fontsize=fontsize)
plt.ylabel('Dataset', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

max_val = max(data_list)
plt.xlim(0.0, max_val + 8)

plt.tight_layout()
plt.savefig(f"{config_name}_harness_plot.png", dpi=300)
