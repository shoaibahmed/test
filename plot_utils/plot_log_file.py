import os
import sys
import json
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn theme
sns.set(style="whitegrid")


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <{sys.argv[1]}>")
    exit()

log_file = sys.argv[1]
assert os.path.exists(log_file), log_file
print("Loading log file:", log_file)
log_file_base_name = os.path.splitext(os.path.split(log_file)[1])[0]

with open(log_file, 'r') as f:
    lines = f.readlines()
lines = [l for l in lines if l[0] == '{']
lines = [json.loads(l) for l in lines]
print("Total lines loaded:", len(lines))

baseline_stats = []
train_stats = []
eval_stats = []
final_stats = []
for l in lines:
    key_list = list(l.keys())
    if "train_loss" in l:
        train_stats.append(l)
    else:
        if len(key_list) == 1:
            if "baseline_" in key_list[0]:
                baseline_stats.append(l)
            elif "final_" in key_list[0]:
                final_stats.append(l)
            elif "train" in key_list[0]:
                eval_stats.append(l)
            else:
                raise RuntimeError(f"Unknown format: {l}")
        else:  # legacy format
            assert len(key_list) == 4 and "split" in key_list
            name = l["split"]
            if "baseline_" in name:
                baseline_stats.append(l)
            elif "final_" in name:
                final_stats.append(l)
            elif "train" in name:
                eval_stats.append(l)
            else:
                raise RuntimeError(f"Unknown legacy format: {l}")

max_len = 5
print("Baseline stats:", baseline_stats[:max_len])
print("Train stats:", train_stats[:max_len])
print("Eval stats:", eval_stats[:max_len])
print("Final stats:", final_stats[:max_len])

# Collect the head losses
train_head_losses = {}
lr_list = []
tokens_seen_list = []
for l in train_stats:
    head_losses = l["prediction_head_losses"]
    lr_list.append(l["lr"])
    tokens_seen_list.append(l["tokens_seen"])
    for k in head_losses:
        if k not in train_head_losses:
            train_head_losses[k] = []
        train_head_losses[k].append(head_losses[k])

# Plot the head losses
gradient_accumulation = 8
plot_grad_steps = False
plot_lines = False
for k in train_head_losses:
    print("Plotting loss for head:", k)
    data_list = train_head_losses[k]
    if not plot_grad_steps:  # stride to compute optim steps
        data_list = data_list[::gradient_accumulation]
    plt.figure(figsize=(8, 6))
    fontsize = 12
    plt.title(f"BoW head size: {k}", fontsize=fontsize)
    if plot_lines:
        plt.plot(range(len(data_list)), data_list)
    else:
        plt.scatter(range(len(data_list)), data_list, alpha=0.1)
    plt.xlabel("grad steps" if plot_grad_steps else "optim steps", fontsize=fontsize)
    plt.ylabel("training loss", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{log_file_base_name}_head_{k}.png", dpi=300)

# Plot the other stats (LR / tokens seen list)
for type, data_list in [("lr", lr_list), ("tokens_seen", tokens_seen_list)]:
    plt.figure(figsize=(8, 6))
    fontsize = 12
    plt.plot(range(len(data_list)), data_list)
    plt.xlabel("grad steps", fontsize=fontsize)
    plt.ylabel(type, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{log_file_base_name}_{type}.png", dpi=300)

# TODO: Plot the baseline stats

# TODO: Plot the training-time eval stats

# Plot the final stats
harness_eval_results = {}
perplexity_eval_results = {}
only_plot_acc = True
for l in final_stats:
    key_list = list(l.keys())
    if len(key_list) == 1:  # legacy format
        dataset_name = key_list[0].replace("final_", "")
        l = l[key_list[0]]  # get the stats
        key_list = list(l.keys())  # get the keys from the new list
        if len(key_list) in [1, 2]:  # harness evals
            for metric in key_list:
                if only_plot_acc and "acc" not in metric:
                    continue
                harness_eval_results[f"{dataset_name}_{metric}"] = l[metric]
        else:  # perplexity eval
            assert len(key_list) == 3, key_list
            assert "prediction_head_losses" in key_list
            for k in l["prediction_head_losses"]:
                perplexity_eval_results[f"{dataset_name}_head_{k}"] = l["prediction_head_losses"][k]
    else:  # legacy format
        assert len(key_list) == 4, key_list
        assert "split" in key_list and "prediction_head_losses" in key_list, key_list
        dataset_name = l["split"].replace("final_", "")
        for k in l["prediction_head_losses"]:
            perplexity_eval_results[f"{dataset_name}_head_{k}"] = l["prediction_head_losses"][k]

print("Perplexity eval dict:", perplexity_eval_results)
print("Harness eval dict:", harness_eval_results)

# Plot the perplexity evals
if len(perplexity_eval_results.keys()) > 0:
    plt.figure(figsize=(8, 6))
    fontsize = 12

    label_list = list(perplexity_eval_results.keys())
    data_list = [perplexity_eval_results[k] for k in label_list]

    colors = sns.color_palette("viridis", len(label_list))
    bars = plt.barh(range(len(label_list)), data_list, color=colors)
    plt.yticks(range(len(label_list)), label_list, fontsize=fontsize)

    # Adding value labels
    for bar in bars:
        plt.gca().text(bar.get_width() + 0.5,  # x-coordinate for label
                    bar.get_y() + bar.get_height() / 2,  # y-coordinate for label
                    f'{bar.get_width():.2f}',  # label text
                    ha='center', va='center', fontsize=fontsize, color='black')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.xlabel('Loss', fontsize=fontsize)
    plt.ylabel('Head size', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    max_val = max(data_list)
    plt.xlim(0.0, max_val + 1.0)

    plt.tight_layout()
    plt.savefig(f"{log_file_base_name}_final_perplexity.png", dpi=300)

if len(harness_eval_results.keys()) > 0:
    # Plot the harness eval results
    plt.figure(figsize=(8, 6))
    fontsize = 12

    label_list = list(harness_eval_results.keys())
    data_list = [harness_eval_results[k] * 100. for k in label_list]

    # Sort the labels (from highest to lowest)
    sorted_idx = np.argsort(data_list)
    label_list = [label_list[i] for i in sorted_idx]
    data_list = [data_list[i] for i in sorted_idx]

    colors = sns.color_palette("viridis", len(label_list))
    bars = plt.barh(range(len(label_list)), data_list, color=colors)
    plt.yticks(range(len(label_list)), label_list, fontsize=fontsize)

    # Adding value labels
    for bar in bars:
        plt.gca().text(bar.get_width() + 5,  # x-coordinate for label
                    bar.get_y() + bar.get_height() / 2,  # y-coordinate for label
                    f'{bar.get_width():.2f}%',  # label text
                    ha='center', va='center', fontsize=fontsize, color='black')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.xlabel('Accuracy', fontsize=fontsize)
    plt.ylabel('Dataset', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    max_val = max(data_list)
    plt.xlim(0.0, max_val + 10)

    plt.tight_layout()
    plt.savefig(f"{log_file_base_name}_final_harness.png", dpi=300)
