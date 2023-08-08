# import pdb; pdb.set_trace()

import argparse
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import pandas as pd

from seaborn_fig_2_grid import SeabornFig2Grid

from collections import defaultdict

plt.style.use('seaborn')
plt.rc('font', size=24)
plt.rc('figure', titlesize=36)
sns.set_style("whitegrid")
sns.set(font_scale=2)


# NOTE: to reproduce Figure 4, use `--fig grid`

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--target", type=str,
                      default="abs_mean_from_oracle")
  parser.add_argument("--fig", type=str,
                      default="grid")
  parser.add_argument("--text_seed", type=int, default=None)
  parser.add_argument("--struc_seed", type=int, default=None)
  args = parser.parse_args()


  valid_targets = ['mean_from_oracle', 'abs_mean_from_oracle']
  assert args.target in valid_targets, valid_targets
  ylabel = "Causal Error"
  xlabel = "Test Accuracy"
   
  dfs = []
  for dataset in ['trivial_0801', 'lda_0727', 'gpt2_0803']:
    dataset_name = {"gpt2": "GPT2", "lda": "LDA", "trivial": "Trivial"}
    dataset_name = dataset_name[dataset.split("_")[0]]

    for method in ["Prop", "IPW", "ME"]:
      infn = "acc_err_{}_{}_{}.csv".format(dataset, method.lower(), args.target)
      df = pd.read_csv(infn)
      df.columns = [col.strip() for col in df.columns]
      df.rename({'target': ylabel, 'test_acc': xlabel}, inplace=True, axis='columns')
      df = df.assign(dataset=[dataset_name for _ in range(df.shape[0])])
      df = df.assign(method=[method for _ in range(df.shape[0])])

      if args.text_seed is not None:
        df = df[df["text_seed"] == args.text_seed]
      if args.struc_seed is not None:
        df = df[df["struc_seed"] == args.struc_seed]

      dfs.append(df)

  df = pd.concat(dfs, ignore_index=True)

  methods = ["Prop", "IPW", "ME"]
  datasets = ["Trivial", "LDA", "GPT2"]
  markers = dict(zip(datasets, ["x", "+", "o", "*"]))
  # colors = dict(zip(methods, sns.color_palette(palette="colorblind", n_colors=4)))
  colors = sns.color_palette(palette="colorblind", n_colors=4)

  if args.fig == 'grid':
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(3, 3)

    xlim = (0.4, 1.0)
    ylim = (0.0, 0.4)
    idx = 0
    for row, method in enumerate(methods):
      for col, dataset in enumerate(datasets):
        marker = 'o'
        where = np.logical_and(df["method"] == method, df["dataset"] == dataset)
        g = sns.JointGrid(hue='struc_seed', palette="pastel",
            data=df[where], x=xlabel, y=ylabel, xlim=xlim, ylim=ylim)
        g.plot_joint(sns.kdeplot, fill=False, alpha=0.8)
        g.plot_marginals(sns.kdeplot, fill=True, bw_adjust=0.5)
        g.plot_joint(sns.scatterplot, marker=marker, alpha=0.5)
        ax = g.fig.axes[0]
        ax.get_legend().remove()

        where2 = np.all([df[where][xlabel] >= xlim[0], df[where][xlabel] <=xlim[1],
                         df[where][ylabel] >= ylim[0], df[where][ylabel] <= ylim[1]], axis=0)
        print("{} x {} = {:.1f}".format(method, dataset, 100 * np.mean(where2)))

        xticks = [0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(xticks)
        if row == 2:
          ax.set_xticklabels(map(str, xticks))
        else:
          ax.set_xticklabels([])
          ax.set_xlabel('')

        yticks = [0.0, 0.2, 0.4]
        ax.set_yticks(yticks)
        if col == 0:
          ax.set_yticklabels(map(str, yticks))
        else:
          ax.set_yticklabels([])
          ax.set_ylabel('')

        # Row/Col labels from:
        # https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
        if row == 0:
          g.fig.axes[1].annotate(
              dataset, xy=(0.5, 1), xytext=(0, 5), xycoords='axes fraction',
              textcoords='offset points', size='large', ha='center', va='baseline')
        if col == 0:
          ax = g.fig.axes[0]
          ax.annotate(
              method, xy=(-0.4, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
              xycoords='axes fraction', textcoords='offset points',
              size='large', ha='right', va='center')

        SeabornFig2Grid(g, fig, gs[idx])
        idx += 1

    gs.tight_layout(fig)
    gs.update(left=0.15, top=0.95)
    # plt.show()
    outfn = "test_grid{}.png".format(str(args.text_seed) if args.text_seed else "")
    plt.savefig(outfn)

  elif args.fig == 'lda_ipw':
    # fig = plt.subplots(1, 1, figsize=(8, 8))
    method, dataset = "IPW", "LDA"
    xlim = (0.4, 1.0)
    ylim = (0.0, 2.0)
    where = np.logical_and(df["method"] == method, df["dataset"] == dataset)
    g = sns.JointGrid(hue='struc_seed', palette="pastel",
        data=df[where], x=xlabel, y=ylabel, xlim=xlim, ylim=ylim)
    g.plot_joint(sns.kdeplot, fill=False, alpha=0.8)
    g.plot_marginals(sns.kdeplot, fill=True, bw_adjust=0.5)
    g.plot_joint(sns.scatterplot, marker='o', alpha=0.5)
    ax = g.fig.axes[0]
    yticks = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    ax.set_yticks(yticks)
    leg = ax.get_legend()
    if leg is not None:
      leg.remove()
    where2 = np.all([df[where][xlabel] >= xlim[0], df[where][xlabel] <=xlim[1],
                     df[where][ylabel] >= ylim[0], df[where][ylabel] <= ylim[1]], axis=0)
    print("{} x {} = {:.1f}".format(method, dataset, 100 * np.mean(where2)))
    # plt.show()
    outfn = "ipw_lda.png"
    plt.savefig(outfn)

  elif args.fig == 'mean':
    fig, axs = plt.subplots(1, 3, sharey=True)
    for col, dataset in enumerate(datasets):
      for row, method in enumerate(methods):
        color = colors[method]
        where = np.logical_and(df["method"] == method, df["dataset"] == dataset)

        step = 0.02
        x = np.arange(0.5, 1.01, step)
        y = []
        # for cutoff in x:
        #   y.append(np.mean(df["Causal Error"][np.logical_and(where, df["Test Accuracy"] > cutoff)]))
        for center in x:
          y.append(np.mean(df["Causal Error"][np.all(
              [where, df["Test Accuracy"] > center - step / 2,
               df["Test Accuracy"] < center + step / 2], axis=0)]))
        axs[col].set_ylim([0, 0.4])
        axs[col].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        axs[col].plot(x, y, color=color, label=method)
        axs[col].set_title(dataset)
        axs[col].set_xticks([0.5, 0.75, 1.0])
        axs[col].set_xticklabels([.5, .75, 1.])

    axs[0].legend()
    plt.show()

  elif args.fig == 'seed_mean':
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
    for col, dataset in enumerate(datasets):
      for row, method in enumerate(methods):
      # for row, method in enumerate(methods[:2]):
      # for row, method in enumerate(methods[2:]):

        for struc_seed in range(1, 5):
          color = "C" + str(struc_seed)
          where = np.all([df["method"] == method, df["dataset"] == dataset,
                          df["struc_seed"] == struc_seed], axis=0)
        # for text_seed in range(1, 5):
        #   color = "C" + str(text_seed)
        #   where = np.all([df["method"] == method, df["dataset"] == dataset,
        #                   df["text_seed"] == text_seed], axis=0)

          step = 0.02
          accs = df["Test Accuracy"][where]
          x = np.arange(np.min(accs), np.max(accs), step)
          means, mins, maxs, p025s, p975s = [], [], [], [], []
          for center in x:
            data = df["Causal Error"][np.all(
                [where, df["Test Accuracy"] > center - step / 2,
                 df["Test Accuracy"] < center + step / 2], axis=0)]
            if data.shape[0] > 0:
              a, b = np.percentile(data, [2.5, 97.5])
              p025s.append(a)
              p975s.append(b)
              means.append(np.mean(data))
              mins.append(np.min(data))
              maxs.append(np.max(data))
            elif len(means) > 0:
              p025s.append(p025s[-1])
              p975s.append(p025s[-1])
              means.append(means[-1])
              mins.append(mins[-1])
              maxs.append(maxs[-1])
            else:
              p025s.append(np.nan)
              p975s.append(np.nan)
              means.append(np.nan)
              mins.append(np.nan)
              maxs.append(np.nan)
   
          axs[row, col].set_ylim([0, 0.4])
          axs[row, col].set_yticks([0, 0.4])
          axs[row, col].plot(x, means, color=color, alpha=0.8)
          axs[row, col].fill_between(x, p025s, p975s, color=color, alpha=0.4)
          # axs[col].set_ylim([0, 0.4])
          # axs[col].set_yticks([0, 0.4])
          # axs[col].plot(x, means, color=color, alpha=0.8)
          # axs[col].fill_between(x, p025s, p975s, color=color, alpha=0.4)
      # axs[row, col].fill_between(x, mins, maxs, color=color, alpha=0.2)

    # axs[0, 0].legend()
    plt.show()

if __name__ == "__main__":
  main()
