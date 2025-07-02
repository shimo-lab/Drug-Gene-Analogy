from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main(search_names=['gene']):

    # analogy results
    data_path = Path("data/kge_datasplit") /\
        f"analogy_search-{'-and-'.join(search_names)}.csv"
    analogy_df = pd.read_csv(data_path)

    train_states = 'train'
    analogy_df = analogy_df[analogy_df['train_states'] == train_states]

    # TODO: add code for generating the following kge results
    # ps    MR	    MRR	    H1	    H3	    H10
    # 0.6	135     0.798	0.752	0.831	0.873
    # 0.5	160	    0.572	0.428	0.674	0.839
    # 0.4	189	    0.467	0.313	0.562	0.759
    # 0.3	229	    0.376	0.229	0.455	0.671
    # 0.2	296	    0.299	0.166	0.363	0.581
    # 0.1	459	    0.183	0.069	0.214	0.419
    ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    kge_top1 = [0.069, 0.166, 0.229, 0.313, 0.428, 0.752]
    kge_top10 = [0.419, 0.581, 0.671, 0.759, 0.839, 0.873]
    kge_mrr = [0.183, 0.299, 0.376, 0.467, 0.572, 0.798]

    ana_top1 = []
    ana_top10 = []
    ana_mrr = []
    for i, p in enumerate(ps):
        ana_top1.append(analogy_df[analogy_df['p'] == p]['top1'].values[0])
        ana_top10.append(analogy_df[analogy_df['p'] == p]['top10'].values[0])
        ana_mrr.append(analogy_df[analogy_df['p'] == p]['mrr'].values[0])

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(
        left=0.02, right=0.99, bottom=0.02, top=0.92, wspace=0.4)

    ana_color = 'red'
    kge_color = 'blue'

    metric2values = {
        'Top1': (kge_top1, ana_top1),
        'Top10': (kge_top10, ana_top10),
        'MRR': (kge_mrr, ana_mrr)
    }

    label_size = 20
    tick_size = 17
    marker_size = 10
    legend_size = 17
    subtitle_size = 22
    title_size = 24

    ps100 = [int(p * 100) for p in ps]
    for i, (metric, (kge_values, ana_values)) in enumerate(
            metric2values.items()):
        axes[i].grid()
        axes[i].set_title(metric, fontsize=subtitle_size, pad=10)
        axes[i].plot(ps100, ana_values, marker='o', color=ana_color,
                     label='Analogy', zorder=10, markersize=marker_size)
        axes[i].plot(ps100, kge_values, marker='^', color=kge_color,
                     label='TransE', zorder=5, markersize=marker_size)
        axes[i].set_xlabel('Proportion of training data (%)',
                           fontsize=label_size, labelpad=10)
        axes[i].set_xlim(5, 65)
        axes[i].set_xticks(ps100)
        axes[i].set_xticklabels(ps100, fontsize=tick_size)

        axes[i].set_ylim(-0.05, 1.05)
        axes[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes[i].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                fontsize=tick_size)
        if i == 0:
            axes[i].set_ylabel('Score', fontsize=label_size, labelpad=10)
            axes[i].legend(loc='center', fontsize=legend_size, ncol=2,
                           columnspacing=0.6, facecolor='white',
                           edgecolor='black', framealpha=1.0,
                           bbox_to_anchor=(0.425, 0.865))

    fig.suptitle('Comparison of Analogy and TransE in Setting P1',
                 fontsize=title_size)
    plt.tight_layout()

    # save the figure
    if len(search_names) == 1:
        fig.savefig(Path("data/kge_datasplit") / "analogy_vs_transe.pdf")
        plt.close()
    else:
        fig.savefig(Path("data/kge_datasplit") /
                    f"analogy_vs_transe-{'-and-'.join(search_names)}.pdf")
        plt.close()


if __name__ == "__main__":
    main()
