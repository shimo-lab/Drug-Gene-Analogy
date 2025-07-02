import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    input_dir = Path("output/analogy")
    output_dir = Path("output/images/ansdist")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_csv = 'output/analogy/stats.csv'
    df = pd.read_csv(stats_csv)
    for i, row in df.iterrows():
        print(f"Row {i}:")
        for col, val in row.items():
            print(f"{col}: {val}")

    d2g_G = r"$\mathrm{E}_{d\in D}\{|[d]|\}$"
    d2g_P1 = \
        r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in D_p}\{|[d]_p|\}\}$"
    d2g_P2 = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in \mathcal{D}_p\cap D}\{|[d]|\}\}$"  # noqa: E501

    g2d_G = r"$\mathrm{E}_{g\in G}\{|[g]|\}$"
    g2d_P1 = \
        r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in G_p}\{|[g]_p|\}\}$"
    g2d_p2 = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in \mathcal{G}_p\cap G}\{|[g]|\}\}$"  # noqa: E501

    embed2expdict = {}
    for embed in ['bcv', 'homemade']:
        embed2expdict[embed] = {
            "drug2gene": {"G": float(f"{df.loc[df['embed'] == embed, d2g_G].values[0]:.3f}"),  # noqa: E501
                          "P1": float(f"{df.loc[df['embed'] == embed, d2g_P1].values[0]:.3f}"),  # noqa: E501
                          "P2": float(f"{df.loc[df['embed'] == embed, d2g_P2].values[0]:.3f}")},  # noqa: E501
            "gene2drug": {"G": float(f"{df.loc[df['embed'] == embed, g2d_G].values[0]:.3f}"),  # noqa: E501
                          "P1": float(f"{df.loc[df['embed'] == embed, g2d_P1].values[0]:.3f}"),  # noqa: E501
                          "P2": float(f"{df.loc[df['embed'] == embed, g2d_p2].values[0]:.3f}")},  # noqa: E501
        }

    for embed in ["bcv", "homemade"]:
        for rdx, query_direction in enumerate(["drug2gene", "gene2drug"]):
            if embed == "bcv" and query_direction == "drug2gene":
                d = "d"
                setting_g_mean = r"$\mathrm{E}_{d\in \mathcal{D}}\{|[d]|\}=2.938$"  # noqa: E501
                setting_p1_mean = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in D_p}\{|[d]_p|\}\}=1.965$"  # noqa: E501
                setting_p2_mean = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in \mathcal{D}_p\cap D}\{|[d]|\}\}=2.738$"  # noqa: E501
            elif embed == "bcv" and query_direction == "gene2drug":
                d = "g"
                setting_g_mean = r"$\mathrm{E}_{g\in \mathcal{G}}\{|[g]|\}=10.008$"  # noqa: E501
                setting_p1_mean = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in G_p}\{|[g]_p|\}\}=5.050$"  # noqa: E501
                setting_p2_mean = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in \mathcal{G}_p\cap G}\{|[g]|\}\}=7.131$"  # noqa: E501
            elif embed == "homemade" and query_direction == "drug2gene":
                d = "d"
                setting_g_mean = r"$\mathrm{E}_{d\in \mathcal{D}}\{|[d]|\}=3.014$"  # noqa: E501
                setting_p1_mean = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in D_p}\{|[d]_p|\}\}=1.990$"  # noqa: E501
                setting_p2_mean = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{d\in \mathcal{D}_p\cap D}\{|[d]|\}\}=2.776$"  # noqa: E501
            elif embed == "homemade" and query_direction == "gene2drug":
                d = "g"
                setting_g_mean = r"$\mathrm{E}_{g\in \mathcal{G}}\{|[g]|\}=9.413$"  # noqa: E501
                setting_p1_mean = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in G_p}\{|[g]_p|\}\}=4.847$"  # noqa: E501
                setting_p2_mean = r"$\mathrm{E}_{p\in\mathcal{P}}\{\mathrm{E}_{g\in \mathcal{G}_p\cap G}\{|[g]|\}\}=6.714$"  # noqa: E501

            setting_g_stats = r"$|[" + d + r"]|$"
            setting_g_pkl = input_dir / f"G_{query_direction}_{embed}.pkl"

            setting_p1_stats = r"$|[" + d + r"]_p|$"
            setting_p1_pkl = input_dir / f"P1_{query_direction}_{embed}.pkl"

            setting_p2_stats = r"$|[" + d + r"]|$"
            setting_p2_pkl = input_dir / f"P2_{query_direction}_{embed}.pkl"

            with open(setting_g_pkl, "rb") as f:
                setting_g = pkl.load(f)
            with open(setting_p1_pkl, "rb") as f:
                setting_p1 = pkl.load(f)
            with open(setting_p2_pkl, "rb") as f:
                setting_p2 = pkl.load(f)

            # draw histogram
            # 1rows x 3cols
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            names = ["G", "P1", "P2"]
            stats = [setting_g_stats, setting_p1_stats, setting_p2_stats]
            stats_mean = [setting_g_mean, setting_p1_mean, setting_p2_mean]
            label_size = 20
            tick_size = 18
            title_size = 25
            legend_size = 15
            for idx, setting in enumerate([setting_g, setting_p1, setting_p2]):
                ax = axes[idx]
                bins = np.arange(
                    np.min(setting) - 0.5, np.max(setting) + 1.5, 1
                )

                if query_direction == "drug2gene":
                    color = "orange"
                    span = 5
                    title = "Setting " + names[idx]
                else:
                    color = "blue"
                    span = 20
                    title = "Setting " + names[idx] + r"$^{'}$"

                ax.hist(setting, bins=bins, color=color,
                        edgecolor="black", align="mid")
                # considering margin
                ax.set_title(title, fontsize=title_size, pad=10)
                ax.set_xticks(np.arange(0, np.max(setting) + 1, span))
                ax.tick_params(labelsize=tick_size)

                ax.set_xlabel(stats[idx], fontsize=label_size)
                ax.set_ylabel("Frequency", fontsize=label_size)

                # draw mean vertical line, mean from dict
                mean = embed2expdict[embed][query_direction][names[idx]]
                ax.axvline(
                    mean,
                    color="red",
                    linestyle="dashed",
                    linewidth=1,
                    label=stats_mean[idx],
                )
                ax.legend(fontsize=legend_size, loc="upper right")

            plt.tight_layout()
            save_path = output_dir / f"ansdist_{embed}_{query_direction}.pdf"
            print(f"Saving figure to {save_path}")
            fig.subplots_adjust(
                left=0.06, right=0.99, bottom=0.16, top=0.88, wspace=0.25
            )
            plt.savefig(save_path)
            plt.close()


if __name__ == "__main__":
    main()
