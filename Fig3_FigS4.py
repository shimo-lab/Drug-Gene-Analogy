from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    settings = ["Y1", "Y2", "P1Y1", "P2Y1", "P1Y2", "P2Y2"]

    df_all = pd.DataFrame()
    if "Y1" in settings:
        csv_path = Path("output/analogy_Y1/result.csv")
        df_tmp = pd.read_csv(csv_path)
        df_all = pd.concat([df_all, df_tmp], ignore_index=True)
    if "Y2" in settings:
        csv_path = Path("output/analogy_Y2/result.csv")
        df_tmp = pd.read_csv(csv_path)
        df_all = pd.concat([df_all, df_tmp], ignore_index=True)
    if "P1Y1" in settings and "P2Y1" in settings:
        csv_path = Path("output/analogy_P1Y1_and_P2Y1/result.csv")
        df_tmp = pd.read_csv(csv_path)
        df_all = pd.concat([df_all, df_tmp], ignore_index=True)
    if "P1Y2" in settings and "P2Y2" in settings:
        csv_path = Path("output/analogy_P1Y2_and_P2Y2/result.csv")
        df_tmp = pd.read_csv(csv_path)
        df_all = pd.concat([df_all, df_tmp], ignore_index=True)

    ls = 18
    ts = 15
    ds = 10
    lw = 2
    ts_2023 = 12

    output_dir = Path("output/images/years")
    output_dir.mkdir(parents=True, exist_ok=True)

    top1_color = "#ff7f0e"
    top10_color = "#1f77b4"
    mrr_color = "#2ca02c"

    for si, setting in enumerate(settings):
        fig, ax2 = plt.subplots(figsize=(7, 4))
        fig.subplots_adjust(left=0.11, right=0.85, bottom=0.25, top=0.97)
        ax1 = ax2.twinx()
        methods = [setting, f"{setting}_random"]
        for method in methods:
            if method == setting:
                df = df_all.query(
                    f'query_direction=="drug2gene" & method == "{method}" & '
                    "weighted_mean==True & centering==True"
                )
            else:
                df = df_all.query(
                    f'query_direction=="drug2gene" & method == "{method}"'
                )
            df = df.drop(
                columns=[f"top{i}_acc" for i in range(1, 11)
                         if i not in [1, 10]]
            )
            data = []
            years = df["year"].values
            top1_accs = df["top1_acc"].values
            top10_accs = df["top10_acc"].values
            mrrs = df["mrr"].values
            query_nums = df["query_num"].values
            for year, top1_acc, top10_acc, mrr, query_num in zip(
                years, top1_accs, top10_accs, mrrs, query_nums
            ):
                data.append([year, int(query_num), top1_acc, top10_acc, mrr])

                if year == 2023:
                    if method == setting:
                        ax2.text(
                            year,
                            top1_acc + 0.01,
                            f"{top1_acc:.3f}",
                            fontsize=ts_2023,
                            ha="center",
                            va="bottom",
                            color=top1_color,
                        )
                        ax2.text(
                            year,
                            top10_acc + 0.01,
                            f"{top10_acc:.3f}",
                            fontsize=ts_2023,
                            ha="center",
                            va="bottom",
                            color=top10_color,
                        )
                        ax2.text(
                            year,
                            mrr + 0.01,
                            f"{mrr:.3f}",
                            fontsize=ts_2023,
                            ha="center",
                            va="bottom",
                            color=mrr_color,
                        )
                    else:
                        # query num
                        if "P" in setting:
                            ax1.text(
                                year,
                                query_num + 50,
                                f"{int(query_num)}",
                                fontsize=ts_2023,
                                ha="center",
                                va="bottom",
                                color="black",
                            )
                        else:
                            ax1.text(
                                year,
                                query_num - 50,
                                f"{int(query_num)}",
                                fontsize=ts_2023,
                                ha="center",
                                va="top",
                                color="black",
                            )

            df = pd.DataFrame(
                data, columns=["year", "query", "top1", "top10", r"MRR"])

            # draw graph
            # 2 y axis
            if method == setting:
                ax2.plot(
                    df["year"],
                    df["top1"],
                    color=top1_color,
                    label="Top1",
                    linewidth=lw,
                    marker="o",
                )
                ax2.plot(
                    df["year"],
                    df["top10"],
                    color=top10_color,
                    label="Top10",
                    linewidth=lw,
                    marker="^",
                )
                ax2.plot(
                    df["year"],
                    df[r"MRR"],
                    color=mrr_color,
                    label="MRR",
                    linewidth=lw,
                    marker="s",
                )
            else:
                ax2.plot(
                    df["year"],
                    df["top1"],
                    color=top1_color,
                    label="Random Top1",
                    linestyle="dotted",
                    linewidth=lw,
                    marker="o",
                )
                ax2.plot(
                    df["year"],
                    df["top10"],
                    color=top10_color,
                    label="Random Top10",
                    linestyle="dotted",
                    linewidth=lw,
                    marker="^",
                )
                ax2.plot(
                    df["year"],
                    df[r"MRR"],
                    color=mrr_color,
                    label="Random MRR",
                    linestyle="dotted",
                    linewidth=lw,
                    marker="s",
                )

        # score
        ax2.set_ylim(0.0, 1.1)
        ax2.set_ylabel("Score", fontsize=ls)
        ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.tick_params(axis="y", labelsize=ts)
        ax2.set_xlabel("Year", fontsize=ls)
        # tick rotation
        ax2.tick_params(axis="x", labelsize=ts, rotation=60)
        ax2.set_xticks([1975 + 5 * i for i in range(10)] + [2023])

        # query
        ax1.plot(
            years,
            query_nums,
            color="black",
            label="Number of Queries",
            linestyle="-.",
            marker="x",
        )
        ax1.set_ylabel(
            "Number of Queries", rotation=270, fontsize=ls, labelpad=25)
        ax1.tick_params(axis="y", labelsize=ts)

        if max(query_nums) < 1000:
            ymax = max(query_nums) + (249 - max(query_nums) % 200)
            ax1.set_ylim(0, ymax + 100)
        else:
            ymax = max(query_nums) + (599 - max(query_nums) % 500)
            ax1.set_ylim(0, ymax)
        if ymax < 1000:
            ax1.set_yticks([200 * i for i in range(ymax // 200 + 1)])
        else:
            ax1.set_yticks([1000 * i for i in range(ymax // 1000 + 1)])

        if si == 1 or si == 5:
            ax2.legend(
                loc="upper left", fontsize=ds, ncol=2, columnspacing=0.5)
            ax1.legend(loc="upper right", fontsize=ds)

        # save
        save_pdf_path = output_dir / f"{setting}.pdf"
        print(save_pdf_path)
        plt.savefig(save_pdf_path)


if __name__ == "__main__":
    main()
