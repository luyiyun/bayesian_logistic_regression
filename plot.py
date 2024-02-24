from typing import Dict, Tuple, Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


def parse_xlsx(fn: str) -> Dict[str, pd.DataFrame]:
    xlsx = pd.read_excel(fn, sheet_name=None, header=[0, 1])
    res = {}
    for k, sheet in xlsx.items():
        if not k.startswith("scenario"):
            continue
        sheet.iloc[:, :2] = sheet.iloc[:, :2].ffill(axis=0)
        sheet = sheet.rename(columns={"Naive.1": "Bayes"})
        index_cols = sheet.iloc[:, :2]
        index_cols.columns = index_cols.columns.droplevel(1)
        sheet = sheet.iloc[:, 2:]
        sheet.index = pd.MultiIndex.from_frame(index_cols)
        sheet = sheet.stack().reset_index()
        colnames = sheet.columns.tolist()
        colnames[1] = "OR"
        colnames[2] = "Methods"
        sheet.columns = [ci.strip() for ci in colnames]
        sheet["OR"] = sheet["OR"].str.extract(r"log\((.*?)\)").astype(float)
        res[k] = sheet
    return res


def plot_lines(
    df: pd.DataFrame,
    exclude_methods: Tuple[str] = (),
    exclude_metrics: Tuple[str] = ("Bias", "Standard error"),
    palette: Optional[List] = None,
) -> plt.figure:
    df["Percent bias"] = df["Percent bias"].abs()
    df.drop(columns=list(exclude_metrics), inplace=True)
    df.columns = df.columns.str.strip()
    df = df[~df["Methods"].isin(exclude_methods)]
    df["Methods"] = df["Methods"].astype("category")

    prevalances = df["Prevalence"].unique()
    n_methods = df["Methods"].unique().shape[0]

    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    subfigs = fig.subfigures(3, 1, hspace=0.07)

    for i, (metrici, metric_name) in enumerate(
        zip(
            ["Percent bias", "MSE", "Coverage rate"],
            ["Percent Bias", "Mean Square Error", "Coverage Rate"],
        )
    ):
        subfigs[i].suptitle(metric_name)
        axs = subfigs[i].subplots(ncols=len(prevalances))
        for j, ax in enumerate(axs):
            dfi = df[df["Prevalence"] == prevalances[j]]
            if metrici == "Coverage rate":
                dfi = dfi.query("Methods != 'Naive'")
            sns.lineplot(
                data=dfi,
                x="OR",
                y=metrici,
                hue="Methods",
                ax=ax,
                palette=palette,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ax.set_title("Prevalence = %.2f" % prevalances[j])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=n_methods // 2,
        frameon=False,
        fancybox=False,
    )

    for subfig in subfigs:
        for ax in subfig.axes:
            ax.get_legend().remove()

    return fig


fn = "./results/simulation_result.xlsx"
dfs = parse_xlsx(fn)

keys = ["scenario6", "scenario7", "scenario8", "scenario9"]
for key in keys:
    fig = plot_lines(dfs[key], exclude_methods=("Naive", "x_only"))
    fig.savefig("./results/%s.png" % key)
