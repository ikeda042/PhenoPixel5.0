import os
import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np
import re

drug_order = {"gen": 0, "tri": 1, "cip": 2}


def sort_key(file: str) -> tuple[int, int]:
    """
    ファイル名から薬剤種と時間を抽出し、
    (薬剤の順序, 時間) のタプルを返します。
    """
    basename = os.path.basename(file)
    # パターン: sk326(薬剤種)(時間)min 例: sk326gen30min_dot_positions.csv
    m = re.search(r"sk326(gen|tri|cip)(\d+)min", basename)
    if m:
        drug = m.group(1)
        time_val = int(m.group(2))
        return drug_order[drug], time_val
    return (99, 0)  # 該当しない場合は末尾に


def plot_combined_dot_locations_from_csv(csv_path: str, output_path: str) -> None:
    """
    CSVファイルから各ドットの正規化された Rel X, Rel Y, Brightness を読み込み、
    Combined Dot Locations グラフ（散布図）を生成して指定のパスに保存します。

    数式:
        $$ \text{norm}_x = \frac{x}{w},\quad \text{norm}_y = \frac{y}{h} $$

    Latex生コード:
        \[
        \text{norm}_x = \frac{x}{w}
        \]
        \[
        \text{norm}_y = \frac{y}{h}
        \]

    Parameters:
        csv_path (str): 入力となるCSVファイルのパス。
        output_path (str): 作成するグラフ画像の保存先パス。
    """
    import matplotlib.colors

    dots: list[tuple[float, float, float]] = []
    with open(csv_path, mode="r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader, None)  # ヘッダー行をスキップ
        for row in csv_reader:
            try:
                rel_x, rel_y, brightness = map(float, row)
                dots.append((rel_x, rel_y, brightness))
            except ValueError:
                continue

    fig, ax = plt.subplots(figsize=(6, 6))
    if dots:
        xs = [dot[0] for dot in dots]
        ys = [dot[1] for dot in dots]
        brightness_vals = [dot[2] / 255 for dot in dots]
        sc = ax.scatter(
            xs,
            ys,
            c=brightness_vals,
            cmap="jet",
            norm=matplotlib.colors.NoNorm(),
            s=30,
            label="IbpA-GFP relative position",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No dots detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title("Combined Dot Locations")
    ax.set_xlabel("Rel. X (normalized)")
    ax.set_ylabel("Rel. Y (normalized)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined_average_dot_locations(csv_dir: str, output_path: str) -> None:
    """
    CSVファイル群から各ドットの Rel X, Rel Y のデータを読み込み、各ファイルごとに平均値を算出して、
    その平均点を1つのグラフ上に散布図としてプロットします。

    数式:
        $$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i,\quad \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i $$

    Latex生コード:
        \[
        \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
        \]
        \[
        \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i
        \]

    Parameters:
        csv_dir (str): CSVファイルが格納されているディレクトリのパス。
        output_path (str): 作成するグラフ画像の保存先パス。
    """
    file_names: list[str] = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    file_names.sort(key=sort_key)
    if not file_names:
        print("指定されたディレクトリにCSVファイルが見つかりませんでした。")
        return

    drug_colors = {"gen": "tab:orange", "cip": "tab:blue", "tri": "tab:green"}

    drug_data: dict[str, dict[str, list]] = {
        "gen": {"xs": [], "ys": [], "labels": []},
        "cip": {"xs": [], "ys": [], "labels": []},
        "tri": {"xs": [], "ys": [], "labels": []},
    }

    for file in file_names:
        csv_path = os.path.join(csv_dir, file)
        xs: list[float] = []
        ys: list[float] = []
        with open(csv_path, mode="r", newline="") as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader, None)
            for row in csv_reader:
                try:
                    x, y, _ = map(float, row)
                    xs.append(x)
                    ys.append(y)
                except ValueError:
                    continue
        if xs and ys:
            avg_x = sum(xs) / len(xs)
            avg_y = sum(ys) / len(ys)
            file_lower = file.lower()
            drug = None
            for key in ["gen", "cip", "tri"]:
                if key in file_lower:
                    drug = key
                    break
            if drug is None:
                drug = "gen"
            drug_data[drug]["xs"].append(avg_x)
            drug_data[drug]["ys"].append(avg_y)
            drug_data[drug]["labels"].append(file.replace(".csv", ""))
        else:
            print(f"CSVファイル {file} からデータが読み込めませんでした。")

    if not any(len(drug_data[d]["xs"]) > 0 for d in drug_data):
        print("有効なデータがありませんでした。")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for drug in ["gen", "cip", "tri"]:
        if drug_data[drug]["xs"]:
            ax.scatter(
                drug_data[drug]["xs"],
                drug_data[drug]["ys"],
                s=50,
                c=drug_colors[drug],
                label=f"{drug.upper()} Average Position",
            )
    ax.set_title("Combined Average Dot Locations")
    ax.set_xlabel("Rel. X (normalized)")
    ax.set_ylabel("Rel. Y (normalized)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined_cv_dot_locations(csv_dir: str, output_path: str) -> None:
    """
    CSVファイル群から各ドットの Rel X, Rel Y のデータを読み込み、各ファイルごとに変動係数（Coefficient of Variation）を計算して、
    その結果を1つのグラフ上にグループ化された棒グラフとしてプロットします。

    数式:
        $$ CV_x = \frac{\sigma_x}{\bar{x}},\quad CV_y = \frac{\sigma_y}{\bar{y}} $$

    Latex生コード:
        \[
        CV_x = \frac{\sigma_x}{\bar{x}}
        \]
        \[
        CV_y = \frac{\sigma_y}{\bar{y}}
        \]

    Parameters:
        csv_dir (str): CSVファイルが格納されているディレクトリのパス。
        output_path (str): 作成するグラフ画像の保存先パス。
    """
    file_names: list[str] = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    file_names.sort(key=sort_key)
    if not file_names:
        print("指定されたディレクトリにCSVファイルが見つかりませんでした。")
        return

    labels: list[str] = []
    cv_xs: list[float] = []
    cv_ys: list[float] = []

    for file in file_names:
        csv_path = os.path.join(csv_dir, file)
        xs: list[float] = []
        ys: list[float] = []
        with open(csv_path, mode="r", newline="") as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader, None)
            for row in csv_reader:
                try:
                    x, y, _ = map(float, row)
                    xs.append(x)
                    ys.append(y)
                except ValueError:
                    continue
        if len(xs) < 2 or len(ys) < 2:
            print(f"CSVファイル {file} から十分なデータが読み込めませんでした。")
            continue
        mean_x = statistics.mean(xs)
        mean_y = statistics.mean(ys)
        stdev_x = statistics.stdev(xs)
        stdev_y = statistics.stdev(ys)
        cv_x = stdev_x / mean_x if mean_x != 0 else 0
        cv_y = stdev_y / mean_y if mean_y != 0 else 0

        labels.append(file.replace(".csv", ""))
        cv_xs.append(cv_x)
        cv_ys.append(cv_y)

    if not cv_xs or not cv_ys:
        print("有効な変動係数データがありませんでした。")
        return

    x_indices = np.arange(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(
        x_indices - bar_width / 2,
        cv_xs,
        width=bar_width,
        color="tab:blue",
        label="CV of Rel. X",
    )
    bars2 = ax.bar(
        x_indices + bar_width / 2,
        cv_ys,
        width=bar_width,
        color="tab:green",
        label="CV of Rel. Y",
    )

    ax.set_title("Combined Coefficient of Variation of Dot Locations")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined_average_dot_locations_with_errorbars(
    csv_dir: str, output_path: str
) -> None:
    """
    CSVファイル群から、ファイル名に "120min" を含むデータのみを対象に、
    各薬剤ごと（gen, tri, cip）に各CSVファイルから Rel X, Rel Y の平均値を算出し、
    それら3つの値の平均と標準偏差を用いてエラーバー付きの散布図をプロットします.

    数式:
        $$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i,\quad \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i $$
        $$ \sigma_x = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2},\quad \sigma_y = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (y_i - \bar{y})^2} $$

    Latex生コード:
        \[
        \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
        \]
        \[
        \bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i
        \]
        \[
        \sigma_x = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}
        \]
        \[
        \sigma_y = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (y_i - \bar{y})^2}
        \]

    Parameters:
        csv_dir (str): CSVファイルが格納されているディレクトリのパス。
        output_path (str): 作成するグラフ画像の保存先パス。
    """
    drug_data: dict[str, dict[str, list[float]]] = {
        "gen": {"xs": [], "ys": []},
        "tri": {"xs": [], "ys": []},
        "cip": {"xs": [], "ys": []},
    }

    file_names: list[str] = [
        f for f in os.listdir(csv_dir) if f.endswith(".csv") and "120min" in f
    ]
    file_names.sort(key=sort_key)
    if not file_names:
        print("対象となる120minのCSVファイルが見つかりませんでした。")
        return

    for file in file_names:
        csv_path = os.path.join(csv_dir, file)
        xs: list[float] = []
        ys: list[float] = []
        with open(csv_path, mode="r", newline="") as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader, None)
            for row in csv_reader:
                try:
                    x, y, _ = map(float, row)
                    xs.append(x)
                    ys.append(y)
                except ValueError:
                    continue
        if xs and ys:
            avg_x = sum(xs) / len(xs)
            avg_y = sum(ys) / len(ys)
            file_lower = file.lower()
            for key in ["gen", "tri", "cip"]:
                if key in file_lower:
                    drug_data[key]["xs"].append(avg_x)
                    drug_data[key]["ys"].append(avg_y)
                    break
        else:
            print(f"CSVファイル {file} から十分なデータが読み込めませんでした。")

    drug_colors = {"gen": "tab:orange", "cip": "tab:blue", "tri": "tab:green"}
    fig, ax = plt.subplots(figsize=(8, 8))

    for drug in ["gen", "cip", "tri"]:
        rep_x = drug_data[drug]["xs"]
        rep_y = drug_data[drug]["ys"]
        if rep_x and rep_y:
            mean_x = statistics.mean(rep_x)
            mean_y = statistics.mean(rep_y)
            std_x = statistics.stdev(rep_x) if len(rep_x) > 1 else 0
            std_y = statistics.stdev(rep_y) if len(rep_y) > 1 else 0
            ax.errorbar(
                mean_x,
                mean_y,
                xerr=std_x,
                yerr=std_y,
                fmt="o",
                color=drug_colors[drug],
                capsize=5,
                label=f"{drug.upper()} 120min Average",
            )
        else:
            print(f"{drug.upper()} の120minデータが不足しています。")

    ax.set_title("Combined 120min Average Dot Locations with Error Bars")
    ax.set_xlabel("Rel. X (normalized)")
    ax.set_ylabel("Rel. Y (normalized)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined_n_dot_locations_for_drugs(
    csv_files: list[str], output_path: str
) -> None:
    """
    CSVファイル群から各ドットの Rel X, Rel Y のデータを読み込み、
    各薬剤（gen, tri, cip）について、n1, n2, n3 のデータを結合して散布図を作成します。

    数式:
        \[
        \text{Rel X (normalized)}
        \]
        \[
        \text{Rel Y (normalized)}
        \]

    Parameters:
        csv_files (list[str]): 結合対象のCSVファイルのリスト。ファイル名には薬剤種（gen, tri, cip）および n 番号（n1, n2, n3）が含まれている必要があります。
        output_path (str): 作成するグラフ画像の保存先パス。
    """
    drug_data: dict[str, dict[str, list[float]]] = {
        "gen": {"xs": [], "ys": []},
        "tri": {"xs": [], "ys": []},
        "cip": {"xs": [], "ys": []},
    }

    drug_colors = {"gen": "tab:orange", "cip": "tab:blue", "tri": "tab:green"}

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"CSVファイル {csv_file} が存在しません。")
            continue
        with open(csv_file, mode="r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                try:
                    x, y, _ = map(float, row)
                    file_lower = os.path.basename(csv_file).lower()
                    if "gen" in file_lower:
                        drug_data["gen"]["xs"].append(x)
                        drug_data["gen"]["ys"].append(y)
                    elif "tri" in file_lower:
                        drug_data["tri"]["xs"].append(x)
                        drug_data["tri"]["ys"].append(y)
                    elif "cip" in file_lower:
                        drug_data["cip"]["xs"].append(x)
                        drug_data["cip"]["ys"].append(y)
                    else:
                        print(
                            f"CSVファイル {csv_file} から薬剤種を判定できませんでした。"
                        )
                except ValueError:
                    continue

    fig, ax = plt.subplots(figsize=(6, 6))
    for drug, data in drug_data.items():
        if data["xs"] and data["ys"]:
            ax.scatter(
                data["xs"],
                data["ys"],
                s=20,
                c=drug_colors[drug],
                label=f"{drug.upper()}",
            )
        else:
            print(f"{drug.upper()} のデータが不足しています。")

    ax.set_title("Combined n1, n2, n3 Dot Locations for Each Antibiotic")
    ax.set_xlabel("Rel. X")
    ax.set_ylabel("Rel. Y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined_n_boxplot_for_antibiotics(
    csv_files: list[str], output_path: str
) -> None:
    """
    CSVファイル群から各ドットの Rel X, Rel Y のデータを読み込み、
    各抗生物質（gen, tri, cip）について、n1, n2, n3 のデータを結合し、
    それぞれの Rel X および Rel Y に対するボックスプロットをSeabornで描画し、
    各群間の統計的有意差を p 値とアスタリスクで示します。

    ボックスプロットは以下の統計量を示します:
        - 第1四分位数 (Q1)
        - 中央値 (Q2)
        - 第3四分位数 (Q3)

    数式:
        $$ Q_1 = \text{25th percentile},\quad Q_2 = \text{median},\quad Q_3 = \text{75th percentile} $$

    Latex生コード:
        \[
        Q_1 = \text{25th percentile}
        \]
        \[
        Q_2 = \text{median}
        \]
        \[
        Q_3 = \text{75th percentile}
        \]

    Parameters:
        csv_files (list[str]): 結合対象のCSVファイルのリスト。ファイル名には抗生物質種（gen, tri, cip）および n 番号（n1, n2, n3）が含まれている必要があります。
        output_path (str): 作成するボックスプロット画像の保存先パス。
    """
    import os
    import csv
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy.stats import ttest_ind

    # 抗生物質ごとのデータを保持する辞書
    antibiotic_data: dict[str, dict[str, list[float]]] = {
        "gen": {"xs": [], "ys": []},
        "tri": {"xs": [], "ys": []},
        "cip": {"xs": [], "ys": []},
    }

    # 各CSVファイルからデータを読み込み
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"CSVファイル {csv_file} が存在しません。")
            continue
        with open(csv_file, mode="r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                try:
                    x, y, _ = map(float, row)
                    file_lower = os.path.basename(csv_file).lower()
                    if "gen" in file_lower:
                        antibiotic_data["gen"]["xs"].append(x)
                        antibiotic_data["gen"]["ys"].append(y)
                    elif "tri" in file_lower:
                        antibiotic_data["tri"]["xs"].append(x)
                        antibiotic_data["tri"]["ys"].append(y)
                    elif "cip" in file_lower:
                        antibiotic_data["cip"]["xs"].append(x)
                        antibiotic_data["cip"]["ys"].append(y)
                    else:
                        print(
                            f"CSVファイル {csv_file} から抗生物質種を判定できませんでした。"
                        )
                except ValueError:
                    continue

    antibiotics = ["gen", "tri", "cip"]
    # DataFrameを作成（Rel X, Rel Yそれぞれ）
    records_x = []
    records_y = []
    for antibiotic in antibiotics:
        for val in antibiotic_data[antibiotic]["xs"]:
            records_x.append({"Antibiotic": antibiotic.upper(), "RelX": val})
        for val in antibiotic_data[antibiotic]["ys"]:
            records_y.append({"Antibiotic": antibiotic.upper(), "RelY": val})
    df_x = pd.DataFrame(records_x)
    df_y = pd.DataFrame(records_y)

    # カラーパレットの設定
    antibiotic_colors = {"GEN": "tab:orange", "TRI": "tab:green", "CIP": "tab:blue"}

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 8))

    # Seabornのboxplotを用いてRel Xのプロット（外れ値を表示しない）
    sns.boxplot(
        x="Antibiotic",
        y="RelX",
        data=df_x,
        order=["GEN", "TRI", "CIP"],
        palette=antibiotic_colors,
        width=0.5,
        ax=ax1,
        showfliers=False,
    )
    ax1.set_title("Box Plot of Rel. X")
    ax1.set_ylabel("Rel. X ", fontsize=16)
    ax1.grid(True)

    # Seabornのboxplotを用いてRel Yのプロット（外れ値を表示しない）
    sns.boxplot(
        x="Antibiotic",
        y="RelY",
        data=df_y,
        order=["GEN", "TRI", "CIP"],
        palette=antibiotic_colors,
        width=0.5,
        ax=ax2,
        showfliers=False,
    )
    ax2.set_title("Box Plot of Rel. Y")
    ax2.set_ylabel("Rel. Y ", fontsize=16)
    ax2.grid(True)

    # ヘルパー関数：p値に応じた有意性マーカーを返す
    def significance_marker(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    # ヘルパー関数：統計的有意差の線とテキストを描画する
    def add_stat_annotation(
        ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str
    ) -> None:
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c="k")
        ax.text((x1 + x2) * 0.5, y + h, text, ha="center", va="bottom", color="k")

    # Seabornのboxplot上のカテゴリのx位置は0,1,2の順序に対応
    data_x = [antibiotic_data[a]["xs"] for a in antibiotics]
    data_y = [antibiotic_data[a]["ys"] for a in antibiotics]
    pairs = [(0, 1), (0, 2), (1, 2)]

    # Rel Xについての統計的有意差の注釈（ax1）
    for idx, (i, j) in enumerate(pairs):
        group1 = data_x[i]
        group2 = data_x[j]
        if group1 and group2:
            stat = ttest_ind(group1, group2)
            p = stat.pvalue
            marker = significance_marker(p)
            y_max = max(max(group1), max(group2))
            offset = 0.02 * (idx + 1)
            y_coord = y_max + offset
            add_stat_annotation(ax1, i, j, y_coord, 0.005, f"p={p:.3g}\n{marker}")

    # Rel Yについての統計的有意差の注釈（ax2）
    for idx, (i, j) in enumerate(pairs):
        group1 = data_y[i]
        group2 = data_y[j]
        if group1 and group2:
            stat = ttest_ind(group1, group2)
            p = stat.pvalue
            marker = significance_marker(p)
            y_max = max(max(group1), max(group2))
            offset = 0.02 * (idx + 1)
            y_coord = y_max + offset
            add_stat_annotation(ax2, i, j, y_coord, 0.005, f"p={p:.3g}\n{marker}")

    fig.suptitle("Combined Box Plots for Each Antibiotic")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined_n_violin_for_drugs(csv_files: list[str], output_path: str) -> None:
    """
    CSVファイル群から各ドットの Rel X, Rel Y のデータを読み込み、
    各薬剤（gen, tri, cip）について、n1, n2, n3 のデータを結合し、
    それぞれの Rel X および Rel Y に対するバイオリンプロットを描画します。

    バイオリンプロットは、データの分布をカーネル密度推定 (Kernel Density Estimate, KDE) により視覚化します。

    数式 (KDE の概念):
        $$ \hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x-x_i}{h}\right) $$

    Latex生コード:
        \[
        \hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x-x_i}{h}\right)
        \]

    Parameters:
        csv_files (list[str]): 結合対象のCSVファイルのリスト。ファイル名には薬剤種（gen, tri, cip）および n 番号（n1, n2, n3）が含まれている必要があります。
        output_path (str): 作成するバイオリンプロット画像の保存先パス。
    """
    drug_data: dict[str, dict[str, list[float]]] = {
        "gen": {"xs": [], "ys": []},
        "tri": {"xs": [], "ys": []},
        "cip": {"xs": [], "ys": []},
    }

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"CSVファイル {csv_file} が存在しません。")
            continue
        with open(csv_file, mode="r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                try:
                    x, y, _ = map(float, row)
                    file_lower = os.path.basename(csv_file).lower()
                    if "gen" in file_lower:
                        drug_data["gen"]["xs"].append(x)
                        drug_data["gen"]["ys"].append(y)
                    elif "tri" in file_lower:
                        drug_data["tri"]["xs"].append(x)
                        drug_data["tri"]["ys"].append(y)
                    elif "cip" in file_lower:
                        drug_data["cip"]["xs"].append(x)
                        drug_data["cip"]["ys"].append(y)
                    else:
                        print(
                            f"CSVファイル {csv_file} から薬剤種を判定できませんでした。"
                        )
                except ValueError:
                    continue

    drugs = ["gen", "tri", "cip"]
    data_x = [drug_data[d]["xs"] for d in drugs]
    data_y = [drug_data[d]["ys"] for d in drugs]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # Violin plot for Rel X
    parts_x = ax1.violinplot(data_x, showmedians=True, showmeans=False)
    drug_colors = {"gen": "tab:orange", "tri": "tab:green", "cip": "tab:blue"}
    for i, body in enumerate(parts_x["bodies"]):
        body.set_facecolor(drug_colors[drugs[i]])
        body.set_edgecolor("black")
        body.set_alpha(0.7)
    ax1.set_title("Violin Plot of Rel. X (normalized)")
    ax1.set_ylabel("Rel. X (normalized)")
    ax1.set_xticks(np.arange(1, len(drugs) + 1))
    ax1.set_xticklabels([d.upper() for d in drugs])
    ax1.grid(True)

    # Violin plot for Rel Y
    parts_y = ax2.violinplot(data_y, showmedians=True, showmeans=False)
    for i, body in enumerate(parts_y["bodies"]):
        body.set_facecolor(drug_colors[drugs[i]])
        body.set_edgecolor("black")
        body.set_alpha(0.7)
    ax2.set_title("Violin Plot of Rel. Y (normalized)")
    ax2.set_ylabel("Rel. Y (normalized)")
    ax2.set_xticks(np.arange(1, len(drugs) + 1))
    ax2.set_xticklabels([d.upper() for d in drugs])
    ax2.grid(True)

    fig.suptitle("Combined n1, n2, n3 Violin Plots for Each Antibiotic")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    # CSVファイルが保存されているディレクトリのパス（例: experimental/DotPatternMap/images）
    csv_directory = "experimental/DotPatternMap/images"

    # 平均点の散布図の保存先
    avg_output_file = os.path.join(csv_directory, "combined_average_dot_locations.png")
    plot_combined_average_dot_locations(csv_directory, avg_output_file)
    print(f"平均点グラフを {avg_output_file} に保存しました。")

    # 変動係数の棒グラフの保存先
    cv_output_file = os.path.join(csv_directory, "combined_cv_dot_locations.png")
    plot_combined_cv_dot_locations(csv_directory, cv_output_file)
    print(f"変動係数グラフを {cv_output_file} に保存しました。")

    # 各ドットの位置情報の散布図の保存先
    paths = [
        os.path.join(csv_directory, i)
        for i in os.listdir(csv_directory)
        if i.endswith(".csv")
    ]
    paths.sort(key=sort_key)
    for path in paths:
        print(f"CSVファイル {path} から散布図を作成しています...")
        output = path.replace(".csv", ".png")
        plot_combined_dot_locations_from_csv(path, output)
        print(f"散布図を {output} に保存しました。")

    # 120minのデータについて、エラーバー付き平均位置グラフを作成
    avg_err_output_file = os.path.join(
        csv_directory, "combined_120min_average_with_errorbars.png"
    )
    plot_combined_average_dot_locations_with_errorbars(
        csv_directory, avg_err_output_file
    )
    print(
        f"120minデータのエラーバー付き平均位置グラフを {avg_err_output_file} に保存しました。"
    )

    # n1, n2, n3 のデータを結合した散布図の保存
    plot_combined_n_dot_locations_for_drugs(
        paths, os.path.join(csv_directory, "combined_n_dot_locations.png")
    )

    # n1, n2, n3 の結合データに対するボックスプロットの保存
    boxplot_output_file = os.path.join(csv_directory, "combined_n_boxplot.png")
    plot_combined_n_boxplot_for_antibiotics(paths, boxplot_output_file)
    print(f"ボックスプロットを {boxplot_output_file} に保存しました。")

    # n1, n2, n3 の結合データに対するバイオリンプロットの保存
    violin_output_file = os.path.join(csv_directory, "combined_n_violin.png")
    plot_combined_n_violin_for_drugs(paths, violin_output_file)
    print(f"バイオリンプロットを {violin_output_file} に保存しました。")
