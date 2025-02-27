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
    CSVファイルから各ドットの正規化されたRel X, Rel Y, Brightnessを読み込み、
    Combined Dot Locationsグラフ（散布図）を生成して指定のパスに保存します。

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
        # 輝度は0〜255の値として保存されているため、正規化（0〜1）します
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
        # plt.colorbar(sc, ax=ax, label="IbpA-GFP Intensity (normalized)")
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
    CSVファイル群から各ドットのRelX, RelYのデータを読み込み、各ファイルごとに平均値を計算して、
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

    # 薬剤ごとの色のマッピング
    drug_colors = {"gen": "orange", "cip": "blue", "tri": "green"}

    # 薬剤ごとに平均値のデータを格納する辞書
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
            header = next(csv_reader, None)  # ヘッダー行があればスキップ
            for row in csv_reader:
                try:
                    # CSVは [RelX, RelY, Brightness] と仮定
                    x, y, _ = map(float, row)
                    xs.append(x)
                    ys.append(y)
                except ValueError:
                    continue
        if xs and ys:
            avg_x = sum(xs) / len(xs)
            avg_y = sum(ys) / len(ys)
            # ファイル名から薬剤種を判定
            file_lower = file.lower()
            drug = None
            for key in ["gen", "cip", "tri"]:
                if key in file_lower:
                    drug = key
                    break
            if drug is None:
                drug = "gen"  # 薬剤種が判定できない場合はgenとする
            drug_data[drug]["xs"].append(avg_x)
            drug_data[drug]["ys"].append(avg_y)
            drug_data[drug]["labels"].append(file.replace(".csv", ""))
        else:
            print(f"CSVファイル {file} からデータが読み込めませんでした。")

    # 有効なデータがあるか確認
    if not any(len(drug_data[d]["xs"]) > 0 for d in drug_data):
        print("有効なデータがありませんでした。")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    # 薬剤ごとに散布図をプロット
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
    CSVファイル群から各ドットのRelX, RelYのデータを読み込み、各ファイルごとに変動係数（Coefficient of Variation）を計算して、
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
                    # CSVは [RelX, RelY, Brightness] と仮定
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

    # グループ化された棒グラフのプロット
    x_indices = np.arange(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(
        x_indices - bar_width / 2,
        cv_xs,
        width=bar_width,
        color="blue",
        label="CV of Rel. X",
    )
    bars2 = ax.bar(
        x_indices + bar_width / 2,
        cv_ys,
        width=bar_width,
        color="green",
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
    各薬剤ごと（gen, tri, cip）に各CSVファイルからRel X, Rel Yの平均値を算出し、
    それら3つの値の平均と標準偏差を用いてエラーバー付きの散布図をプロットします。

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
    # 薬剤ごとのデータを保持する辞書（各CSVから算出した平均値のリスト）
    drug_data: dict[str, dict[str, list[float]]] = {
        "gen": {"xs": [], "ys": []},
        "tri": {"xs": [], "ys": []},
        "cip": {"xs": [], "ys": []},
    }

    # 対象ファイルは "120min" を含むものとする
    file_names: list[str] = [
        f for f in os.listdir(csv_dir) if f.endswith(".csv") and "120min" in f
    ]
    file_names.sort(key=sort_key)
    if not file_names:
        print("対象となる120minのCSVファイルが見つかりませんでした。")
        return

    # 各CSVファイルから平均値を算出
    for file in file_names:
        csv_path = os.path.join(csv_dir, file)
        xs: list[float] = []
        ys: list[float] = []
        with open(csv_path, mode="r", newline="") as csv_file:
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader, None)  # ヘッダー行があればスキップ
            for row in csv_reader:
                try:
                    # CSVは [RelX, RelY, Brightness] と仮定
                    x, y, _ = map(float, row)
                    xs.append(x)
                    ys.append(y)
                except ValueError:
                    continue
        if xs and ys:
            avg_x = sum(xs) / len(xs)
            avg_y = sum(ys) / len(ys)
            file_lower = file.lower()
            # ファイル名から薬剤種を判定
            for key in ["gen", "tri", "cip"]:
                if key in file_lower:
                    drug_data[key]["xs"].append(avg_x)
                    drug_data[key]["ys"].append(avg_y)
                    break
        else:
            print(f"CSVファイル {file} から十分なデータが読み込めませんでした。")

    # 薬剤ごとの平均値と標準偏差を計算し、エラーバー付きでプロット
    drug_colors = {"gen": "orange", "cip": "blue", "tri": "green"}
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
    print(paths)
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
