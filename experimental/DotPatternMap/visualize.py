import os
import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np


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
    import csv
    import matplotlib.pyplot as plt
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
        plt.colorbar(sc, ax=ax, label="IbpA-GFP Intensity (normalized)")
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
    if not file_names:
        print("指定されたディレクトリにCSVファイルが見つかりませんでした。")
        return

    labels: list[str] = []
    avg_xs: list[float] = []
    avg_ys: list[float] = []

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
            avg_xs.append(avg_x)
            avg_ys.append(avg_y)
            labels.append(file.replace(".csv", ""))
        else:
            print(f"CSVファイル {file} からデータが読み込めませんでした。")

    if not avg_xs or not avg_ys:
        print("有効なデータがありませんでした。")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(avg_xs, avg_ys, s=100, c="red", label="Average Position")
    for i, label in enumerate(labels):
        ax.annotate(
            label, (avg_xs[i], avg_ys[i]), textcoords="offset points", xytext=(5, 5)
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
        f"experimental/DotPatternMap/images/{i}"
        for i in os.listdir(csv_directory)
        if i.endswith(".csv")
    ]
    for path in paths:
        print(f"CSVファイル {path} から散布図を作成しています...")
        print(path)
        output = path.replace(".csv", ".png")
        plot_combined_dot_locations_from_csv(path, output)
        print(f"散布図を {output} に保存しました。")
