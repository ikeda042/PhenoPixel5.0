def plot_average_dot_location_from_csv(csv_path: str, output_path: str) -> None:
    """
    CSVファイルから各ドットのRelX, RelYを読み込み、各座標の平均値を計算して、
    平均点をプロットするグラフ（散布図）を生成して指定のパスに保存します。

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
        csv_path (str): 入力となるCSVファイルのパス。
        output_path (str): 作成するグラフ画像の保存先パス。
    """
    import csv
    import matplotlib.pyplot as plt

    xs: list[float] = []
    ys: list[float] = []

    with open(csv_path, mode="r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader, None)  # ヘッダー行をスキップ
        for row in csv_reader:
            try:
                # CSVは [RelX, RelY, Brightness] としているため、brightnessは無視
                x, y, _ = map(float, row)
                xs.append(x)
                ys.append(y)
            except ValueError:
                continue

    if not xs or not ys:
        print(f"CSVファイル {csv_path} からデータが読み込めませんでした。")
        return

    avg_x = sum(xs) / len(xs)
    avg_y = sum(ys) / len(ys)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(avg_x, avg_y, color="red", s=100, label="Average Position")
    ax.set_title("Average Dot Location")
    ax.set_xlabel("Rel. X (normalized)")
    ax.set_ylabel("Rel. Y (normalized)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cv_dot_location_from_csv(csv_path: str, output_path: str) -> None:
    """
    CSVファイルから各ドットのRelX, RelYを読み込み、各座標の変動係数（Coefficient of Variation）を計算して、
    変動係数をプロットするグラフ（棒グラフ）を生成して指定のパスに保存します。

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
        csv_path (str): 入力となるCSVファイルのパス。
        output_path (str): 作成するグラフ画像の保存先パス。
    """
    import csv
    import matplotlib.pyplot as plt
    import statistics

    xs: list[float] = []
    ys: list[float] = []

    with open(csv_path, mode="r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader, None)  # ヘッダー行をスキップ
        for row in csv_reader:
            try:
                # CSVは [RelX, RelY, Brightness] としているため、brightnessは無視
                x, y, _ = map(float, row)
                xs.append(x)
                ys.append(y)
            except ValueError:
                continue

    if len(xs) < 2 or len(ys) < 2:
        print(f"CSVファイル {csv_path} から十分なデータが読み込めませんでした。")
        return

    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    stdev_x = statistics.stdev(xs)
    stdev_y = statistics.stdev(ys)
    cv_x = stdev_x / mean_x if mean_x != 0 else 0
    cv_y = stdev_y / mean_y if mean_y != 0 else 0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(["Rel. X", "Rel. Y"], [cv_x, cv_y], color=["blue", "green"])
    ax.set_title("Coefficient of Variation of Dot Locations")
    ax.set_ylabel("Coefficient of Variation")
    ax.grid(True, axis="y")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import os

    # CSVファイルが保存されているディレクトリ（例: experimental/DotPatternMap/images）
    csv_dir = "experimental/DotPatternMap/images"
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    print(csv_files)
    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        avg_output_path = os.path.join(csv_dir, csv_file.replace(".csv", "_avg.png"))
        cv_output_path = os.path.join(csv_dir, csv_file.replace(".csv", "_cv.png"))
        plot_average_dot_location_from_csv(csv_path, avg_output_path)
        plot_cv_dot_location_from_csv(csv_path, cv_output_path)
