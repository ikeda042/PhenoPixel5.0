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


# 使用例:
# plot_combined_dot_locations_from_csv("experimental/DotPatternMap/images/dbName_dot_positions.csv",
#                                        "experimental/DotPatternMap/images/dbName_combined_dot_locations_from_csv.png")
if __name__ == "__main__":
    path = "experimental/DotPatternMap/images/sk326tri30min_dot_positions.csv"
    output = "experimental/DotPatternMap/images/sk326tri30min_combined_dot_locations_from_csv.png"
    plot_combined_dot_locations_from_csv(path, output)
    print(f"Saved the combined dot locations graph to {output}")
