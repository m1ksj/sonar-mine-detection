from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIGURE_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results_tables")


def save_bar(counts, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close(fig)


def plot_category_distribution(manifest):
    counts = manifest["image_category"].value_counts()

    save_bar(
        counts,
        "Overall image-category distribution",
        "Number of images",
        "overall_image_category_distribution.png",
    )


def plot_year_category_distribution(manifest):
    table = pd.crosstab(manifest["year"], manifest["image_category"])

    fig, ax = plt.subplots(figsize=(8, 4))
    table.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title("Per-year image-category distribution")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Year")

    fig.tight_layout()
    output_path = FIGURE_DIR / "per_year_image_category_distribution.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_year_resolution_distribution(manifest):
    table = pd.crosstab(manifest["year"], manifest["resolution"])

    fig, ax = plt.subplots(figsize=(8, 4))
    table.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title("Images per year split by original resolution")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Year")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "year_resolution_distribution.png", dpi=200)
    plt.close(fig)


def plot_bbox_distribution(boxes):
    fig, ax = plt.subplots(figsize=(6, 5))

    for class_name, group in boxes.groupby("class_name"):
        ax.scatter(
            group["box_width_px"],
            group["box_height_px"],
            alpha=0.6,
            label=class_name,
        )

    ax.set_title("Bounding-box width versus height")
    ax.set_xlabel("Bounding-box width (px)")
    ax.set_ylabel("Bounding-box height (px)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bbox_width_height.png", dpi=200)
    plt.close(fig)


def save_summary(manifest, boxes):
    summary = pd.DataFrame(
        [
            {
                "n_images": len(manifest),
                "n_boxes": len(boxes),
                "n_empty": int((manifest["image_category"] == "empty").sum()),
                "n_milco_only": int(
                    (manifest["image_category"] == "milco_only").sum()
                ),
                "n_nombo_only": int(
                    (manifest["image_category"] == "nombo_only").sum()
                ),
                "n_mixed": int((manifest["image_category"] == "mixed").sum()),
            }
        ]
    )

    summary.to_csv(RESULTS_DIR / "eda_summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv("data/metadata/dataset_manifest.csv")
    boxes = pd.read_csv("reports/results_tables/bbox_summary.csv")

    plot_category_distribution(manifest)
    plot_year_category_distribution(manifest)
    plot_year_resolution_distribution(manifest)
    plot_bbox_distribution(boxes)
    save_summary(manifest, boxes)

    print("EDA plots saved in reports/figures/")


if __name__ == "__main__":
    main()
