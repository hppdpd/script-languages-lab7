import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # зберігаємо графіки у файли, вікна не потрібні
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lab7.models import BaseModel, ForestModel, LogisticModel

OUT = Path("output")


@dataclass
class Result:
    model: BaseModel
    fit_time: float
    predict_time: float
    y_pred: np.ndarray


def evaluate(
    model: BaseModel, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> Result:
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = model.predict(x_test)
    predict_time = time.perf_counter() - t0
    return Result(model, fit_time, predict_time, y_pred)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load_breast_cancer()
    x, y = data.data, data.target
    names = [str(n) for n in data.target_names]  # 0 = malignant, 1 = benign

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Датасет: {x.shape[0]} зразків, {x.shape[1]} ознак, класи: {names}")
    print(f"Навчальна вибірка: {len(x_train)}, тестова: {len(x_test)}")

    # LSP: обидві моделі обробляються однаковим кодом
    models: list[BaseModel] = [LogisticModel(), ForestModel()]
    results = [evaluate(m, x_train, y_train, x_test) for m in models]

    print("\n" + "=" * 60)
    for r in results:
        print(f"\n--- {r.model.name} ---")
        print(classification_report(y_test, r.y_pred, target_names=names, digits=4))

    print("=" * 60)
    header = f"{'Модель':<22}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}"
    print(header + f"{'fit, c':>10}{'predict, c':>12}")
    for r in results:
        p, rec, f1, _ = precision_recall_fscore_support(
            y_test, r.y_pred, average="macro"
        )
        acc = accuracy_score(y_test, r.y_pred)
        print(
            f"{r.model.name:<22}{acc:>8.4f}{p:>8.4f}{rec:>8.4f}{f1:>8.4f}"
            f"{r.fit_time:>10.4f}{r.predict_time:>12.4f}"
        )
    print("(Prec, Recall, F1 — середнє macro по двох класах)")

    # Матриці помилок (heatmap)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    for ax, r in zip(axes, results, strict=True):
        ConfusionMatrixDisplay.from_predictions(
            y_test, r.y_pred, display_labels=names, ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(f"Confusion matrix: {r.model.name}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # Scatter plot: ознаки стискаємо до 2 компонент (PCA)
    pca = make_pipeline(StandardScaler(), PCA(n_components=2)).fit(x_train)
    coords = np.asarray(pca.transform(x_test))
    colors = {0: "tab:red", 1: "tab:green"}

    def scatter(
        ax: plt.Axes, labels: np.ndarray, title: str, wrong: np.ndarray | None
    ) -> None:
        for cls in (0, 1):
            m = labels == cls
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                c=colors[cls],
                s=25,
                alpha=0.7,
                label=names[cls],
            )
        if wrong is not None and wrong.any():
            ax.scatter(
                coords[wrong, 0],
                coords[wrong, 1],
                marker="x",
                c="black",
                s=80,
                label="помилка",
            )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()

    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    scatter(axes[0], y_test, "Справжні класи", None)
    for ax, r in zip(axes[1:], results, strict=True):
        wrong = r.y_pred != y_test
        title = f"Передбачено: {r.model.name} (помилок: {int(wrong.sum())})"
        scatter(ax, r.y_pred, title, wrong)
    fig.tight_layout()
    fig.savefig(OUT / "scatter.png", dpi=150)
    plt.close(fig)
    print(f"\nГрафіки збережено в {OUT}/: confusion_matrices.png, scatter.png")


if __name__ == "__main__":
    main()
