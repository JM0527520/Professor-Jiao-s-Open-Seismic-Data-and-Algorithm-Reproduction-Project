"""Consistent three-way seismic/reference/prediction views in array coordinates."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if any(f.name == font for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [font]
        break
plt.rcParams["axes.unicode_minus"] = False


def three_views(path: Path, seismic: np.ndarray, labels: np.ndarray, title: str,
                *, prediction: np.ndarray | None = None, indices: tuple[int, int, int] | None = None,
                offsets=(0, 0, 0)) -> None:
    if seismic.shape != labels.shape or (prediction is not None and prediction.shape != labels.shape):
        raise ValueError("Plot arrays must share the same shape")
    if indices is None:
        indices = tuple(s // 2 for s in seismic.shape)
    nonzero = np.abs(seismic[seismic != 0])
    limit = max(float(np.quantile(nonzero, 0.99)) if nonzero.size else 1.0, 1e-6)
    columns = 4 if prediction is not None else 2
    fig, axes = plt.subplots(3, columns, figsize=(4.6 * columns, 12), constrained_layout=True)
    specs = [
        (seismic[indices[0]], labels[indices[0]], None if prediction is None else prediction[indices[0]],
         "水平切片", "横向轴3 / 分区剖面索引", "横向轴2索引", 2, 1, indices[0] + offsets[0]),
        (seismic[:, :, indices[2]], labels[:, :, indices[2]], None if prediction is None else prediction[:, :, indices[2]],
         "固定轴3的垂向剖面", "横向轴2索引", "垂向采样点", 1, 0, indices[2] + offsets[2]),
        (seismic[:, indices[1], :], labels[:, indices[1], :], None if prediction is None else prediction[:, indices[1], :],
         "固定轴2的垂向剖面", "横向轴3 / 分区剖面索引", "垂向采样点", 2, 0, indices[1] + offsets[1]),
    ]
    for row, (image, truth, pred, view, xlabel, ylabel, xaxis, yaxis, fixed) in enumerate(specs):
        extent = (offsets[xaxis] - 0.5, offsets[xaxis] + image.shape[1] - 0.5,
                  offsets[yaxis] + image.shape[0] - 0.5, offsets[yaxis] - 0.5)
        for col in range(columns):
            ax = axes[row, col]
            if col == 2 and pred is not None:
                im = ax.imshow(pred, cmap="magma", vmin=0, vmax=1, aspect="auto", extent=extent, interpolation="nearest")
                fig.colorbar(im, ax=ax, shrink=0.75, label="断层概率")
            else:
                ax.imshow(image, cmap="gray", vmin=-limit, vmax=limit, aspect="auto", extent=extent, interpolation="nearest")
                if col == 1:
                    ax.imshow(np.ma.masked_where(~truth.astype(bool), truth), cmap="autumn", vmin=0, vmax=1,
                              alpha=0.8, aspect="auto", extent=extent, interpolation="nearest")
                elif col == 3:
                    if truth.any() and not truth.all():
                        ax.contour(truth, levels=[0.5], colors="lime", linewidths=0.55, extent=extent, origin="upper")
                    if (pred >= 0.5).any() and not (pred >= 0.5).all():
                        ax.contour(pred, levels=[0.5], colors="magenta", linewidths=0.55, extent=extent, origin="upper")
            labels_col = ["地震振幅", "地震＋参考标签", "预测概率", "参考绿 / 预测紫（阈值0.5）"]
            ax.set_title(f"{view} [{fixed}]\n{labels_col[col]}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
    fig.suptitle(title + "\n坐标为数组索引；图中零标签不等同于已确认无断层", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def full_section(path: Path, seismic: np.ndarray, labels: np.ndarray, title: str) -> None:
    limit = max(float(np.quantile(np.abs(seismic[seismic != 0]), 0.99)), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax in axes:
        ax.imshow(seismic.T, cmap="gray", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
        ax.set_xlabel("数组横向轴2索引")
        ax.set_ylabel("垂向采样点")
    axes[0].set_title("完整振幅剖面（含补零范围）")
    axes[1].imshow(np.ma.masked_where(~labels.T, labels.T), cmap="autumn", vmin=0, vmax=1,
                   alpha=0.8, aspect="auto", interpolation="nearest")
    axes[1].set_title("参考断层叠加；空白区不表示完整负标注")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
