#!/usr/bin/env python3
"""
Render Forest/Dot Plot of all 10 matched comparisons in Table 3
==============================================================
Plots the empirical effect sizes (delta in Exact Match %) with:
- 25k multi-seed error bars (range of deltas across seeds 42, 43, 44)
- Filled markers for statistically significant deltas (p < 0.05)
- Hollow markers for non-significant deltas
- Clean banding for the two intervention mechanisms
- Zero line reference
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Use clean font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

# Data items (ordered from top to bottom)
# Each comparison: (label, tam_delta, tam_p, tam_err, mal_delta, mal_p, mal_err)
# Y positions:
# Block 1: Auxiliary Decoder Loss
# 25k: y = 8
# 1.0M: y = 6.5
# Block 2: Phonotactic String Tags
# 250k BiGRU: y = 4.5
# 250k Transformer: y = 3
# 500k Transformer: y = 1.5

tam_color = '#1f77b4'  # Deep blue
mal_color = '#d95f02'  # Rust orange

# Data definitions
# 25k deltas: Tam runs = [-0.20, +0.17, -0.39] -> mean -0.14, min -0.39, max +0.17 -> err = [0.25, 0.31]
# Mal runs = [+0.04, +0.49, -0.09] -> mean +0.15, min -0.09, max +0.49 -> err = [0.24, 0.34]

comparisons = [
    {
        "y": 7.8,
        "name": "25k (3 seeds)",
        "tam": {"delta": -0.14, "p": 0.585, "err": [[0.25], [0.31]]},
        "mal": {"delta": 0.15, "p": 0.901, "err": [[0.24], [0.34]]},
        "offset": 0.18
    },
    {
        "y": 6.2,
        "name": "1.0M (bilingual)",
        "tam": {"delta": -2.33, "p": 0.0003, "err": None},
        "mal": {"delta": -1.37, "p": 0.0295, "err": None},
        "offset": 0.18
    },
    {
        "y": 4.0,
        "name": "250k (BiGRU)",
        "tam": {"delta": 1.53, "p": 0.0183, "err": None},
        "mal": {"delta": 1.08, "p": 0.0893, "err": None},
        "offset": 0.18
    },
    {
        "y": 2.6,
        "name": "250k (Transformer)",
        "tam": {"delta": 0.77, "p": 0.2371, "err": None},
        "mal": {"delta": -0.10, "p": 0.8689, "err": None},
        "offset": 0.18
    },
    {
        "y": 1.2,
        "name": "500k (Transformer)",
        "tam": {"delta": -0.22, "p": 0.7358, "err": None},
        "mal": {"delta": -0.54, "p": 0.3928, "err": None},
        "offset": 0.18
    },
]

# Draw horizontal connecting guidelines
for item in comparisons:
    y = item["y"]
    off = item["offset"]
    
    # Tamil point
    yt = y + off
    dt = item["tam"]["delta"]
    pt = item["tam"]["p"]
    err_t = item["tam"]["err"]
    ax.plot([0, dt], [yt, yt], color=tam_color, alpha=0.5, lw=1.2, zorder=2)
    if err_t is not None:
        ax.errorbar(dt, yt, xerr=err_t, color=tam_color, fmt='none', capsize=0, lw=2.0, zorder=3)
    if pt < 0.05:
        ax.scatter(dt, yt, color=tam_color, s=70, zorder=4)
    else:
        ax.scatter(dt, yt, facecolor='white', edgecolor=tam_color, lw=2.0, s=70, zorder=4)
        
    # Malayalam point
    ym = y - off
    dm = item["mal"]["delta"]
    pm = item["mal"]["p"]
    err_m = item["mal"]["err"]
    ax.plot([0, dm], [ym, ym], color=mal_color, alpha=0.5, lw=1.2, zorder=2)
    if err_m is not None:
        ax.errorbar(dm, ym, xerr=err_m, color=mal_color, fmt='none', capsize=0, lw=2.0, zorder=3)
    if pm < 0.05:
        ax.scatter(dm, ym, color=mal_color, s=70, zorder=4)
    else:
        ax.scatter(dm, ym, facecolor='white', edgecolor=mal_color, lw=2.0, s=70, zorder=4)

# Vertical zero reference line
ax.axvline(0, color='#666666', lw=1.0, linestyle='-', zorder=1)

# Annotations & Section Headers
ax.text(-3.05, 8.8, r"$\it{Auxiliary\ phonotactic\ loss\ on\ the\ decoder\ (\lambda = 0.3)}$", fontsize=9.5, color='#222222')
ax.text(-3.05, 5.0, r"$\it{Phonotactic\ tags\ in\ the\ output\ string}$", fontsize=9.5, color='#222222')

# Direct Labels on BiGRU points for clarity
ax.text(1.58, 4.0 + 0.18, " Tamil", color=tam_color, va='center', fontsize=9.5, fontweight='medium')
ax.text(1.18, 4.0 - 0.25, "Malayalam", color=mal_color, va='center', fontsize=9.5, fontweight='medium')

# Y-axis ticks and labels
y_ticks = [item["y"] for item in comparisons]
y_labels = [item["name"] for item in comparisons]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=9.5)

# Axis limits & X label
ax.set_xlim(-3.1, 2.3)
ax.set_ylim(0.4, 9.4)
ax.set_xlabel("Change in exact-match accuracy from phonological supervision (points)", fontsize=10.5, labelpad=8)

# Title & Directional prompt
ax.set_title("One significant gain and two significant losses across ten matched comparisons", fontsize=11, loc='left', pad=10, fontweight='medium')
ax.text(2.28, 9.15, r"$\text{phonology helps} \rightarrow$", fontsize=8.5, color='#666666', ha='right')

# Custom Legend in bottom left
legend_y = 1.0
ax.scatter(-2.9, 0.9, color='#444444', s=60, zorder=5)
ax.text(-2.75, 0.9, r"$p < 0.05$", fontsize=8.5, va='center', color='#333333')
ax.scatter(-2.9, 0.55, facecolor='white', edgecolor='#444444', lw=1.8, s=60, zorder=5)
ax.text(-2.75, 0.55, "not significant", fontsize=8.5, va='center', color='#333333')

# Clean spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')

plt.tight_layout()

# Save figure to docs/revised2 and artifacts
out_docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "revised2", "fig_effects_corrected.png")
out_scaling = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "revised2", "fig_scaling_corrected.png")
out_artifacts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "fig_effects_corrected.png")

plt.savefig(out_docs, bbox_inches='tight')
plt.savefig(out_scaling, bbox_inches='tight')
plt.savefig(out_artifacts, bbox_inches='tight')
print(f"Saved: {out_docs}")
print(f"Saved: {out_scaling}")
