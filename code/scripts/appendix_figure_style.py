"""Shared presentation conventions for the appendix figures (no data changes)."""
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, PURPLE = '#2468A0', '#BF611C', '#288167', '#8A4E98'
GRAY, DARK = '#777777', '#333333'
PALETTE = [BLUE, ORANGE, GREEN, PURPLE]
LINEWIDTH, MARKERSIZE = 1.15, 3.5


def apply_style():
    return plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5, 'legend.fontsize': 7, 'axes.linewidth': .6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': LINEWIDTH, 'lines.markersize': MARKERSIZE,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': None})


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(axis='y', color='#D8DEE5', linewidth=.45)
    ax.tick_params(length=2.5, pad=2, labelsize=7.5)
    ax.minorticks_off()
    ax.spines[['top', 'right']].set_visible(False)
