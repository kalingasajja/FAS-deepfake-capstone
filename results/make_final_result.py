import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data (AUC on ALL-seen test subset) ──────────────────────────────────────
models      = ["Classical\n(HGB)\nNo Aug", "Classical\n(HGB)\nWith Aug",
               "CNN\n(EfficientNet)\nNo Aug", "CNN\n(EfficientNet)\nWith Aug"]
clean_auc   = [0.765, 0.756, 0.842, 0.836]
rc_seen_auc = [0.478, 0.538, 0.496, 0.575]
rc_unsn_auc = [0.506, 0.573, 0.524, 0.525]

x = np.arange(len(models))
w = 0.26

C_CLEAN  = "#4A90D9"
C_SEEN   = "#E8A838"
C_UNSEEN = "#D94A4A"
C_RANDOM = "#888888"

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#161B22")

# ── Bars ──────────────────────────────────────────────────────────────────────
b1 = ax.bar(x - w, clean_auc,   w, color=C_CLEAN,  zorder=3,
            label="Clean digital video", edgecolor="#0F1117", linewidth=0.8)
b2 = ax.bar(x,     rc_seen_auc, w, color=C_SEEN,   zorder=3,
            label="Screen-replay (seen settings)", edgecolor="#0F1117", linewidth=0.8)
b3 = ax.bar(x + w, rc_unsn_auc, w, color=C_UNSEEN, zorder=3,
            label="Screen-replay (unseen settings)", edgecolor="#0F1117", linewidth=0.8)

# ── Random-chance line ────────────────────────────────────────────────────────
ax.axhline(0.50, color=C_RANDOM, linewidth=1.4, linestyle="--", zorder=4,
           label="Random chance (AUC = 0.50)")
ax.text(3.75, 0.503, "Random chance", color=C_RANDOM, fontsize=8.5, va="bottom", ha="right")

# ── Value labels on bars ──────────────────────────────────────────────────────
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.2f}", ha="center", va="bottom",
                fontsize=8.5, color="white", fontweight="bold", zorder=6)

# ── Annotation 1: best clean performance ─────────────────────────────────────
#   Arrow from top-left text box → top of CNN-No-Aug blue bar
ax.annotate(
    "Best clean performance:\n   AUC = 0.84",
    xy=(2 - w, 0.842),         # arrowhead at bar top
    xytext=(0.55, 0.935),      # text box top-left area — far from centre
    fontsize=9, color="#4A90D9", fontweight="bold", ha="center",
    arrowprops=dict(arrowstyle="-|>", color="#4A90D9", lw=1.5,
                    connectionstyle="arc3,rad=-0.20"),
    bbox=dict(boxstyle="round,pad=0.35", fc="#0d1d35", ec="#4A90D9", alpha=0.92)
)

# ── Annotation 2: collapses on screen-replay ─────────────────────────────────
#   Text box placed in upper-right quadrant, arrow pointing down to CNN rc_seen bar
ax.annotate(
    "⚠ Collapses to near-random\n   on screen-replay!",
    xy=(2, 0.496),             # arrowhead at CNN rc_seen bar top
    xytext=(3.05, 0.80),       # text placed far right — no overlap
    fontsize=9, color="#E8A838", fontweight="bold", ha="left",
    arrowprops=dict(arrowstyle="-|>", color="#E8A838", lw=1.5,
                    connectionstyle="arc3,rad=0.40"),
    bbox=dict(boxstyle="round,pad=0.4", fc="#2a2200", ec="#E8A838", alpha=0.95)
)

# ── Annotation 3: augmentation helps ─────────────────────────────────────────
#   Small label placed directly above the CNN-aug orange bar
ax.annotate(
    "+7.9% AUC\nAug helps ↑",
    xy=(3, 0.575),             # arrowhead at CNN-aug rc_seen bar top
    xytext=(3.50, 0.68),       # text placed lower-right of annotation 2
    fontsize=9, color="#50FA7B", fontweight="bold", ha="right",
    arrowprops=dict(arrowstyle="-|>", color="#50FA7B", lw=1.5,
                    connectionstyle="arc3,rad=-0.25"),
    bbox=dict(boxstyle="round,pad=0.35", fc="#0d2a0d", ec="#50FA7B", alpha=0.95)
)

# ── Vertical divider between Classical and CNN ────────────────────────────────
ax.axvline(1.5, color="#ffffff22", linewidth=1, linestyle=":")
ax.text(0.75, 1.008, "Classical ML", ha="center", fontsize=10,
        color="#aaaaaa", transform=ax.get_xaxis_transform())
ax.text(2.75, 1.008, "Deep Learning (CNN)", ha="center", fontsize=10,
        color="#aaaaaa", transform=ax.get_xaxis_transform())

# ── Axes styling ──────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(models, color="white", fontsize=10)
ax.set_ylabel("AUC  (Area Under ROC Curve)", color="white", fontsize=11)
ax.set_ylim(0.35, 1.07)
ax.tick_params(colors="white", labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#333344")
ax.yaxis.grid(True, color="#ffffff18", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(loc="upper left", framealpha=0.3, facecolor="#1a1a2e",
          edgecolor="#555566", labelcolor="white", fontsize=9.5,
          bbox_to_anchor=(0.01, 0.99))

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Deepfake Detection Performance Under Screen-Replay Attack\n"
    "FaceForensics++ Dataset  ·  AUC on ALL test methods  ·  Higher is Better",
    color="white", fontsize=12.5, fontweight="bold", pad=15, loc="left")

# ── Key finding footer ────────────────────────────────────────────────────────
finding = (
    "KEY FINDING:  When trained only on clean digital video, both models collapse to near-random chance (AUC ≈ 0.50) on screen-replayed faces.\n"
    "Recapture augmentation partially recovers performance — but a large gap remains, motivating the proposed ViT + rPPG architecture."
)
fig.text(0.5, 0.01, finding, ha="center", va="bottom", fontsize=9, color="#cccccc",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a2035",
                   edgecolor="#4466aa", alpha=0.95))

plt.tight_layout(rect=[0, 0.10, 1, 1])
out = "results/final_result.png"
plt.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print(f"Saved → {out}")
