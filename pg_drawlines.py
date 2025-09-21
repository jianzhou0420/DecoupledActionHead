import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x) + 0.1 * np.random.randn(100)
y2 = np.cos(x) + 0.1 * np.random.randn(100)

plt.figure(figsize=(3.5, 2.6))
plt.plot(x, y1, label="Method A", color="C0", linewidth=2)
plt.plot(x, y2, label="Method B", color="C1", linestyle="--", linewidth=2)

plt.fill_between(x, y1 - 0.2, y1 + 0.2, color="C0", alpha=0.2)
plt.fill_between(x, y2 - 0.2, y2 + 0.2, color="C1", alpha=0.2)

plt.xlabel("Training Steps (×10³)")
plt.ylabel("Success Rate (%)")
plt.legend(frameon=False, fontsize=8)
plt.grid(alpha=0.3)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("icra_pretty_curve.pdf")
plt.show()
