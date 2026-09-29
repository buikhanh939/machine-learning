#bài tập 2:
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-8, 8, 400)
y = sigmoid(z)

plt.plot(z, y, label="Sigmoid")
plt.axhline(0.5, linestyle="--", label="y = 0.5")
plt.axvline(0, linestyle="--", label="z = 0")

plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.title("Ham sigmoid")
plt.legend()
plt.grid(True)

plt.savefig("figures/sigmoid.png")
plt.show()
