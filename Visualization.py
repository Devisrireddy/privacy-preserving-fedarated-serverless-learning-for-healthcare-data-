import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("evaluation_results.csv")

# --- Plot Accuracy vs Rounds ---
plt.figure(figsize=(8,6))
for setup in df["setup"].unique():
    subset = df[df["setup"] == setup]
    plt.plot(subset["round"], subset["accuracy"], marker="o", label=setup)

plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)")
plt.title("Federated Learning Evaluation")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_rounds.png")
plt.show()

# --- Final Accuracy Bar Chart ---
plt.figure(figsize=(8,6))
final_acc = df.groupby("setup")["accuracy"].max().reset_index()
plt.bar(final_acc["setup"], final_acc["accuracy"], color="skyblue")
plt.xticks(rotation=20)
plt.ylabel("Final Accuracy (%)")
plt.title("Final Accuracy Comparison")
plt.savefig("final_accuracy.png")
plt.show()
