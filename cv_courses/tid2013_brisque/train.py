import math

import numpy as np
import seaborn as sns
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def main():
    dataset = load_dataset("xmba15/TID2013")["train"]

    all_indices = range(len(dataset))
    scores = np.array([e["score"] for e in dataset])

    max_range = math.ceil(max(scores))
    print(max(scores))

    train_indices, test_indices = train_test_split(
        all_indices,
        stratify=np.digitize(
            scores,
            np.linspace(
                0,
                max_range,
                num=max_range + 1,
            ),
        ),
        random_state=2024,
        test_size=0.15,
    )

    scores = scores[train_indices]
    scores = scores[test_indices]

    sns.histplot(scores, kde=True)
    plt.title("Distribution Plot of Scores")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig("scores_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
