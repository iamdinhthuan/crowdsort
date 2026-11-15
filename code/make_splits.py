"""Hold out the development splits used in the paper from the training manifests.

AISHELL-4: every 13th training session (index 5 mod 13 after sorting by path) -> aishell4_dev, rest -> aishell4_trainsub.
MSDWild:   250 training clips drawn with random.seed(11) after sorting by path -> msdwild_fewdev, rest -> msdwild_train_v2.
The resulting IDs are listed in data_lists/.
"""
import json
import os
import random

CROWDSORT_ROOT = os.environ.get("CROWDSORT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M = f"{CROWDSORT_ROOT}/data/manifests"


def load(name):
    rows = [json.loads(l) for l in open(f"{M}/{name}.jsonl")]
    return sorted(rows, key=lambda x: x["audio_filepath"])


def save(name, rows):
    with open(f"{M}/{name}.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)


def main():
    a = load("aishell4_train")
    save("aishell4_dev", [x for i, x in enumerate(a) if i % 13 == 5])
    save("aishell4_trainsub", [x for i, x in enumerate(a) if i % 13 != 5])
    random.seed(11)
    m = load("msdwild_train")
    idx = list(range(len(m)))
    random.shuffle(idx)
    dev = set(idx[:250])
    save("msdwild_train_v2", [x for i, x in enumerate(m) if i not in dev])
    save("msdwild_fewdev", [x for i, x in enumerate(m) if i in dev])


if __name__ == "__main__":
    main()
