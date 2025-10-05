from pathlib import Path
import argparse, urllib.request

DATA = {
    "pima": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
        "dst": "pima-indians-diabetes.data.csv",
    },
    "transfusion": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data",
        "dst": "transfusion.csv",
    },
    "heart": {
        "url": "https://raw.githubusercontent.com/amirhossein-sys/heart-disease-uci/master/heart.csv",
        "dst": "heart.csv",
    },
}

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)

def fetch_one(key: str):
    meta = DATA[key]
    dst = RAW / meta["dst"]
    print(f"→ Downloading {key} → {dst}")
    urllib.request.urlretrieve(meta["url"], dst)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dataset", choices=list(DATA), nargs="?")
    args = ap.parse_args()
    if args.all:
        for k in DATA: fetch_one(k)
    else:
        fetch_one(args.dataset)
