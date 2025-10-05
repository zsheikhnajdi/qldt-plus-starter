
# No top-level execution here. Everything is callable from run_qldt.py.

import os, shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Digraph
import graphviz
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Windows Graphviz PATH helper (still here to keep behavior identical) ---
if not shutil.which("dot"):
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


# ---------- small helper: resolve dataset path gracefully ----------
def _resolve_file(name_or_path: str):
    """
    Try to locate a dataset by checking common places:
    - exact path if exists
    - repo root
    - data/raw/
    Fallback to provided string.
    """
    p = Path(name_or_path)
    if p.exists():
        return str(p)
    for cand in [Path(".")/name_or_path, Path("data/raw")/name_or_path]:
        if cand.exists():
            return str(cand)
    return name_or_path  # let pandas raise if not found


# =========================
# Step 1: Load & preprocess
# =========================

def load_transfusion_data():
    # Your original expectation was "transfusion.csv" sitting next to the script.
    csv_path = _resolve_file("transfusion.csv")
    df = pd.read_csv(csv_path)
    df.rename(columns={"whether he/she donated blood in March 2007": "Class"}, inplace=True)

    # Keep your exact sampling sizes for balance (178/178)
    df_pos = df[df["Class"] == 1].sample(n=178, random_state=123)
    df_neg = df[df["Class"] == 0].sample(n=178, random_state=123)
    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=77).reset_index(drop=True)

    X = df_balanced.drop(columns=["Class"])
    y = df_balanced["Class"]

    # MinMax -> rank -> normalize to (0,1] like your QLDT+ input
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_ranked = X_scaled.rank(method="max", axis=0)
    X_scaled = (X_ranked / len(X_scaled)).to_numpy()

    return train_test_split(X_scaled, y.to_numpy(), test_size=0.3, random_state=42, stratify=y)


def load_pima_data():
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    # Keep your absolute path as a fallback; but first try common relative paths
    candidates = [
        "pima-indians-diabetes.data.csv",
        Path("data/raw")/"pima-indians-diabetes.data.csv",
        r"c:\MyData\MasterAI\Thesis\implementation\pima-indians-diabetes.data.csv",  # your original
    ]
    path = None
    for c in candidates:
        if Path(c).exists():
            path = c
            break
    if path is None:
        # last resort uses your absolute path; pandas will error clearly if missing.
        path = candidates[-1]

    df = pd.read_csv(path, header=None, names=columns)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # 🔍 Optional balancing (kept as-is)
    min_class = min(sum(y == 0), sum(y == 1))
    df_balanced = pd.concat([
        df[df["Outcome"] == 1].sample(n=min_class, random_state=77),
        df[df["Outcome"] == 0].sample(n=min_class, random_state=77)
    ]).sample(frac=1, random_state=77)

    X = df_balanced.drop(columns=["Outcome"])
    y = df_balanced["Outcome"]

    # Normalization and ranking
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_ranked = X_scaled.rank(method="max", axis=0)
    X_scaled = (X_ranked / len(X_scaled)).to_numpy()

    return train_test_split(X_scaled, y.to_numpy(), test_size=0.3, random_state=42, stratify=y)


def load_heart_data():
    columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
               'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
               'Oldpeak', 'ST_Slope', 'HeartDisease']

    csv_path = _resolve_file("heart.csv")
    df = pd.read_csv(csv_path, names=columns, header=0)

    # Balance 0/1 equally (kept from your code)
    min_class = min(sum(df["HeartDisease"] == 0), sum(df["HeartDisease"] == 1))
    df_balanced = pd.concat([
        df[df["HeartDisease"] == 0].sample(n=min_class, random_state=77),
        df[df["HeartDisease"] == 1].sample(n=min_class, random_state=77)
    ]).sample(frac=1, random_state=77)

    # One-hot for categoricals
    X = pd.get_dummies(df_balanced.drop(columns=["HeartDisease"]))
    y = df_balanced["HeartDisease"]

    feature_names = X.columns.tolist()  # keep feature names

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_ranked = X_scaled.rank(method="max", axis=0)
    X_scaled = (X_ranked / len(X_scaled)).to_numpy()

    return train_test_split(X_scaled, y.to_numpy(), test_size=0.3, random_state=42, stratify=y), feature_names


# ==================
# QLDT+ core engine
# ==================

class QLDTNode:
    def __init__(self, is_leaf=False, prediction=None, feature_index=None, left=None, right=None, path=None, score=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature_index = feature_index
        self.left = left
        self.right = right
        self.path = path if path is not None else []
        self.score = score


def compute_rho(X, y, path):
    E = N = 0
    for xi, yi in zip(X, y):
        match = 1
        for j, polarity in path:
            xij = xi[j]
            match *= xij if polarity else (1 - xij)
        if yi == 1:
            E += match
        else:
            N += match
    return E / (E + N) if (E + N) != 0 else 0


def build_tree_qldt(X, y, path=[], used_features=set(), max_depth=10, depth=0, theta_p=0.5, tau=0.045):
    rho = compute_rho(X, y, path)
    if len(used_features) == X.shape[1] or depth >= max_depth:
        label = int(rho >= theta_p)
        return QLDTNode(is_leaf=True, prediction=label, path=path, score=rho)

    best_diff, best_attr = 0, None
    for j in range(X.shape[1]):
        if j in used_features:
            continue
        rho_pos = compute_rho(X, y, path + [(j, True)])
        rho_neg = compute_rho(X, y, path + [(j, False)])
        d = abs(rho_pos - rho_neg)
        if d > best_diff:
            best_diff, best_attr = d, j
            best_rho_pos, best_rho_neg = rho_pos, rho_neg

    if best_attr is None:
        label = int(rho >= theta_p)
        return QLDTNode(is_leaf=True, prediction=label, path=path, score=rho)

    r = X.shape[1] - len(used_features)
    rho_largest = 1 - (1 - max(best_rho_pos, best_rho_neg)) / r
    rho_smallest = min(best_rho_pos, best_rho_neg) / r

    if rho_largest <= theta_p:
        return QLDTNode(is_leaf=True, prediction=0, path=path, score=rho)
    if rho_smallest >= theta_p:
        return QLDTNode(is_leaf=True, prediction=1, path=path, score=rho)
    if abs(best_rho_pos - best_rho_neg) < tau:
        label = int(rho >= theta_p)
        return QLDTNode(is_leaf=True, prediction=label, path=path, score=rho)

    used_features_next = used_features.union({best_attr})
    left = build_tree_qldt(X, y, path + [(best_attr, True)], used_features_next, max_depth, depth + 1, theta_p, tau)
    right = build_tree_qldt(X, y, path + [(best_attr, False)], used_features_next, max_depth, depth + 1, theta_p, tau)

    return QLDTNode(is_leaf=False, feature_index=best_attr, left=left, right=right, path=path, score=rho)


def prune_zero_leaves(node):
    if node is None or node.is_leaf:
        return node if node and node.prediction == 1 else None
    node.left = prune_zero_leaves(node.left)
    node.right = prune_zero_leaves(node.right)
    if node.left is None and node.right is None:
        return None
    if node.left and node.right and node.left.is_leaf and node.right.is_leaf and node.left.prediction == node.right.prediction:
        return QLDTNode(is_leaf=True, prediction=node.left.prediction, path=node.path, score=node.score)
    return node


# =========
# Inference
# =========
def predict_single(x, node, theta_p=0.5):
    def evaluate_path(path):
        prod = 1
        for j, polarity in path:
            xj = x[j]
            prod *= xj if polarity else (1 - xj)
        return prod

    def gather_1_leaves(node):
        if node is None:
            return []
        if node.is_leaf and node.prediction == 1:
            return [node]
        return gather_1_leaves(node.left) + gather_1_leaves(node.right)

    total_score = sum(evaluate_path(leaf.path) for leaf in gather_1_leaves(node))
    return 1 if total_score >= theta_p else 0


def predict(X, tree, theta_p=0.5):
    return [predict_single(x, tree, theta_p) for x in X]


# ============
# Visualization
# ============
def draw_qldt_tree(tree, filename="qldt_tree"):
    dot = Digraph()
    def add_nodes(node, node_id):
        if node is None:
            return
        if node.is_leaf:
            label = f"Predict: {node.prediction}\n\u03c1 = {node.score:.2f}"
            dot.node(node_id, label, shape="box", style="filled", fillcolor="#DDFFDD")
        else:
            label = f"x[{node.feature_index}]\n\u03c1 = {node.score:.2f}"
            # label = f"x[{node.feature_index}] ≤ 0.5\n\u03c1 = {node.score:.2f}"
            dot.node(node_id, label, shape="ellipse", style="filled", fillcolor="#FFFFCC")
            left_id = f"{node_id}L"
            right_id = f"{node_id}R"
            add_nodes(node.left, left_id)
            add_nodes(node.right, right_id)
            dot.edge(node_id, left_id, label="Yes")
            dot.edge(node_id, right_id, label="No")
    add_nodes(tree, "root")
    dot.render(filename, format="png", cleanup=True)


def draw_cart_tree(model, feature_names, filename="cart_tree"):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(filename, format='png', cleanup=True)


# =========================
# Stats & grid search utils
# =========================
def get_tree_stats_qldt(tree):
    def traverse(node, depth=0):
        if node is None:
            return depth, 0, 0
        if node.is_leaf:
            return depth, 1, 1 if node.prediction == 1 else 0
        left_depth, left_leaves, left_ones = traverse(node.left, depth + 1)
        right_depth, right_leaves, right_ones = traverse(node.right, depth + 1)
        return max(left_depth, right_depth), left_leaves + right_leaves, left_ones + right_ones

    depth, leaves, one_leaves = traverse(tree)
    return {"depth": depth, "leaves": leaves, "positive_leaves": one_leaves}


def get_tree_stats_cart(model):
    depth = model.get_depth()
    leaves = model.get_n_leaves()
    return {"depth": depth, "leaves": leaves}


def run_param_grid_search_qldt(X_train, X_test, y_train, y_test):
    param_grid = {
        "tau": [0.01, 0.02, 0.03, 0.045, 0.06],
        "theta_p": [0.4, 0.5, 0.6, 0.7],
        "max_depth": [4, 6, 8, 10]
    }

    results = []

    for tau in param_grid["tau"]:
        for theta_p in param_grid["theta_p"]:
            for max_depth in param_grid["max_depth"]:
                print(f"➡️ Testing: tau={tau}, theta_p={theta_p}, max_depth={max_depth}")

                tree = build_tree_qldt(
                    X_train, y_train,
                    theta_p=theta_p,
                    tau=tau,
                    max_depth=max_depth
                )
                tree = prune_zero_leaves(tree)
                y_pred = predict(X_test, tree, theta_p=theta_p)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                stats = get_tree_stats_qldt(tree)

                results.append({
                    "tau": tau,
                    "theta_p": theta_p,
                    "max_depth": max_depth,
                    "depth": stats["depth"],
                    "leaves": stats["leaves"],
                    "positive_leaves": stats["positive_leaves"],
                    "accuracy": round(acc, 4),
                    "f1_score": round(f1, 4)
                })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="accuracy", ascending=False).reset_index(drop=True)

    # pick best row (your original tie-break order)
    best_rows = df_results.sort_values(
        by=["accuracy", "f1_score", "leaves", "depth", "max_depth", "tau"],
        ascending=[False, False, True, True, True, True]
    )
    best_row = best_rows.iloc[0]

    tree_best = build_tree_qldt(
        X_train, y_train,
        theta_p=best_row["theta_p"],
        tau=best_row["tau"],
        max_depth=int(best_row["max_depth"])
    )
    tree_best = prune_zero_leaves(tree_best)

    print(f"\n✅ Best QLDT+ parameters:")
    print(f"tau = {best_row['tau']}, theta_p = {best_row['theta_p']}, depth = {int(best_row['depth'])}, max_depth = {int(best_row['max_depth'])}, leaves = {int(best_row['leaves'])}")

    # Caller decides where to save; here we only return df
    return df_results, tree_best


# ========================
# Interpretability helpers
# ========================
def extract_positive_paths(tree):
    paths = []
    def dfs(node, current_path):
        if node is None:
            return
        if node.is_leaf and node.prediction == 1:
            paths.append((current_path.copy(), node.score))
        else:
            if node.left:
                dfs(node.left, current_path + [(node.feature_index, True)])
            if node.right:
                dfs(node.right, current_path + [(node.feature_index, False)])
    dfs(tree, [])
    return paths


def evaluate_path(path, x):
    prod = 1
    for j, polarity in path:
        xj = x[j]
        prod *= xj if polarity else (1 - xj)
    return prod


def get_feature_names_from_indices(path, feature_names):
    return [feature_names[i] for i, _ in path]


def build_interpretability_table(tree, X, y, feature_names, filename="qldt_interpretability_table"):
    paths = extract_positive_paths(tree)
    results = []
    for i, (path, rho) in enumerate(paths, start=1):
        matched = []
        correct = 0
        for xi, yi in zip(X, y):
            match = evaluate_path(path, xi)
            if match >= 0.5:
                matched.append((xi, yi))
                if yi == 1:
                    correct += 1
        support = len(matched) / len(X)
        accuracy = correct / len(matched) if matched else 0
        results.append({
            "Path ID": i,
            "Features Used": ", ".join(get_feature_names_from_indices(path, feature_names)),
            "Rule Length": len(path),
            "Tree Depth": len(path),
            "Support": round(support, 2),
            "Accuracy": round(accuracy, 2),
            "Avg Rho (ρ)": round(rho, 2)
        })
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(22, len(df)*0.8 + 2))
    ax.axis('off')
    tbl = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.title("QLDT+ Interpretability Metrics", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close()
    return df


# =============================
# Feature importance (QLDT+)
# =============================
def compute_qldt_feature_importance(tree, feature_names, filename="qldt_feature_importance"):
    importance_dict = defaultdict(float)

    def traverse(node):
        if node is None or node.is_leaf:
            return
        if node.left and node.right and hasattr(node.left, 'score') and hasattr(node.right, 'score'):
            importance_dict[node.feature_index] += abs(node.left.score - node.right.score)
        traverse(node.left)
        traverse(node.right)

    traverse(tree)

    feature_importance = pd.DataFrame([
        {"Feature": feature_names[idx], "Importance": value}
        for idx, value in importance_dict.items()
    ])

    total = feature_importance["Importance"].sum()
    if total > 0:
        feature_importance["Importance"] = 100 * feature_importance["Importance"] / total
    else:
        feature_importance["Importance"] = 0

    feature_importance = feature_importance.sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(feature_importance["Feature"], feature_importance["Importance"], color="#5DADE2")
    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_title("QLDT+ Feature Importance", fontsize=14)

    xlim = ax.get_xlim()[1]
    for bar in bars:
        width = bar.get_width()
        if width > 0.85 * xlim:
            x_pos = width - 0.02 * xlim; align = 'right'; color = 'white'
        else:
            x_pos = width + 0.02 * xlim; align = 'left'; color = 'black'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2.0, f"{width:.1f}%", va='center', ha=align, fontsize=11, color=color)

    fig.tight_layout(pad=2.0)
    fig.savefig(f"{filename}.png", dpi=300)
    plt.close()

    return feature_importance
