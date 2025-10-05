"""
Runner/CLI for your QLDT+ project.
- Keeps your prints.
- Writes all artifacts into ./artifacts/
Usage:
  python -m src.run_qldt --dataset pima --grid --draw
"""

import argparse
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.utils_io import ensure_graphviz_on_path
from src.qldtplus import (
    load_transfusion_data, load_pima_data, load_heart_data,
    build_tree_qldt, prune_zero_leaves, predict,
    draw_qldt_tree, draw_cart_tree,
    get_tree_stats_qldt, get_tree_stats_cart,
    run_param_grid_search_qldt,
    compute_qldt_feature_importance,
    extract_positive_paths, evaluate_path, get_feature_names_from_indices,
    build_interpretability_table,
)

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

FEATURE_NAMES = {
    "pima": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
             'BMI', 'DiabetesPedigreeFunction', 'Age'],
    "transfusion": ["Recency", "Frequency", "Monetary", "Time"],
}

def main():
    ensure_graphviz_on_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pima", "transfusion", "heart"], default="pima")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.045)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--draw", action="store_true")
    args = ap.parse_args()

    # ---- Load data ----
    if args.dataset == "heart":
        (X_train, X_test, y_train, y_test), feature_names = load_heart_data()
    elif args.dataset == "pima":
        X_train, X_test, y_train, y_test = load_pima_data()
        feature_names = FEATURE_NAMES["pima"]
    else:
        X_train, X_test, y_train, y_test = load_transfusion_data()
        feature_names = FEATURE_NAMES["transfusion"]

    # ---- Train QLDT+ ----
    tree = build_tree_qldt(X_train, y_train, theta_p=args.theta, tau=args.tau, max_depth=args.max_depth)
    tree = prune_zero_leaves(tree)
    y_pred = predict(X_test, tree, theta_p=args.theta)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"✅ QLDT+ Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    # ---- Draw QLDT+ tree ----
    if args.draw:
        draw_qldt_tree(tree, filename=str(ART / f"qldt_{args.dataset}"))

    # ---- CART baseline  ----
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
    clf.fit(X_train, y_train)
    y_cart = clf.predict(X_test)
    acc_cart = accuracy_score(y_test, y_cart)
    f1_cart = f1_score(y_test, y_cart)
    print(f"✅ CART  Accuracy: {acc_cart:.4f}, F1 Score: {f1_cart:.4f}")

    if args.draw and feature_names:
        draw_cart_tree(clf, feature_names, filename=str(ART / f"cart_{args.dataset}"))

    # ---- Stats print ----
    qldt_stats = get_tree_stats_qldt(tree)
    cart_stats = {"depth": clf.get_depth(), "leaves": clf.get_n_leaves()}

    print("\n📊 Tree Structure Comparison:")
    print(f"QLDT+ Depth: {qldt_stats['depth']}, Leaves: {qldt_stats['leaves']}, Positive Leaves: {qldt_stats['positive_leaves']}")
    print(f"CART   Depth: {cart_stats['depth']}, Leaves: {cart_stats['leaves']}")
    print("\n🎯 Performance Comparison:")
    print(f"QLDT+ Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(f"CART  Accuracy: {acc_cart:.4f}, F1 Score: {f1_cart:.4f}")

    # ---- Grid search (CSV saved) ----
    if args.grid:
        df_grid, tree_best = run_param_grid_search_qldt(X_train, X_test, y_train, y_test)
        df_grid.to_csv(ART / "qldt_param_grid_search_results.csv", index=False)
        draw_qldt_tree(tree_best, filename=str(ART / "qldt_best_tree"))
        print("✅ Best QLDT+ tree saved as artifacts/qldt_best_tree.png")
        print("✅ Grid Search Results saved to artifacts/qldt_param_grid_search_results.csv")

    # ---- Feature importance (PNG) ----
    if args.draw and feature_names:
        compute_qldt_feature_importance(tree, feature_names, filename=str(ART / "qldt_feature_importance"))
        print("✅ Feature importance saved as artifacts/qldt_feature_importance.png")

    # ---- Interpretability table (HTML) ----
    if args.draw and feature_names:
        # Using your own helper to compute per-path metrics on TEST set
        paths = extract_positive_paths(tree)
        results = []
        for i, (path, rho) in enumerate(paths, start=1):
            matched = 0
            correct = 0
            for xi, yi in zip(X_test, y_test):
                if evaluate_path(path, xi) >= 0.5:
                    matched += 1
                    if yi == 1:
                        correct += 1
            support = matched / len(X_test) if len(X_test) else 0
            accuracy_rule = (correct / matched) if matched else 0
            results.append({
                "Path ID": i,
                "Features Used": ", ".join(get_feature_names_from_indices(path, feature_names)),
                "Rule Length": len(path),
                "Tree Depth": len(path),
                "Support": round(support, 2),
                "Accuracy": round(accuracy_rule, 2),
                "Avg Rho (ρ)": round(rho, 2)
            })
        import pandas as pd
        df_interpret = pd.DataFrame(results)
        df_interpret.style.set_caption("Table: QLDT+ Interpretability Metrics") \
            .to_html(ART / "qldt_interpretability_table.html")
        print("✅ Interpretability table saved as artifacts/qldt_interpretability_table.html")


if __name__ == "__main__":
    main()
