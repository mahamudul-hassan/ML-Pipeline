"""Native smoke test of the engine: every model, every preprocessing / selection option family."""
import json, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import engine as E

def cfg(target, models, **over):
    c = {
        "dataset": {"target": target, "task": "auto", "exclude": ["id"]},
        "split": {"test_size": 0.2, "val_size": 0.1, "stratify": True, "shuffle": True, "seed": 42},
        "cv": {"strategy": "stratified_kfold", "folds": 3},
        "preprocess": {"num_impute": "median", "encoding": "onehot", "scaling": "standard", "imbalance": "none"},
        "fe": {"custom_features": [{"name": "ratio", "expr": "study_hours / (sleep_hours + 1)"}], "date_parts": True},
        "fs": {"variance_filter": True, "method": "none"},
        "models": models, "ensembles": [],
        "eval": {"primary_metric": "auto", "permutation": "best", "shap": "best", "shap_rows": 10, "learning_curve": "none"},
        "tuning": {"enabled": False},
        "advanced": {"slow_model_rows": 600, "max_train_rows": 1500},
    }
    for k, v in over.items():
        c[k] = {**c.get(k, {}), **v} if isinstance(v, dict) else v
    return c

def check(out, label):
    r = json.loads(out)
    if "error" in r:
        print("FAIL", label, r["error"]); print(r.get("trace", "")[-1500:]); sys.exit(1)
    return r

t = time.time()
prof = check(E.load_sample("student_performance"), "load")
print("profile", prof["profile"]["rows"], prof["profile"]["target_guess"])
models = json.load(open("/tmp/models.json"))
cls_models = [m for m in models if "classification" in m["tasks"]]
r = check(E.run(json.dumps(cfg("pass", cls_models))), "cls all models")
bad = [(m["name"], m.get("error")) for m in r["models"].values() if m["status"] != "ok"]
print(f"classification: {len(r['models'])} results, errors: {bad}, best {r['best']} {r['primary']} in {time.time()-t:.1f}s")
for mid in r["ranking"][:5]:
    m = r["models"][mid]; print("  ", m["name"], round(m["score"], 4), "test", round(m["test"]["f1"], 4))
# analyses
check(E.analyze("learning_curve", r["best"], "{}"), "lc")
check(E.analyze("pdp", r["best"], json.dumps({"feature": "study_hours"})), "pdp")
check(E.analyze("pdp", r["best"], json.dumps({"feature": "gender"})), "pdp cat")
check(E.analyze("local_shap", r["best"], json.dumps({"row": 3})), "local shap")
print("predict", check(E.predict(r["best"], json.dumps([{"age": 20, "gender": "Male", "study_hours": 8, "attendance": 95, "previous_score": 90, "parental_education": "PhD", "extracurricular": "Yes", "sleep_hours": 7.5, "screen_time": 2, "test_score": 88}])), "predict"))
print("whatif", check(E.what_if(r["best"], 0, json.dumps({"study_hours": 1})), "whatif")["after"])
rc = check(E.run_code("from sklearn.ensemble import ExtraTreesClassifier\nprint(len(X_train))\nres = register_model('My ET', ExtraTreesClassifier(n_estimators=50))\nimport matplotlib.pyplot as plt\nplt.plot([1,2,3])\nres"), "run_code")
print("run_code", rc["stdout"].strip(), rc["result"][:80], len(rc["figures"]), rc["results_changed"], rc["exception"])
exp = check(E.export_model(r["best"]), "export"); print("export bytes", len(exp["b64"]))
check(E.predict_csv(r["best"], "age,gender,study_hours\n20,Male,5\n"), "predict csv")
check(E.eda("pass"), "eda")

# regression + heavy options
t = time.time()
reg_models = [m for m in json.load(open("/tmp/models_reg.json")) if "regression" in m["tasks"]]
c = cfg("test_score", reg_models, cv={"strategy": "kfold", "folds": 3},
        preprocess={"num_impute": "iterative", "encoding": "target", "scaling": "robust", "transform": "yeo-johnson", "outliers": "iqr"},
        fe={"polynomial": 2, "poly_columns": ["study_hours", "attendance"], "splines": True, "spline_columns": ["sleep_hours"], "bin_columns": ["age"]},
        fs={"method": "kbest_mi", "k": 12, "drop_correlated": True, "reduction": "none"},
        ensembles=[{"id": "vote", "key": "voting", "name": "Voting Ensemble", "top_k": 3}, {"id": "stack", "key": "stacking", "name": "Stacking Ensemble", "top_k": 3}],
        tuning={"enabled": True, "method": "random", "models": "top3", "n_iter": 4, "cv_folds": 3})
r = check(E.run(json.dumps(c)), "regression")
bad = [(m["name"], m.get("error")) for m in r["models"].values() if m["status"] != "ok"]
print(f"regression: {len(r['models'])} results, errors: {bad}, best {r['models'][r['best']]['name']} r2 {r['models'][r['best']]['score']:.4f} in {time.time()-t:.1f}s")
print("tuned:", [ (m["name"], round(m["tuning"]["before"],4), round(m["tuning"]["after"],4)) for m in r["models"].values() if m.get("tuning")])

# multiclass + every FS / preprocessing variant quickly with 2 models
check(E.load_sample("wine"), "wine")
fast = [m for m in cls_models if m["key"] in ("logreg", "rf")]
for fs in ["kbest_f", "kbest_chi2", "percentile_f", "rfe", "rfecv", "l1", "tree_importance", "sfs_forward"]:
    r = check(E.run(json.dumps(cfg("cultivar", fast, fe={"custom_features": []}, fs={"method": fs, "k": 5}, eval={"permutation": "none", "shap": "none"}))), fs)
    assert all(m["status"] == "ok" for m in r["models"].values()), (fs, [m.get("error") for m in r["models"].values()])
for red in ["pca", "svd", "ica", "kernel_pca", "lda"]:
    r = check(E.run(json.dumps(cfg("cultivar", fast, fe={"custom_features": []}, fs={"method": "none", "reduction": red, "n_components": 2 if red != 'pca' else 0.9}, eval={"permutation": "none", "shap": "none"}))), red)
    assert all(m["status"] == "ok" for m in r["models"].values()), (red, [m.get("error") for m in r["models"].values()])
for imb in ["class_weight", "random_over", "random_under", "smote"]:
    for enc in ["ordinal", "frequency", "onehot_drop_first"]:
        pass
check(E.load_sample("student_performance"), "student")
for imb in ["class_weight", "random_over", "random_under", "smote"]:
    r = check(E.run(json.dumps(cfg("pass", fast, preprocess={"imbalance": imb, "encoding": "frequency", "num_impute": "knn", "scaling": "quantile_normal", "outliers": "isolation_forest", "outlier_factor": 0.02}, eval={"permutation": "none", "shap": "none"}, cv={"strategy": "repeated_stratified_kfold", "folds": 3, "repeats": 2}, tuning={"enabled": True, "method": "halving_random", "models": "best", "cv_folds": 3}))), imb)
    assert all(m["status"] == "ok" for m in r["models"].values()), (imb, [m.get("error") for m in r["models"].values()])
for cvs in ["kfold", "shuffle_split", "stratified_shuffle_split", "time_series", "group_kfold", "none"]:
    c2 = cfg("pass", fast, cv={"strategy": cvs, "folds": 3}, eval={"permutation": "none", "shap": "none"}, tuning={"enabled": True, "method": "grid", "models": "best"})
    c2["dataset"]["group_col"] = "parental_education" if cvs == "group_kfold" else ""
    r = check(E.run(json.dumps(c2)), cvs)
    assert all(m["status"] == "ok" for m in r["models"].values()), (cvs, [m.get("error") for m in r["models"].values()])
# Optuna TPE / random / QMC, built-in GP Bayesian, custom ranges, and ensembles containing XGBoost
xl = [m for m in cls_models if m["key"] in ("xgb", "lgbm", "rf", "logreg")]
for method, sampler in [("optuna", "tpe"), ("optuna", "qmc"), ("bayesian", None)]:
    tun = {"enabled": True, "method": method, "sampler": sampler or "tpe", "models": "top3", "n_iter": 8, "cv_folds": 3,
           "grids": {"xgb": {"n_estimators": {"low": 30, "high": 200, "type": "int"}, "learning_rate": {"low": 0.01, "high": 0.3, "log": True}, "max_depth": [3, 5]}}}
    r = check(E.run(json.dumps(cfg("pass", xl, eval={"permutation": "none", "shap": "none"}, ensembles=[{"id": "st", "key": "stacking", "name": "Stacking", "top_k": 3}], tuning=tun))), method)
    bad = [(m["name"], m.get("error")) for m in r["models"].values() if m["status"] != "ok"]
    assert not bad, bad
    tuned = [m for m in r["models"].values() if m.get("tuning")]
    assert len(tuned) == 3 and all(len(m["tuning"]["history"]) == 8 for m in tuned), [(m["name"], m["tuning"]["n_trials"]) for m in tuned]
    print(method, sampler, [(m["name"], m["tuning"]["method"], round(m["tuning"]["before"], 4), round(m["tuning"]["after"], 4), (m["tuning"]["importance"] or [{}])[0].get("param")) for m in tuned])
print("variants ok")
print("versions", E.versions())
