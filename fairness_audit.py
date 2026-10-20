"""
Ethical audit of a loan-approval model: bias analysis, fairness metrics, mitigation.
Run: python fairness_audit.py   (writes results.json and PNG charts to ./out)
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

OUT = "out"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N = 20000

# ---------------------------------------------------------------- 1. DATA
# Synthetic applicants. Group A = privileged, Group B = unprivileged (protected attribute).
# Historical bias is injected in two ways that mirror real lending data:
#   (a) structural: Group B has lower average income / credit history due to past inequity
#   (b) label bias: past human decisions penalised Group B at equal creditworthiness
group = rng.choice(["A", "B"], size=N, p=[0.65, 0.35])
is_b = (group == "B").astype(int)

income = rng.lognormal(mean=10.9 - 0.25 * is_b, sigma=0.45)           # annual income
credit_hist = np.clip(rng.normal(9 - 2.5 * is_b, 4), 0, 30)            # years
debt_ratio = np.clip(rng.normal(0.32 + 0.05 * is_b, 0.12), 0.02, 0.9)
employment = np.clip(rng.normal(7 - 1.5 * is_b, 4), 0, 35)            # years
# Proxy feature: neighbourhood score correlates with group (e.g. redlining legacy)
zip_score = np.clip(rng.normal(0.65 - 0.30 * is_b, 0.15), 0, 1)

# True repayment ability (what we would ideally predict)
z = (0.9 * (np.log(income) - 10.8) + 0.07 * credit_hist - 3.0 * (debt_ratio - 0.3)
     + 0.03 * employment)
p_repay = 1 / (1 + np.exp(-z))
repays = rng.binomial(1, p_repay)

# Historical approval label = merit + discrimination against group B
z_label = z + 0.9 * (zip_score - 0.5) - 0.7 * is_b
approved_hist = rng.binomial(1, 1 / (1 + np.exp(-z_label)))

df = pd.DataFrame(dict(group=group, income=income, credit_hist=credit_hist,
                       debt_ratio=debt_ratio, employment=employment,
                       zip_score=zip_score, approved=approved_hist, repays=repays))
df.to_csv(f"{OUT}/synthetic_loans.csv", index=False)

FEATURES = ["income", "credit_hist", "debt_ratio", "employment", "zip_score"]

# ---------------------------------------------------------------- 2. DATA AUDIT
data_audit = {
    "n": int(N),
    "share_B": float(is_b.mean()),
    "hist_approval_rate_A": float(df[df.group == "A"].approved.mean()),
    "hist_approval_rate_B": float(df[df.group == "B"].approved.mean()),
    "repay_rate_A": float(df[df.group == "A"].repays.mean()),
    "repay_rate_B": float(df[df.group == "B"].repays.mean()),
    "corr_zip_group": float(np.corrcoef(df.zip_score, is_b)[0, 1]),
    "median_income_A": float(df[df.group == "A"].income.median()),
    "median_income_B": float(df[df.group == "B"].income.median()),
}

# ---------------------------------------------------------------- 3. MODEL
tr, te = train_test_split(df, test_size=0.3, random_state=1, stratify=df.group)
scaler = StandardScaler().fit(tr[FEATURES])
Xtr, Xte = scaler.transform(tr[FEATURES]), scaler.transform(te[FEATURES])


def fit(sample_weight=None, features_idx=None):
    cols = slice(None) if features_idx is None else features_idx
    m = LogisticRegression(max_iter=1000)
    m.fit(Xtr[:, cols], tr.approved, sample_weight=sample_weight)
    return m


base = fit()
score_te = base.predict_proba(Xte)[:, 1]
pred_base = (score_te >= 0.5).astype(int)


# ---------------------------------------------------------------- 4. METRICS
def fairness(y_true, y_pred, grp, label):
    """Group fairness metrics. y_true = ground-truth outcome (repays)."""
    r = {}
    for g in ["A", "B"]:
        m = grp == g
        yt, yp = y_true[m], y_pred[m]
        r[g] = dict(
            selection=float(yp.mean()),
            tpr=float(yp[yt == 1].mean()),
            fpr=float(yp[yt == 0].mean()),
            ppv=float(yt[yp == 1].mean()) if (yp == 1).any() else float("nan"),
        )
    return dict(
        model=label,
        sel_A=r["A"]["selection"], sel_B=r["B"]["selection"],
        disparate_impact=r["B"]["selection"] / r["A"]["selection"],
        dp_diff=r["B"]["selection"] - r["A"]["selection"],
        tpr_A=r["A"]["tpr"], tpr_B=r["B"]["tpr"],
        fpr_A=r["A"]["fpr"], fpr_B=r["B"]["fpr"],
        eo_tpr_gap=r["B"]["tpr"] - r["A"]["tpr"],
        eo_fpr_gap=r["B"]["fpr"] - r["A"]["fpr"],
        ppv_A=r["A"]["ppv"], ppv_B=r["B"]["ppv"],
        accuracy_vs_repay=float(accuracy_score(y_true, y_pred)),
        auc_vs_repay=float(roc_auc_score(y_true, y_pred)),
    )


y_te = te.repays.values
g_te = te.group.values
results = [fairness(y_te, pred_base, g_te, "Baseline (all features)")]

# ---------------------------------------------------------------- 5. MITIGATION
# 5a. Fairness through unawareness: drop the proxy feature (zip_score)
idx = [i for i, f in enumerate(FEATURES) if f != "zip_score"]
m_nozip = fit(features_idx=idx)
pred_nozip = (m_nozip.predict_proba(Xte[:, idx])[:, 1] >= 0.5).astype(int)
results.append(fairness(y_te, pred_nozip, g_te, "Drop proxy (zip_score)"))

# 5b. Reweighing (Kamiran & Calders): weight = P(g)P(y) / P(g,y)
w = np.ones(len(tr))
for g in ["A", "B"]:
    for y in [0, 1]:
        mask = ((tr.group == g) & (tr.approved == y)).values
        w[mask] = ((tr.group == g).mean() * (tr.approved == y).mean()) / mask.mean()
m_rw = fit(sample_weight=w, features_idx=idx)
pred_rw = (m_rw.predict_proba(Xte[:, idx])[:, 1] >= 0.5).astype(int)
results.append(fairness(y_te, pred_rw, g_te, "Drop proxy + reweighing"))

# 5c. Post-processing: group-specific thresholds on the baseline score to equalise TPR
def tpr_at(thr, grp):
    m = g_te == grp
    return (score_te[m][y_te[m] == 1] >= thr).mean()

thr_A = 0.5
target = tpr_at(thr_A, "A")
grid = np.linspace(0.05, 0.95, 181)
thr_B = grid[np.argmin([abs(tpr_at(t, "B") - target) for t in grid])]
pred_thr = np.where(g_te == "B", score_te >= thr_B, score_te >= thr_A).astype(int)
results.append(fairness(y_te, pred_thr, g_te, f"Group thresholds (B={thr_B:.2f})"))

res = pd.DataFrame(results)
res.to_csv(f"{OUT}/fairness_results.csv", index=False)

# Bootstrap CI on baseline disparate impact
boot = []
for _ in range(500):
    i = rng.integers(0, len(te), len(te))
    gA, gB = g_te[i] == "A", g_te[i] == "B"
    boot.append(pred_base[i][gB].mean() / pred_base[i][gA].mean())
di_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

# Proxy strength: coefficient + how well non-protected features predict group
coefs = dict(zip(FEATURES, [float(c) for c in base.coef_[0]]))
proxy_clf = LogisticRegression(max_iter=1000).fit(Xtr, (tr.group == "B").astype(int))
proxy_auc = float(roc_auc_score((te.group == "B").astype(int), proxy_clf.predict_proba(Xte)[:, 1]))

json.dump(dict(data_audit=data_audit, results=results, di_ci=di_ci, coefs=coefs,
               proxy_auc=proxy_auc, thr_B=float(thr_B)), open(f"{OUT}/results.json", "w"), indent=2)

# ---------------------------------------------------------------- 6. VISUALS
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
cA, cB = "#2b6cb0", "#dd6b20"

# Fig 1: feature distributions by group
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for a, f, t in zip(ax, ["income", "credit_hist", "zip_score"],
                   ["Annual income (log scale)", "Credit history (years)", "Neighbourhood score (proxy)"]):
    for g, c in [("A", cA), ("B", cB)]:
        v = df[df.group == g][f]
        a.hist(np.log10(v) if f == "income" else v, bins=40, alpha=.6, color=c, density=True, label=f"Group {g}")
    a.set_title(t); a.set_yticks([])
ax[0].legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_distributions.png", dpi=170); plt.close()

# Fig 2: historical label bias vs repayment (same merit, different approval)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
x = np.arange(2)
ax.bar(x - .18, [data_audit["repay_rate_A"], data_audit["repay_rate_B"]], .36, color="#718096", label="Actual repayment rate")
ax.bar(x + .18, [data_audit["hist_approval_rate_A"], data_audit["hist_approval_rate_B"]], .36, color=[cA, cB], label="Historical approval rate")
ax.set_xticks(x); ax.set_xticklabels(["Group A", "Group B"]); ax.set_ylim(0, 1)
ax.set_ylabel("Rate"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Approvals fall further than repayment does")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_label_bias.png", dpi=170); plt.close()

# Fig 3: score distributions of baseline model
fig, ax = plt.subplots(figsize=(6, 3.6))
for g, c in [("A", cA), ("B", cB)]:
    ax.hist(score_te[g_te == g], bins=40, alpha=.6, color=c, density=True, label=f"Group {g}")
ax.axvline(.5, color="k", ls="--", lw=1); ax.set_xlabel("Predicted approval probability"); ax.set_yticks([])
ax.legend(frameon=False); ax.set_title("Baseline model score distributions")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_scores.png", dpi=170); plt.close()

# Fig 4: selection rate and disparate impact across models
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
labels = [r["model"].replace(" (B=", "\n(B=").replace("Drop proxy + ", "Drop proxy +\n").replace("Drop proxy (", "Drop proxy\n(").replace("Baseline (", "Baseline\n(") for r in results]
xx = np.arange(len(results))
ax[0].bar(xx - .18, res.sel_A, .36, color=cA, label="Group A")
ax[0].bar(xx + .18, res.sel_B, .36, color=cB, label="Group B")
ax[0].set_xticks(xx); ax[0].set_xticklabels(labels, fontsize=7.5); ax[0].set_title("Selection (approval) rate by group")
ax[0].legend(frameon=False)
cols = ["#c53030" if d < .8 else "#2f855a" for d in res.disparate_impact]
ax[1].bar(xx, res.disparate_impact, color=cols)
ax[1].axhline(.8, color="k", ls="--", lw=1); ax[1].text(len(results) - .5, .81, "four-fifths rule (0.80)", ha="right", fontsize=8)
ax[1].set_xticks(xx); ax[1].set_xticklabels(labels, fontsize=7.5); ax[1].set_title("Disparate impact ratio (B / A)")
ax[1].set_ylim(0, 1.15)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_selection_di.png", dpi=170); plt.close()

# Fig 5: equalized odds gaps
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(xx - .18, res.eo_tpr_gap, .36, color="#805ad5", label="TPR gap (B - A)")
ax.bar(xx + .18, res.eo_fpr_gap, .36, color="#d69e2e", label="FPR gap (B - A)")
ax.axhline(0, color="k", lw=.8)
ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=7.5); ax.legend(frameon=False)
ax.set_title("Equalized-odds gaps (0 = parity)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_equalized_odds.png", dpi=170); plt.close()

# Fig 6: fairness-accuracy trade-off
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.scatter(res.disparate_impact, res.accuracy_vs_repay, s=80, c=["#c53030"] + ["#2b6cb0"] * 3)
for i, r in res.iterrows():
    ax.annotate(f"M{i}", (r.disparate_impact, r.accuracy_vs_repay), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set_xlabel("Disparate impact ratio"); ax.set_ylabel("Accuracy vs. true repayment")
ax.set_title("Fairness vs. accuracy trade-off")
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_tradeoff.png", dpi=170); plt.close()

print(res.round(3).to_string())
print(json.dumps(data_audit, indent=1))
print("DI 95% CI", di_ci, "proxy AUC", proxy_auc, "coefs", coefs)
