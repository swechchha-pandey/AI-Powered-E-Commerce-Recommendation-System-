"""
Hybrid Recommendation System — Flask Web UI
============================================
Segments:
  A  → 0 orders  → Global top-rated (Bayesian)
  B  → 1 order   → Category-based (price-aware)
  C  → 2-4 orders → Multi-category + price
  D  → 5+ orders  → NCF (NeuMF v2) + popularity boost

Auth  : Register → Login → Preferences → Dashboard
Track : All clicks & page views logged to user_interactions.csv
"""

import os, csv, json, uuid
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_CSV            = os.path.join(DATA_DIR, "users.csv")
INTERACTIONS_CSV     = os.path.join(DATA_DIR, "user_interactions.csv")
PREFERENCES_CSV      = os.path.join(DATA_DIR, "user_preferences.csv")
PREF_HISTORY_CSV     = os.path.join(DATA_DIR, "user_preferences_history.csv")
MODEL_DIR            = os.path.join(BASE_DIR, "model_artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")

# ── CSV helpers ────────────────────────────────────────────────────────────

def _ensure_csv(path, headers):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)

_ensure_csv(USERS_CSV,        ["user_id","username","email","password_hash","created_at","order_count"])
_ensure_csv(INTERACTIONS_CSV, ["event_id","user_id","username","event_type",
                                "product_id","category","segment","timestamp","session_id"])
_ensure_csv(PREFERENCES_CSV,  ["user_id","username","categories","updated_at"])
_ensure_csv(PREF_HISTORY_CSV, ["history_id","user_id","username","categories","changed_at","change_count"])

def _read_users():
    users = {}
    with open(USERS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            users[row["username"]] = row
    return users

def _write_user(row_dict):
    headers = ["user_id","username","email","password_hash","created_at","order_count"]
    with open(USERS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writerow(row_dict)

def _update_user_orders(username, new_count):
    users = _read_users()
    if username in users:
        users[username]["order_count"] = str(new_count)
    headers = ["user_id","username","email","password_hash","created_at","order_count"]
    with open(USERS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for u in users.values():
            w.writerow(u)

# ── PREFERENCES helpers ─────────────────────────────────────────────────────

def _read_preferences(username):
    """Return list of preferred categories for a user, or [] if none set."""
    try:
        with open(PREFERENCES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if row["username"] == username:
                    cats = row["categories"]
                    return cats.split("|") if cats else []
    except Exception:
        pass
    return []

def _write_preferences(user_id, username, categories):
    """Save or update user preferences (categories = list of strings)."""
    rows = []
    try:
        with open(PREFERENCES_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        pass

    rows = [r for r in rows if r["username"] != username]
    rows.append({
        "user_id"   : user_id,
        "username"  : username,
        "categories": "|".join(categories),
        "updated_at": datetime.now().isoformat(),
    })

    headers = ["user_id","username","categories","updated_at"]
    with open(PREFERENCES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    change_count = 1
    try:
        with open(PREF_HISTORY_CSV, newline="") as f:
            history_rows = list(csv.DictReader(f))
            user_changes = [r for r in history_rows if r["username"] == username]
            change_count = len(user_changes) + 1
    except Exception:
        pass

    with open(PREF_HISTORY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["history_id","user_id","username",
                                           "categories","changed_at","change_count"])
        w.writerow({
            "history_id"  : str(uuid.uuid4()),
            "user_id"     : user_id,
            "username"    : username,
            "categories"  : "|".join(categories),
            "changed_at"  : datetime.now().isoformat(),
            "change_count": change_count,
        })

def log_interaction(user_id, username, event_type, product_id="", category="", segment=""):
    with open(INTERACTIONS_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            str(uuid.uuid4()),
            user_id, username, event_type,
            product_id, category, segment,
            datetime.now().isoformat(),
            session.get("session_id", "")
        ])

# ── POWER BI EXPORT ─────────────────────────────────────────────────────────

POWERBI_DIR = os.path.join(BASE_DIR, "powerbi_data")
os.makedirs(POWERBI_DIR, exist_ok=True)

def export_for_powerbi():
    import shutil
    shutil.copy(USERS_CSV,        os.path.join(POWERBI_DIR, "pb_users.csv"))
    shutil.copy(INTERACTIONS_CSV, os.path.join(POWERBI_DIR, "pb_interactions.csv"))
    shutil.copy(PREFERENCES_CSV,  os.path.join(POWERBI_DIR, "pb_preferences.csv"))
    if os.path.exists(PREF_HISTORY_CSV):
        shutil.copy(PREF_HISTORY_CSV, os.path.join(POWERBI_DIR, "pb_pref_history.csv"))

    users       = _read_users()
    total_users = len(users)
    new_today   = sum(1 for u in users.values()
                      if u.get("created_at","").startswith(datetime.now().strftime("%Y-%m-%d")))
    seg_a = sum(1 for u in users.values() if int(u.get("order_count",0)) == 0)
    seg_b = sum(1 for u in users.values() if int(u.get("order_count",0)) == 1)
    seg_c = sum(1 for u in users.values() if 2 <= int(u.get("order_count",0)) <= 4)
    seg_d = sum(1 for u in users.values() if int(u.get("order_count",0)) >= 5)

    now = datetime.now().isoformat()
    with open(os.path.join(POWERBI_DIR, "pb_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "last_updated"])
        w.writerow(["total_users",     total_users, now])
        w.writerow(["new_users_today", new_today,   now])
        w.writerow(["segment_A",       seg_a,       now])
        w.writerow(["segment_B",       seg_b,       now])
        w.writerow(["segment_C",       seg_c,       now])
        w.writerow(["segment_D",       seg_d,       now])


# ╔══════════════════════════════════════════════════════════╗
# ║            REAL MODEL — LOAD ARTIFACTS AT STARTUP        ║
# ╚══════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd

# ── File paths for saved artifacts ────────────────────────────────────────
_NCF_MODEL_PATH      = os.path.join(MODEL_DIR, "hybrid_ncf_best.keras")
_USER_ENC_PATH       = os.path.join(MODEL_DIR, "hybrid_user_enc.npy")
_ITEM_ENC_PATH       = os.path.join(MODEL_DIR, "hybrid_item_enc.npy")
_PRODUCT_STATS_PATH  = os.path.join(MODEL_DIR, "hybrid_product_stats.csv")
_CAT_RANKINGS_PATH   = os.path.join(MODEL_DIR, "hybrid_category_rankings.csv")
_USER_PROFILES_PATH  = os.path.join(MODEL_DIR, "hybrid_user_profiles.csv")

# ── Load artifacts (once at startup) ──────────────────────────────────────
print("Loading model artifacts...")

# product_stats — bayesian scores, avg_price, num_ratings per product
product_stats = pd.read_csv(_PRODUCT_STATS_PATH)
product_stats["price_bucket"] = product_stats["price_bucket"].astype(str)

# global top list (sorted by bayesian_score)
global_top = (
    product_stats
    .sort_values("bayesian_score", ascending=False)["product_id"]
    .tolist()
)

# per-category pre-ranked product lists
_cat_rank_df   = pd.read_csv(_CAT_RANKINGS_PATH)
cat_rankings   = (
    _cat_rank_df
    .sort_values("rank")
    .groupby("category")["product_id"]
    .apply(list)
    .to_dict()
)

# user profiles (category weights + price bucket)
_prof_df = pd.read_csv(_USER_PROFILES_PATH)
user_category_profile = {}
user_price_profile    = {}
for uid, grp in _prof_df.groupby("customer_unique_id"):
    cats = [(row["category"], row["weight"]) for _, row in grp.iterrows()]
    user_category_profile[uid] = cats
    user_price_profile[uid]    = grp.iloc[0]["price_bucket"]

# fast lookup dicts
score_lookup  = product_stats.set_index("product_id")["bayesian_score"].to_dict()
cat_lookup    = product_stats.set_index("product_id")["product_category_name_english"].to_dict()
bucket_lookup = product_stats.set_index("product_id")["price_bucket"].to_dict()

# NCF model + encoders
_ncf_model   = None
_user_enc    = None
_item_enc    = None
_num_items   = 0
_pop_array   = None

try:
    from tensorflow import keras
    from sklearn.preprocessing import LabelEncoder

    _ncf_model = keras.models.load_model(_NCF_MODEL_PATH)

    _user_classes = np.load(_USER_ENC_PATH, allow_pickle=True)
    _item_classes = np.load(_ITEM_ENC_PATH, allow_pickle=True)

    _user_enc = LabelEncoder()
    _user_enc.classes_ = _user_classes

    _item_enc = LabelEncoder()
    _item_enc.classes_ = _item_classes

    _num_items = len(_item_classes)

    # popularity array (log-scaled interaction count per item)
    _item_counts = (
        product_stats
        .set_index("product_id")["num_ratings"]
        .reindex(_item_classes)
        .fillna(1)
        .values
        .astype(float)
    )
    _pop_array = np.log1p(_item_counts)
    _pop_array /= (_pop_array.max() + 1e-9)

    print("NCF model loaded successfully.")
except Exception as e:
    print(f"NCF model could not load ({e}). Segment D will fall back to category engine.")

print("All artifacts loaded.")

# ── Price bucket config (matches training script) ──────────────────────────
PRICE_BINS   = [0, 30, 60, 100, 200, 500, float("inf")]
PRICE_LABELS = ["budget", "low", "mid", "mid-high", "high", "premium"]
ADJACENT_BUCKETS = {
    "budget"  : ["budget", "low"],
    "low"     : ["budget", "low", "mid"],
    "mid"     : ["low", "mid", "mid-high"],
    "mid-high": ["mid", "mid-high", "high"],
    "high"    : ["mid-high", "high", "premium"],
    "premium" : ["high", "premium"],
}

# cat+price combo rankings (built from product_stats)
cat_price_rankings = {}
for (cat, bucket), grp in product_stats.groupby(
        ["product_category_name_english", "price_bucket"]):
    cat_price_rankings[(cat, str(bucket))] = (
        grp.sort_values("bayesian_score", ascending=False)["product_id"].tolist()
    )


# ── Internal helpers (mirrors training script logic) ──────────────────────

def _get_candidates(category, price_bucket, seen, n, log):
    candidates = []
    lvl1 = [p for p in cat_price_rankings.get((category, price_bucket), [])
             if p not in seen]
    candidates.extend(lvl1)
    if len(candidates) >= n:
        log.append(f"cat+price({price_bucket})")
        return candidates[:n]

    for bucket in ADJACENT_BUCKETS.get(price_bucket, [price_bucket]):
        extras = [p for p in cat_price_rankings.get((category, bucket), [])
                  if p not in seen and p not in candidates]
        candidates.extend(extras)
    if len(candidates) >= n:
        log.append("cat+adjacent_price")
        return candidates[:n]

    extras = [p for p in cat_rankings.get(category, [])
              if p not in seen and p not in candidates]
    candidates.extend(extras)
    if len(candidates) >= n:
        log.append("cat_only")
        return candidates[:n]

    log.append("global_fallback")
    extras = [p for p in global_top if p not in seen and p not in candidates]
    candidates.extend(extras)
    return candidates[:n]


def _enforce_diversity(recs, seen=None, min_cats=3, top_n=10):
    if seen is None:
        seen = set()
    cats_seen, diverse, remainder = set(), [], []
    for p in recs:
        cat = cat_lookup.get(p)
        if cat and cat not in cats_seen and len(cats_seen) < min_cats:
            diverse.append(p)
            cats_seen.add(cat)
        else:
            remainder.append(p)
    if len(cats_seen) < min_cats:
        recs_set = set(recs)
        for p in global_top:
            if len(cats_seen) >= min_cats:
                break
            cat = cat_lookup.get(p)
            if cat and cat not in cats_seen and p not in recs_set and p not in seen:
                diverse.append(p)
                cats_seen.add(cat)
    return (diverse + remainder)[:top_n]


def _build_rec_rows(product_ids, rank_start=1):
    """Convert a list of product_ids into dashboard-ready dicts."""
    info = (
        product_stats[product_stats["product_id"].isin(product_ids)]
        [["product_id", "product_category_name_english",
          "bayesian_score", "avg_price", "num_ratings"]]
        .copy()
    )
    info = info.set_index("product_id").reindex(product_ids).reset_index()
    rows = []
    for i, row in enumerate(info.itertuples(), start=rank_start):
        rows.append({
            "rank"      : i,
            "product_id": row.product_id,
            "name"      : str(row.product_id)[:16] + "...",
            "category"  : row.product_category_name_english
                          if pd.notna(row.product_category_name_english) else "unknown",
            "score"     : round(float(row.bayesian_score), 3)
                          if pd.notna(row.bayesian_score) else 0.0,
            "price"     : round(float(row.avg_price), 2)
                          if pd.notna(row.avg_price) else 0.0,
            "ratings"   : int(row.num_ratings)
                          if pd.notna(row.num_ratings) else 0,
        })
    return rows


def _category_recommend_recs(customer_unique_id, top_n=10):
    """Returns (recs_list, price_bucket, categories_used, strategy_str)."""
    seen         = set()   # new web users have no purchase history in training data
    cat_profile  = user_category_profile.get(customer_unique_id, [])
    price_bucket = user_price_profile.get(customer_unique_id, "mid")
    log          = []

    if not cat_profile:
        recs      = global_top[:top_n]
        cats_used = ["global"]
        log.append("global_new_user")
    else:
        weights   = [w for _, w in cat_profile]
        raw_slots = [max(1, round(w * top_n)) for w in weights]
        raw_slots[0] += top_n - sum(raw_slots)

        recs, cats_used = [], []
        for (category, _), n_slots in zip(cat_profile, raw_slots):
            if n_slots <= 0:
                continue
            cat_recs = _get_candidates(
                category, price_bucket,
                seen | set(recs), n_slots, log
            )
            recs.extend(cat_recs)
            cats_used.append(category)

        if len(recs) < top_n:
            log.append("global_pad")
            extras = [p for p in global_top if p not in seen and p not in recs]
            recs.extend(extras[:top_n - len(recs)])

    recs = _enforce_diversity(recs[:top_n], seen=seen, min_cats=3, top_n=top_n)
    return (
        _build_rec_rows(recs),
        price_bucket,
        cats_used,
        " → ".join(dict.fromkeys(log)),
    )


def _ncf_recommend_recs(customer_unique_id, top_n=10, pop_weight=0.10):
    """NCF-based recommendations for Segment D users. Returns recs_list."""
    if _ncf_model is None or _user_enc is None:
        return None

    if customer_unique_id not in _user_enc.classes_:
        return None   # unseen user → caller falls back to category engine

    uid_enc = int(_user_enc.transform([customer_unique_id])[0])

    # Build set of already-seen item indices for this user
    # (we don't track per-user purchase history in app, so use all items
    #  that appear in their profile as a proxy — conservative approach)
    seen_enc = set()

    candidates = np.array([i for i in range(_num_items) if i not in seen_enc])
    if len(candidates) == 0:
        return None

    user_arr = np.full(len(candidates), uid_enc)
    scores   = _ncf_model.predict(
        [user_arr, candidates], batch_size=1024, verbose=0
    ).flatten()
    scores  += pop_weight * _pop_array[candidates]
    top_idx  = np.argsort(scores)[::-1][:top_n]

    top_pids = _item_enc.inverse_transform(candidates[top_idx])
    return _build_rec_rows(list(top_pids))


# ── Preference helpers ────────────────────────────────────────────────────

def _category_from_preferences(preferred_cats, price_bucket="mid", top_n=10):
    """
    Build ranked product list from user's preferred categories (Segment A).
    Falls back to global_top if preferred_cats is empty.
    Returns (recs_list, cats_used, strategy_str)
    """
    if not preferred_cats:
        return _build_rec_rows(global_top[:top_n]), ["global"], "global_top_rated (no prefs set)"

    seen      = set()
    recs      = []
    cats_used = []
    log       = []

    # Equal slots across all preferred categories
    n_cats     = len(preferred_cats)
    slots_each = max(1, top_n // n_cats)
    remainder  = top_n - slots_each * n_cats

    for i, cat in enumerate(preferred_cats):
        n_slots = slots_each + (1 if i < remainder else 0)
        cat_recs = _get_candidates(cat, price_bucket, seen | set(recs), n_slots, log)
        recs.extend(cat_recs)
        cats_used.append(cat)

    # Pad with global if not enough
    if len(recs) < top_n:
        log.append("global_pad")
        extras = [p for p in global_top if p not in seen and p not in recs]
        recs.extend(extras[:top_n - len(recs)])

    recs = _enforce_diversity(recs[:top_n], seen=seen, min_cats=3, top_n=top_n)
    return (
        _build_rec_rows(recs),
        cats_used,
        "pref_categories → " + " → ".join(dict.fromkeys(log)),
    )


def _re_rank_with_preferences(recs, preferred_cats, boost=0.25):
    """
    Re-rank an existing recs list by boosting items whose category
    matches user's preferred_cats.
    recs  : list of dicts from _build_rec_rows()
    boost : score boost applied to preferred-category items
    Returns re-ranked list of dicts.
    """
    if not preferred_cats:
        return recs

    pref_set = set(preferred_cats)

    def _sort_key(item):
        base  = item.get("score", 0.0)
        bonus = boost if item.get("category") in pref_set else 0.0
        return -(base + bonus)   # negative for ascending sort

    re_ranked = sorted(recs, key=_sort_key)
    for i, item in enumerate(re_ranked, start=1):
        item["rank"] = i
    return re_ranked


# ╔══════════════════════════════════════════════════════════╗
# ║              MAIN RECOMMENDATION FUNCTION                ║
# ╚══════════════════════════════════════════════════════════╝

def get_recommendations(username, order_count, top_n=10):
    """
    Routes to the correct engine based on order_count (segment).
    Uses real model artifacts — no fake/random data.
    """
    preferred = _read_preferences(username)
    n         = order_count

    # ── Segment A: new user (0 orders) → preference-based recommendation ─────
    if n == 0:
        price_b   = "mid"   # new user has no price history
        if preferred:
            recs, cats_used, strat = _category_from_preferences(preferred, price_b, top_n)
            seg       = "A"
            seg_label = "A — New User (Preference-based)"
            strategy  = f"Preference-based top-rated → {strat}"
        else:
            # No preferences set yet → fall back to global top
            recs      = _build_rec_rows(global_top[:top_n])
            seg       = "A"
            seg_label = "A — New User (Global)"
            strategy  = "Global top-rated products (Bayesian) — set preferences for personalised results"

    # ── Segment B: cold-start (1 order) → category + preference hybrid ───────
    elif n == 1:
        recs, price_b, cats_used, strat = _category_recommend_recs(username, top_n)
        if preferred:
            recs  = _re_rank_with_preferences(recs, preferred)
            strat = strat + " + pref_rerank"
        seg       = "B"
        seg_label = "B — Cold Start"
        strategy  = f"Category + Preference hybrid → {strat}"

    # ── Segment C: warm user (2-4 orders) → category + preference hybrid ─────
    elif n < 5:
        recs, price_b, cats_used, strat = _category_recommend_recs(username, top_n)
        if preferred:
            recs  = _re_rank_with_preferences(recs, preferred)
            strat = strat + " + pref_rerank"
        seg       = "C"
        seg_label = "C — Warm User"
        strategy  = f"Multi-category + price profiling + preference → {strat}"

    # ── Segment D: active user (5+ orders) → NCF + preference re-ranking ─────
    # ── Segment D: active user (5+ orders) → NCF + preference hybrid ─────
# ── Segment D: active user (5+ orders) → NCF + preference hybrid ─────
    else:
        ncf_recs = _ncf_recommend_recs(username, top_n)

        if ncf_recs:
            if preferred:
                pref_recs, _, _ = _category_from_preferences(preferred, "mid", top_n)
                # remove duplicates
                pref_ids = {item["product_id"] for item in pref_recs}
                ncf_filtered = [item for item in ncf_recs if item["product_id"] not in pref_ids]
                # 🔥 mix
                recs = pref_recs[:3] + ncf_filtered[:(top_n - 3)]
            else:
                recs = ncf_recs

            # ❗ ALWAYS DEFINE THESE
            seg       = "D"
            seg_label = "D — Active User"
            strategy  = "NCF (NeuMF v2) + preference hybrid"
            price_b   = user_price_profile.get(username, "unknown")
        else:
            # fallback
            recs, price_b, cats_used, strat = _category_recommend_recs(username, top_n)

            if preferred:
                recs  = _re_rank_with_preferences(recs, preferred)
                strat = strat + " + pref_rerank"
            
            seg       = "D"
            seg_label = "D — Active User (cat fallback)"
            strategy  = f"NCF fallback → {strat}"
    return {
        "segment"       : seg_label,
        "segment_id"    : seg,
        "strategy"      : strategy,
        "recs"          : recs,
        "order_count"   : n,
        "preferred_cats": preferred,
    }


# ── auth decorator ──────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please login to continue.", "info")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username","").strip().lower()
        email    = request.form.get("email","").strip().lower()
        password = request.form.get("password","")
        confirm  = request.form.get("confirm","")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        users = _read_users()
        if username in users:
            flash("Username already taken.", "error")
            return render_template("register.html")
        if any(u["email"] == email for u in users.values()):
            flash("Email already registered.", "error")
            return render_template("register.html")

        _write_user({
            "user_id"      : str(uuid.uuid4()),
            "username"     : username,
            "email"        : email,
            "password_hash": generate_password_hash(password),
            "created_at"   : datetime.now().isoformat(),
            "order_count"  : "0",
        })
        export_for_powerbi()
        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip().lower()
        password = request.form.get("password","")

        users = _read_users()
        user  = users.get(username)

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        session["username"]    = username
        session["user_id"]     = user["user_id"]
        session["email"]       = user["email"]
        session["order_count"] = int(user.get("order_count", 0))
        session["session_id"]  = str(uuid.uuid4())

        log_interaction(user["user_id"], username, "login")

        prefs = _read_preferences(username)
        if not prefs:
            return redirect(url_for("preferences"))
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    if "username" in session:
        log_interaction(session.get("user_id",""), session.get("username",""), "logout")
    session.clear()
    return redirect(url_for("index"))


@app.route("/preferences", methods=["GET","POST"])
@login_required
def preferences():
    username = session["username"]

    if request.method == "POST":
        selected = request.form.getlist("categories")
        if not selected:
            flash("Please select at least one category.", "error")
            return redirect(url_for("preferences"))

        _write_preferences(session["user_id"], username, selected)
        session["preferred_cats"] = selected
        export_for_powerbi()
        flash(f"Preferences saved! Showing recommendations for {len(selected)} categories.", "success")
        return redirect(url_for("dashboard"))

    selected_categories = _read_preferences(username)
    return render_template("preferences.html", selected_categories=selected_categories)


@app.route("/dashboard")
@login_required
def dashboard():
    username    = session["username"]
    order_count = session.get("order_count", 0)
    result      = get_recommendations(username, order_count, top_n=10)
    log_interaction(session["user_id"], username, "page_view", segment=result["segment_id"])
    return render_template("dashboard.html", result=result, username=username, order_count=order_count)


@app.route("/simulate_purchase", methods=["POST"])
@login_required
def simulate_purchase():
    product_id = request.form.get("product_id","")
    category   = request.form.get("category","")
    segment    = request.form.get("segment","")

    order_count = session.get("order_count", 0) + 1
    session["order_count"] = order_count
    _update_user_orders(session["username"], order_count)

    log_interaction(session["user_id"], session["username"],
                    "purchase", product_id, category, segment)
    export_for_powerbi()
    flash(f"Purchase recorded! You now have {order_count} order(s). Recommendations updated.", "success")
    return redirect(url_for("dashboard"))


@app.route("/click_product", methods=["POST"])
@login_required
def click_product():
    product_id = request.form.get("product_id","")
    category   = request.form.get("category","")
    segment    = request.form.get("segment","")
    log_interaction(session["user_id"], session["username"],
                    "product_click", product_id, category, segment)
    return jsonify({"status":"ok"})


@app.route("/profile")
@login_required
def profile():
    username    = session["username"]
    order_count = session.get("order_count", 0)

    history = []
    with open(INTERACTIONS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["username"] == username:
                history.append(row)
    history = list(reversed(history[-50:]))

    seg_id = (
        "A" if order_count == 0 else
        "B" if order_count == 1 else
        "C" if order_count < 5 else "D"
    )
    next_seg = {"A":"B","B":"C","C":"D","D":"D"}[seg_id]
    orders_to_next = (
        1 if seg_id=="A" else
        1 if seg_id=="B" else
        max(0, 5 - order_count) if seg_id=="C" else 0
    )

    preferred_cats = _read_preferences(username)

    return render_template("profile.html",
        username=username,
        email=session.get("email",""),
        order_count=order_count,
        seg_id=seg_id,
        next_seg=next_seg,
        orders_to_next=orders_to_next,
        history=history,
        preferred_cats=preferred_cats,
    )



# ── Run ─────────────────────────────────────────────────────────────────────
# Local development: python app3.py
# Production (Render): gunicorn -w 2 -b 0.0.0.0:5050 app3:app
if __name__ == "__main__":
    app.run(debug=False, port=5050)
