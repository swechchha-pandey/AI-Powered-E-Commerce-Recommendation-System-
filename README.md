# AI-Powered-E-Commerce-Recommendation-System-
E-commerce product recommendation system using Neural Collaborative Filtering (NCF) and category-based recommendations.

## Folder Structure
```
recommender_ui/
├── app.py                      ← Flask application (main entry point)
├── README.md
├── requirements.txt
├── data/                       ← auto-created on first run
│   ├── users.csv               ← registered users (hashed passwords)
│   └── user_interactions.csv   ← all events logged here
├── model_artifacts/            ← PUT YOUR MODEL FILES HERE
│   ├── hybrid_ncf_final.keras
│   ├── hybrid_user_enc.npy
│   ├── hybrid_item_enc.npy
│   ├── hybrid_product_stats.csv
│   ├── hybrid_category_rankings.csv
│   └── hybrid_user_profiles.csv
└── templates/
    ├── base.html               ← shared layout + nav + CSS variables
    ├── landing.html            ← public landing page
    ├── login.html              ← login form
    ├── register.html           ← registration form
    ├── dashboard.html          ← recommendations page
    └── profile.html            ← user profile + interaction log
```

## Setup & Run

### 1. Install dependencies
```bash
pip install flask werkzeug
# Your model also needs:
pip install tensorflow pandas numpy scikit-learn
```

### 2. Place model artifacts
Copy your trained model files into the `model_artifacts/` folder:
- `hybrid_ncf_final.keras`
- `hybrid_user_enc.npy`
- `hybrid_item_enc.npy`
- `hybrid_product_stats.csv`
- `hybrid_category_rankings.csv`
- `hybrid_user_profiles.csv`

### 3. Connect your real model (IMPORTANT)
In `app.py`, find the `get_recommendations()` function (~line 100).
Replace the simulation body with your real model call:

```python
from model_training_evaluation import recommend

def get_recommendations(username, order_count, top_n=10):
    result = recommend(customer_unique_id=username, top_n=top_n)
    recs = result["recommendations"].to_dict("records")
    for r in recs:
        r.setdefault("name", r.get("product_id", ""))
        r.setdefault("price", r.get("avg_price", 0))
        r.setdefault("score", r.get("bayesian_score", 0))
        r.setdefault("ratings", r.get("num_ratings", 0))
    return {
        "segment"    : result["segment"],
        "segment_id" : result["segment"][0],   # "A", "B", "C", or "D"
        "strategy"   : result["strategy"],
        "recs"       : recs,
        "order_count": order_count,
    }
```

### 4. Run
```bash
cd recommender_ui
python app.py
# Open http://localhost:5050
```

## How It Works

### User Flow
1. New visitor → **Landing page** (sees segment overview)
2. Click "Get Started" → **Register** (username, email, password)
3. After register → **Login**
4. After login → **Dashboard** (recommendations shown immediately)
5. Click "Buy" → purchase logged → order_count increments → **segment upgrades automatically**
6. Click "View" → product click logged (AJAX, no page reload)
7. Visit **Profile** → see segment journey, progress bar, full interaction log

### Segment Routing
| Orders | Segment | Strategy |
|--------|---------|----------|
| 0      | A       | Global top-rated (Bayesian) |
| 1      | B       | Category-based + price-aware |
| 2–4    | C       | Multi-category + price profiling |
| 5+     | D       | NCF (NeuMF v2) + popularity boost |

### Interaction Tracking
Every event is written to `data/user_interactions.csv`:
```
event_id, user_id, username, event_type, product_id, category, segment, timestamp, session_id
```

Event types: `login`, `logout`, `page_view`, `purchase`, `product_click`

## Customisation

### Change port
```python
# bottom of app.py
app.run(debug=True, port=5050)   # change 5050 to any port
```

### Production deployment
- Set `app.secret_key` to a random 32-byte hex string
- Use `gunicorn` instead of Flask dev server:
  ```bash
  pip install gunicorn
  gunicorn -w 4 -b 0.0.0.0:8000 app:app
  ```
- Replace CSV storage with SQLite or PostgreSQL for concurrent users
