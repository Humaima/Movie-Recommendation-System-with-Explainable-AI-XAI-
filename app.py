"""
XAI Movie Recommender — Streamlit front end
=============================================
A Streamlit UI for the MovieLens-100K explainable recommendation system
(Popularity baseline, Item-based CF, Content-based filtering, Neural CF)
with genre-based and regression/SHAP-style explanations, ported from the
original MovieLen100K.py / xai-based-movie-recommendation-systems notebook.

Run with:
    streamlit run app.py

First run will download the MovieLens-100K dataset (~5MB) from GroupLens
into a local ./ml-100k folder. If your machine has no internet access,
use the "Upload data" option in the sidebar to supply u.data / u.item / u.user
manually.
"""

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Optional heavy deps — degrade gracefully if unavailable
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

np.random.seed(42)

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="XAI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
MOVIE_COLS = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path("ml-100k")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_movielens() -> bool:
    """Download and extract MovieLens-100K if not already present locally."""
    if (DATA_DIR / "u.data").exists():
        return True
    try:
        resp = requests.get(DATA_URL, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(".")
        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_from_bytes(data_bytes, item_bytes, user_bytes):
    ratings = pd.read_csv(
        io.BytesIO(data_bytes), sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    genre_matrix = pd.read_csv(
        io.BytesIO(item_bytes), sep="|", names=MOVIE_COLS + GENRES, encoding="latin-1",
    )
    users = pd.read_csv(
        io.BytesIO(user_bytes), sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    return ratings, users, genre_matrix


@st.cache_data(show_spinner=False)
def load_from_disk():
    ratings = pd.read_csv(DATA_DIR / "u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
    genre_matrix = pd.read_csv(DATA_DIR / "u.item", sep="|", names=MOVIE_COLS + GENRES, encoding="latin-1")
    users = pd.read_csv(DATA_DIR / "u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])
    return ratings, users, genre_matrix


@st.cache_data(show_spinner=False)
def build_item_table(genre_matrix: pd.DataFrame) -> pd.DataFrame:
    movies = genre_matrix[MOVIE_COLS].copy()
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

    item_table = movies[["movie_id", "clean_title", "year"]].copy()
    for genre in GENRES:
        item_table[genre] = genre_matrix[genre]
    return item_table


@st.cache_data(show_spinner=False)
def preprocess_ratings(ratings: pd.DataFrame, min_user=5, min_movie=5) -> pd.DataFrame:
    user_counts = ratings["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_user].index
    ratings = ratings[ratings["user_id"].isin(active_users)]

    movie_counts = ratings["movie_id"].value_counts()
    popular_movies = movie_counts[movie_counts >= min_movie].index
    ratings = ratings[ratings["movie_id"].isin(popular_movies)]
    return ratings


@st.cache_data(show_spinner=False)
def extract_features(ratings: pd.DataFrame, item_table: pd.DataFrame, users: pd.DataFrame):
    genre_cols = [c for c in item_table.columns if c in GENRES]
    movie_genres = item_table[["movie_id"] + genre_cols]

    user_features = users.copy()
    user_features["gender"] = user_features["gender"].map({"M": 0, "F": 1})
    occupation_dummies = pd.get_dummies(user_features["occupation"], prefix="occupation")
    user_features = pd.concat([user_features, occupation_dummies], axis=1)

    merged = pd.merge(ratings, movie_genres, on="movie_id")
    prefs = []
    for uid, group in merged.groupby("user_id"):
        row = {"user_id": uid}
        for genre in genre_cols:
            vals = group.loc[group[genre] == 1, "rating"]
            row[f"pref_{genre}"] = vals.mean() if len(vals) else 0
        prefs.append(row)
    prefs_df = pd.DataFrame(prefs)
    user_features = pd.merge(user_features, prefs_df, on="user_id", how="left")
    return user_features, movie_genres, genre_cols


@st.cache_data(show_spinner=False)
def split_data(ratings: pd.DataFrame):
    return train_test_split(ratings, test_size=0.2, random_state=42, stratify=ratings["user_id"])


@st.cache_data(show_spinner=False)
def compute_movie_similarity(item_table: pd.DataFrame, genre_cols: list) -> pd.DataFrame:
    movie_features = item_table[["movie_id"] + genre_cols].set_index("movie_id")
    sim = cosine_similarity(movie_features)
    return pd.DataFrame(sim, index=movie_features.index, columns=movie_features.index)


# ----------------------------------------------------------------------------
# Recommendation models
# ----------------------------------------------------------------------------
def popularity_baseline(train_ratings: pd.DataFrame, k=10):
    stats = train_ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
    stats = stats.sort_values(["mean", "count"], ascending=[False, False])
    return stats.head(k).index.tolist(), stats


def item_based_cf(train_ratings, movie_sim_df, user_id, k=10):
    user_rated = train_ratings.loc[train_ratings["user_id"] == user_id, "movie_id"].tolist()
    if not user_rated:
        return popularity_baseline(train_ratings, k)[0]

    scores = {}
    for rated in user_rated:
        if rated not in movie_sim_df.index:
            continue
        sims = movie_sim_df[rated]
        for mid, s in sims.items():
            if mid != rated and mid not in user_rated:
                scores[mid] = scores.get(mid, 0) + s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [mid for mid, _ in ranked]


def content_based_filtering(user_id, train_ratings, item_table, genre_cols, user_features, k=10):
    user_prefs = user_features[user_features["user_id"] == user_id]
    if user_prefs.empty:
        return popularity_baseline(train_ratings, k)[0]

    pref_cols = [f"pref_{g}" for g in genre_cols]
    user_vec = user_prefs[pref_cols].values.flatten()
    genre_matrix = item_table[genre_cols].values
    norms = np.linalg.norm(genre_matrix, axis=1) * (np.linalg.norm(user_vec) + 1e-8) + 1e-8
    sims = genre_matrix.dot(user_vec) / norms

    scored = list(zip(item_table["movie_id"].values, sims))
    scored.sort(key=lambda x: x[1], reverse=True)

    rated = set(train_ratings.loc[train_ratings["user_id"] == user_id, "movie_id"])
    return [mid for mid, _ in scored if mid not in rated][:k]


# ----- Neural Collaborative Filtering (optional, torch) --------------------
if TORCH_AVAILABLE:
    class NCFDataset(Dataset):
        def __init__(self, ratings_df):
            self.u = ratings_df["user_idx"].values
            self.m = ratings_df["movie_idx"].values
            self.r = ratings_df["rating"].values.astype("float32")

        def __len__(self):
            return len(self.r)

        def __getitem__(self, idx):
            return self.u[idx], self.m[idx], self.r[idx]

    class SimpleNCF(nn.Module):
        def __init__(self, num_users, num_movies, embedding_dim=32, hidden_dims=(64, 32)):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
            layers = []
            input_dim = embedding_dim * 2
            for h in hidden_dims:
                layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(0.2)]
                input_dim = h
            layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, user_ids, movie_ids):
            u = self.user_embedding(user_ids)
            m = self.movie_embedding(movie_ids)
            x = torch.cat([u, m], dim=1)
            return self.mlp(x).squeeze()


def prepare_ncf_data(ratings: pd.DataFrame):
    unique_users = sorted(ratings["user_id"].unique())
    unique_movies = sorted(ratings["movie_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
    out = ratings.copy()
    out["user_idx"] = out["user_id"].map(user_to_idx)
    out["movie_idx"] = out["movie_id"].map(movie_to_idx)
    return out, len(unique_users), len(unique_movies), user_to_idx, movie_to_idx


@st.cache_resource(show_spinner=False)
def train_ncf_cached(_train_ratings_hash, ncf_ratings_pickle_path, num_users, num_movies, epochs, batch_size):
    """Trained once per (data, epoch) combo; cached as a resource across reruns."""
    ncf_ratings = pd.read_pickle(ncf_ratings_pickle_path)
    dataset = NCFDataset(ncf_ratings)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNCF(num_users, num_movies)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    losses = []
    for _ in range(epochs):
        total = 0.0
        for u, m, r in loader:
            optimizer.zero_grad()
            pred = model(u, m)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total / max(len(loader), 1))
    model.eval()
    return model, losses


def ncf_recommendations(user_id, model, user_to_idx, movie_to_idx, item_table, train_ratings, k=10):
    if user_id not in user_to_idx:
        return popularity_baseline(train_ratings, k)[0]
    user_idx = user_to_idx[user_id]
    rated = set(train_ratings.loc[train_ratings["user_id"] == user_id, "movie_id"])
    candidates = [m for m in item_table["movie_id"].tolist() if m not in rated and m in movie_to_idx][:500]

    scores = []
    with torch.no_grad():
        for mid in candidates:
            u_t = torch.tensor([user_idx])
            m_t = torch.tensor([movie_to_idx[mid]])
            pred = model(u_t, m_t)
            scores.append((mid, pred.item()))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in scores[:k]]


# ----------------------------------------------------------------------------
# Explainability
# ----------------------------------------------------------------------------
def explain_content_recommendation(user_id, movie_id, item_table, user_features, genre_cols):
    user_prefs = user_features[user_features["user_id"] == user_id]
    if user_prefs.empty:
        return []
    pref_cols = [f"pref_{g}" for g in genre_cols]
    user_vec = user_prefs[pref_cols].values.flatten()
    movie_row = item_table[item_table["movie_id"] == movie_id]
    if movie_row.empty:
        return []
    movie_vec = movie_row[genre_cols].values.flatten()

    contributions = user_vec * movie_vec
    scores = [(g, s) for g, s in zip(genre_cols, contributions) if s > 0]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def favorite_genres_for_user(user_id, ratings, item_table, genre_cols, topk=3):
    user_movies = ratings[ratings["user_id"] == user_id].merge(item_table, on="movie_id")
    if user_movies.empty:
        return []
    genre_scores = {}
    for g in genre_cols:
        active = user_movies[user_movies[g] == 1]
        if len(active):
            genre_scores[g] = active["rating"].sum()
    ranked = sorted(genre_scores.items(), key=lambda x: -x[1])
    return [g for g, _ in ranked[:topk]]


def genre_overlap_text(user_genres, movie_row, genre_cols):
    overlap = [g for g in user_genres if movie_row[g] == 1]
    if overlap:
        return f"Matches your favorite genres: {', '.join(overlap)}"
    return "Different from your usual top genres — a bit of a discovery pick."


@st.cache_resource(show_spinner=False)
def train_shap_style_model(_train_ratings_hash, train_sample_pickle_path, item_table, user_features, genre_cols):
    """Small interpretable linear model over genre-interaction features, used
    to produce SHAP-style (or plain linear-coefficient) local explanations."""
    train_sample = pd.read_pickle(train_sample_pickle_path)
    pref_cols = [f"pref_{g}" for g in genre_cols]

    uf = user_features.set_index("user_id")
    it = item_table.set_index("movie_id")

    X, y = [], []
    for _, row in train_sample.iterrows():
        uid, mid = row["user_id"], row["movie_id"]
        if uid not in uf.index or mid not in it.index:
            continue
        user_prefs = uf.loc[uid, pref_cols].values.astype(float)
        movie_genres = it.loc[mid, genre_cols].values.astype(float)
        X.append(user_prefs * movie_genres)
        y.append(row["rating"])

    X = np.array(X)
    y = np.array(y)
    model = LinearRegression().fit(X, y)

    explainer = None
    if SHAP_AVAILABLE:
        try:
            background = X[: min(100, len(X))]
            explainer = shap.Explainer(model, background)
        except Exception:
            explainer = None
    return model, X, explainer


def generate_feature_explanation(user_id, movie_id, model, explainer, item_table, user_features, genre_cols):
    """Returns list of dicts: genre, contribution, user_pref, movie_has_genre.
    Uses real SHAP values if available, otherwise linear-model coefficient
    contributions (coef * feature value), which are mathematically equivalent
    for a linear model."""
    user_row = user_features[user_features["user_id"] == user_id]
    movie_row = item_table[item_table["movie_id"] == movie_id]
    if user_row.empty or movie_row.empty:
        return []

    pref_cols = [f"pref_{g}" for g in genre_cols]
    user_prefs = user_row[pref_cols].values.flatten()
    movie_genres = movie_row[genre_cols].values.flatten()
    interaction = (user_prefs * movie_genres).reshape(1, -1)

    if explainer is not None:
        try:
            contributions = explainer(interaction).values[0]
        except Exception:
            contributions = model.coef_ * interaction.flatten()
    else:
        contributions = model.coef_ * interaction.flatten()

    out = []
    for i, genre in enumerate(genre_cols):
        if interaction.flatten()[i] > 0:
            out.append({
                "genre": genre,
                "contribution": float(contributions[i]),
                "user_pref": float(user_prefs[i]),
                "movie_has_genre": bool(movie_genres[i]),
            })
    out.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return out


def generate_layered_explanation(user_id, movie_id, item_table, user_features, genre_cols,
                                  model, explainer, level="short"):
    movie_title = item_table.loc[item_table["movie_id"] == movie_id, "clean_title"].iloc[0]
    content_expl = explain_content_recommendation(user_id, movie_id, item_table, user_features, genre_cols)

    if level == "short":
        top = [g for g, _ in content_expl[:2]]
        if top:
            return f"Recommended because you tend to enjoy **{', '.join(top)}** movies."
        return "Recommended based on your overall rating history."

    if level == "detailed":
        lines = [f"**{movie_title}** — genre-match scores:"]
        for g, s in content_expl[:5]:
            lines.append(f"- {g}: {s:.3f}")
        return "\n".join(lines) if len(lines) > 1 else "No strong genre overlap found."

    # technical
    feats = generate_feature_explanation(user_id, movie_id, model, explainer, item_table, user_features, genre_cols)
    if not feats:
        return "No technical (model-based) explanation available for this pair."
    lines = [f"Model contributions ({'SHAP' if explainer is not None else 'linear coefficients'}):"]
    for i, f in enumerate(feats[:5], 1):
        lines.append(f"{i}. {f['genre']}: {f['contribution']:+.3f} (your pref: {f['user_pref']:.2f})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate_models(model_funcs: dict, train_ratings, test_ratings, item_table, genre_cols, k=10, n_users=30):
    test_users = test_ratings["user_id"].unique()[:n_users]
    metrics = {name: {"precision": [], "recall": [], "hit_rate": [], "ndcg": [], "diversity": [], "novelty": []}
               for name in model_funcs}

    for uid in test_users:
        user_test_movies = set(test_ratings.loc[test_ratings["user_id"] == uid, "movie_id"])
        if not user_test_movies:
            continue
        for name, func in model_funcs.items():
            try:
                recs = func(uid)
            except Exception:
                continue
            hits = len(set(recs) & user_test_movies)
            precision = hits / k
            recall = hits / len(user_test_movies)
            hit_rate = 1 if hits > 0 else 0

            dcg = sum(1 / np.log2(i + 2) for i, mid in enumerate(recs) if mid in user_test_movies)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(user_test_movies), k)))
            ndcg = dcg / idcg if idcg > 0 else 0

            rec_genres = []
            for mid in recs:
                row = item_table.loc[item_table["movie_id"] == mid, genre_cols]
                if not row.empty:
                    rec_genres.extend([g for g in genre_cols if row[g].iloc[0] == 1])
            diversity = len(set(rec_genres)) / len(rec_genres) if rec_genres else 0

            novelty_scores = []
            for mid in recs:
                pop = len(train_ratings[train_ratings["movie_id"] == mid])
                novelty_scores.append(1 / (1 + pop))
            novelty = float(np.mean(novelty_scores)) if novelty_scores else 0

            metrics[name]["precision"].append(precision)
            metrics[name]["recall"].append(recall)
            metrics[name]["hit_rate"].append(hit_rate)
            metrics[name]["ndcg"].append(ndcg)
            metrics[name]["diversity"].append(diversity)
            metrics[name]["novelty"].append(novelty)

    results = {}
    for name, m in metrics.items():
        if m["precision"]:
            results[name] = {
                "Precision@K": np.mean(m["precision"]),
                "Recall@K": np.mean(m["recall"]),
                "HitRate@K": np.mean(m["hit_rate"]),
                "NDCG@K": np.mean(m["ndcg"]),
                "Diversity": np.mean(m["diversity"]),
                "Novelty": np.mean(m["novelty"]),
            }
    return pd.DataFrame(results).T


# ----------------------------------------------------------------------------
# Sidebar — data loading
# ----------------------------------------------------------------------------
st.sidebar.title("🎬 XAI Movie Recommender")
st.sidebar.caption("MovieLens-100K · Popularity · Item-CF · Content-Based · Neural CF")

with st.sidebar.expander("📁 Dataset", expanded=not (DATA_DIR / "u.data").exists()):
    data_mode = st.radio(
        "Data source", ["Auto-download (GroupLens)", "Upload manually"],
        help="MovieLens-100K is a small public research dataset (~5MB).",
    )
    ratings_raw = users_raw = genre_matrix_raw = None

    if data_mode == "Auto-download (GroupLens)":
        if st.button("Download / load dataset", type="primary"):
            with st.spinner("Downloading MovieLens-100K..."):
                ok = download_movielens()
            if ok:
                st.success("Dataset ready.")
            else:
                st.error("Download failed — check network access, or switch to manual upload.")
        if (DATA_DIR / "u.data").exists():
            ratings_raw, users_raw, genre_matrix_raw = load_from_disk()
    else:
        u_data = st.file_uploader("u.data", type=None)
        u_item = st.file_uploader("u.item", type=None)
        u_user = st.file_uploader("u.user", type=None)
        if u_data and u_item and u_user:
            ratings_raw, users_raw, genre_matrix_raw = load_from_bytes(
                u_data.getvalue(), u_item.getvalue(), u_user.getvalue()
            )

if ratings_raw is None or users_raw is None or genre_matrix_raw is None:
    st.title("🎬 XAI Movie Recommender")
    st.info(
        "👈 Load the MovieLens-100K dataset from the sidebar to get started "
        "(auto-download, or upload `u.data` / `u.item` / `u.user` manually)."
    )
    st.stop()

assert users_raw is not None and ratings_raw is not None and genre_matrix_raw is not None

# ----------------------------------------------------------------------------
# Pipeline (cached)
# ----------------------------------------------------------------------------
item_table = build_item_table(genre_matrix_raw)
filtered_ratings = preprocess_ratings(ratings_raw)
user_features, movie_genres, genre_cols = extract_features(filtered_ratings, item_table, users_raw)
train_ratings, test_ratings = split_data(filtered_ratings)
movie_sim_df = compute_movie_similarity(item_table, genre_cols)

all_user_ids = sorted(train_ratings["user_id"].unique().tolist())

# ----------------------------------------------------------------------------
# Sidebar — controls
# ----------------------------------------------------------------------------
st.sidebar.divider()
user_id = st.sidebar.selectbox("Select user", all_user_ids, index=0)
k = st.sidebar.slider("Number of recommendations (K)", 5, 20, 10)

st.sidebar.divider()
st.sidebar.subheader("Models to compare")
use_pop = st.sidebar.checkbox("Popularity baseline", value=True)
use_itemcf = st.sidebar.checkbox("Item-based CF", value=True)
use_content = st.sidebar.checkbox("Content-based filtering", value=True)
use_ncf = st.sidebar.checkbox("Neural CF (slower, trains a small model)", value=False, disabled=not TORCH_AVAILABLE)
if not TORCH_AVAILABLE:
    st.sidebar.caption("⚠️ PyTorch not installed — Neural CF unavailable. `pip install torch`")

ncf_model = None
user_to_idx = movie_to_idx = None
if use_ncf and TORCH_AVAILABLE:
    epochs = st.sidebar.slider("NCF training epochs", 1, 20, 5)
    ncf_ratings, num_users, num_movies, user_to_idx, movie_to_idx = prepare_ncf_data(train_ratings)
    pkl_path = "ncf_ratings.pkl"
    ncf_ratings.to_pickle(pkl_path)
    data_hash = f"{len(train_ratings)}-{epochs}"
    with st.spinner(f"Training Neural CF for {epochs} epoch(s)..."):
        ncf_model, ncf_losses = train_ncf_cached(data_hash, pkl_path, num_users, num_movies, epochs, 64)
    st.sidebar.success(f"NCF trained · final loss {ncf_losses[-1]:.3f}")

# Shared "explainer" model (genre-interaction linear model)
train_sample = train_ratings.head(1000)
sample_pkl = "train_sample.pkl"
train_sample.to_pickle(sample_pkl)
shap_model, shap_X, shap_explainer = train_shap_style_model(
    len(train_ratings), sample_pkl, item_table, user_features, genre_cols
)

# ----------------------------------------------------------------------------
# Build recommendation sets for the selected user
# ----------------------------------------------------------------------------
model_recs = {}
if use_pop:
    model_recs["Popularity"] = popularity_baseline(train_ratings, k)[0]
if use_itemcf:
    model_recs["Item-CF"] = item_based_cf(train_ratings, movie_sim_df, user_id, k)
if use_content:
    model_recs["Content-Based"] = content_based_filtering(user_id, train_ratings, item_table, genre_cols, user_features, k)
if use_ncf and ncf_model is not None:
    model_recs["Neural CF"] = ncf_recommendations(user_id, ncf_model, user_to_idx, movie_to_idx, item_table, train_ratings, k)

user_top_genres = favorite_genres_for_user(user_id, train_ratings, item_table, genre_cols, topk=3)

# ----------------------------------------------------------------------------
# Header / user profile
# ----------------------------------------------------------------------------
st.title("🎬 Explainable Movie Recommender")
st.caption("MovieLens-100K · four recommendation strategies, side by side, each with a plain-language *why*.")

profile = users_raw[users_raw["user_id"] == user_id].iloc[0]
n_ratings = len(train_ratings[train_ratings["user_id"] == user_id])
c1, c2, c3, c4 = st.columns(4)
c1.metric("User", f"#{user_id}")
c2.metric("Age / Gender", f"{profile['age']} / {profile['gender']}")
c3.metric("Occupation", profile["occupation"])
c4.metric("Ratings given", n_ratings)
if user_top_genres:
    st.caption(f"⭐ Favorite genres (from rating history): **{', '.join(user_top_genres)}**")

tab_recs, tab_explain, tab_eval, tab_eda = st.tabs(
    ["🎬 Recommendations", "🔍 Explainability", "📊 Model Evaluation", "📈 Data Explorer"]
)

# ----------------------------------------------------------------------------
# Tab 1 — Recommendations
# ----------------------------------------------------------------------------
with tab_recs:
    if not model_recs:
        st.warning("Select at least one model in the sidebar.")
    else:
        cols = st.columns(len(model_recs))
        for col, (name, recs) in zip(cols, model_recs.items()):
            with col:
                st.subheader(name)
                if not recs:
                    st.write("No recommendations available.")
                    continue
                for rank, mid in enumerate(recs, 1):
                    row = item_table[item_table["movie_id"] == mid]
                    if row.empty:
                        continue
                    title = row["clean_title"].iloc[0]
                    year = row["year"].iloc[0]
                    year_str = f" ({int(year)})" if pd.notna(year) else ""
                    short_expl = generate_layered_explanation(
                        user_id, mid, item_table, user_features, genre_cols,
                        shap_model, shap_explainer, level="short",
                    ) if name in ("Content-Based", "Item-CF") else (
                        "Popular among all users." if name == "Popularity"
                        else "Personalized neural prediction from learned user/movie embeddings."
                    )
                    with st.container(border=True):
                        st.markdown(f"**{rank}. {title}{year_str}**")
                        st.caption(short_expl)

# ----------------------------------------------------------------------------
# Tab 2 — Explainability deep-dive
# ----------------------------------------------------------------------------
with tab_explain:
    st.subheader("Explanation dashboard")
    st.caption("Pick a recommended movie to see why it was suggested, at three levels of detail.")

    flat_choices = []
    for name, recs in model_recs.items():
        for mid in recs:
            row = item_table[item_table["movie_id"] == mid]
            if not row.empty:
                flat_choices.append((f"{row['clean_title'].iloc[0]}  ·  ({name})", mid))

    if not flat_choices:
        st.info("No recommendations to explain yet — enable a model and pick a user.")
    else:
        labels = [c[0] for c in flat_choices]
        chosen_label = st.selectbox("Movie", labels)
        chosen_mid = dict(flat_choices)[chosen_label]

        level = st.radio("Explanation depth", ["short", "detailed", "technical"], horizontal=True)
        expl_text = generate_layered_explanation(
            user_id, chosen_mid, item_table, user_features, genre_cols,
            shap_model, shap_explainer, level=level,
        )
        st.info(expl_text)

        content_expl = explain_content_recommendation(user_id, chosen_mid, item_table, user_features, genre_cols)
        feat_expl = generate_feature_explanation(
            user_id, chosen_mid, shap_model, shap_explainer, item_table, user_features, genre_cols
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Content-based genre importance**")
            if content_expl:
                df = pd.DataFrame(content_expl[:8], columns=["genre", "score"])
                fig = px.bar(df, x="score", y="genre", orientation="h", color_discrete_sequence=["#4C78A8"])
                fig.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No genre overlap for this pick.")

        with col_b:
            title_label = "SHAP feature contributions" if shap_explainer is not None else "Linear-model feature contributions"
            st.markdown(f"**{title_label}**")
            if feat_expl:
                df = pd.DataFrame(feat_expl[:8])
                colors = ["#2CA02C" if v > 0 else "#D62728" for v in df["contribution"]]
                fig = go.Figure(go.Bar(x=df["contribution"], y=df["genre"], orientation="h", marker_color=colors))
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Impact on predicted rating")
                fig.add_vline(x=0, line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No model-based explanation available for this pair.")

        with st.expander("Why these numbers? (methodology)"):
            st.markdown(
                "- **Content-based genre importance** multiplies your average rating for each genre "
                "by whether the movie belongs to that genre — genres you rate highly *and* that the "
                "movie has both push its score up.\n"
                "- **Feature contributions** come from a small interpretable linear regression trained "
                "on genre-interaction features"
                + (" (via SHAP)" if shap_explainer is not None else " (via the model's own coefficients, "
                   "which are equivalent to SHAP values for a linear model)")
                + " — green bars push the predicted rating up, red bars pull it down."
            )

# ----------------------------------------------------------------------------
# Tab 3 — Evaluation
# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Offline evaluation")
    st.caption("Precision/Recall/HitRate/NDCG measure accuracy against held-out ratings; "
               "Diversity and Novelty measure how varied and non-obvious the recommendations are.")

    n_eval_users = st.slider("Number of test users to evaluate on", 10, 100, 30, step=10)
    run_eval = st.button("Run evaluation", type="primary")

    if run_eval:
        funcs = {}
        if use_pop:
            funcs["Popularity"] = lambda uid: popularity_baseline(train_ratings, k)[0]
        if use_itemcf:
            funcs["Item-CF"] = lambda uid: item_based_cf(train_ratings, movie_sim_df, uid, k)
        if use_content:
            funcs["Content-Based"] = lambda uid: content_based_filtering(uid, train_ratings, item_table, genre_cols, user_features, k)
        if use_ncf and ncf_model is not None:
            funcs["Neural CF"] = lambda uid: ncf_recommendations(uid, ncf_model, user_to_idx, movie_to_idx, item_table, train_ratings, k)

        if not funcs:
            st.warning("Select at least one model in the sidebar first.")
        else:
            with st.spinner("Evaluating..."):
                results = evaluate_models(funcs, train_ratings, test_ratings, item_table, genre_cols, k=k, n_users=n_eval_users)
            if results.empty:
                st.warning("No metrics could be computed — try more test users or different models.")
            else:
                st.dataframe(results.round(4), use_container_width=True)
                metric_choice = st.selectbox("Chart metric", results.columns.tolist())
                fig = px.bar(results.reset_index().rename(columns={"index": "Model"}),
                             x="Model", y=metric_choice, color="Model")
                fig.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click **Run evaluation** to score the selected models on held-out test ratings.")

# ----------------------------------------------------------------------------
# Tab 4 — Data explorer / EDA
# ----------------------------------------------------------------------------
with tab_eda:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ratings", f"{len(filtered_ratings):,}")
    c2.metric("Users", f"{filtered_ratings['user_id'].nunique():,}")
    c3.metric("Movies", f"{filtered_ratings['movie_id'].nunique():,}")
    sparsity = 1 - len(filtered_ratings) / (filtered_ratings["user_id"].nunique() * filtered_ratings["movie_id"].nunique())
    c4.metric("Sparsity", f"{sparsity:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Rating distribution**")
        fig = px.histogram(filtered_ratings, x="rating", nbins=5, color_discrete_sequence=["#4C78A8"])
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Ratings per user (distribution)**")
        counts = filtered_ratings["user_id"].value_counts()
        fig = px.histogram(counts, nbins=40, color_discrete_sequence=["#F58518"])
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, xaxis_title="Ratings per user")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Movies per genre**")
    genre_counts = item_table[genre_cols].sum().sort_values(ascending=False)
    fig = px.bar(genre_counts, orientation="h", color_discrete_sequence=["#54A24B"])
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                       yaxis=dict(autorange="reversed"), xaxis_title="Number of movies", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Browse raw tables"):
        st.markdown("**Ratings sample**")
        st.dataframe(filtered_ratings.head(20), use_container_width=True)
        st.markdown("**Movies sample**")
        st.dataframe(item_table.head(20), use_container_width=True)
        st.markdown("**User features sample**")
        st.dataframe(user_features.head(20), use_container_width=True)