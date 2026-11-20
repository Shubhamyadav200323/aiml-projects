"""Recommendation demo console (presentation tier).

    streamlit run ui/app.py           # RECSYS_API_URL defaults to http://localhost:8100

Talks only to the API gateway. Clicking "Watch" logs a click event, which feeds
the live A/B experiment report.
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("RECSYS_API_URL", "http://localhost:8100")
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

st.set_page_config(page_title="Recommender Console", page_icon=":clapper:", layout="wide")


@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=5.0)


def call(method: str, path: str, **kw):
    try:
        r = client().request(method, path, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        st.error(f"{path}: {exc}")
        return None


ready = call("GET", "/ready") or {}
st.title("Personalised Recommendations")
st.caption(f"index: {ready.get('index', '?')} · {ready.get('items', 0):,} items · features as of "
           f"{ready.get('features_as_of', '?')} · p99 {ready.get('p99_ms', 0):.1f} ms")

rec_tab, sim_tab, ab_tab = st.tabs(["For you", "Similar items", "A/B experiment"])

with rec_tab:
    c1, c2, c3 = st.columns([2, 3, 1])
    user = c1.text_input("User id", "u1", help="try an unknown id like 'new_user' to see cold start")
    genres = c2.multiselect("Onboarding genres (cold-start users)", GENRES)
    k = c3.number_input("k", 1, 50, 12)
    data = call("GET", f"/v1/recommendations/{user}", params={"k": k, "genres": genres or None})
    if data:
        badge = {"two_stage": "two-tower + FAISS + ranker", "content": "content-based cold start",
                 "popularity": "trending (cold start)"}[data["strategy"]]
        st.info(f"Strategy: **{badge}**" + (f" · ranker variant **{data['variant']}**" if data["variant"] else "")
                + f" · {'cache hit' if data['cached'] else 'computed'} in {data['latency_ms']:.1f} ms")
        cols = st.columns(4)
        for pos, it in enumerate(data["items"]):
            with cols[pos % 4].container(border=True):
                st.markdown(f"**{it['title']}**")
                st.caption(" · ".join(it["genres"]) or "-")
                st.progress(min(max(it["score"], 0.0), 1.0))
                if st.button("Watch", key=f"w{pos}{it['item_id']}"):
                    call("POST", "/v1/events/click", json={"user_id": user, "item_id": it["item_id"],
                                                          "position": pos, "request_id": data["request_id"]})
                    st.toast("click logged")

with sim_tab:
    item = st.text_input("Item id", "i1")
    sims = call("GET", f"/v1/items/{item}/similar", params={"k": 10}) or []
    if sims:
        st.dataframe(pd.DataFrame(sims), hide_index=True, use_container_width=True)

with ab_tab:
    rep = call("GET", "/v1/experiments/report")
    if rep:
        v = pd.DataFrame(rep["variants"]).T
        st.dataframe(v, use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Relative CTR lift (B vs A)", f"{rep['lift_rel']:+.1%}")
        m2.metric("p-value", f"{rep['p_value']:.3f}")
        m3.metric("Winner", rep["winner"] or "no significant difference")
        st.caption(f"95% CI on CTR difference: [{rep['ci95_low']:+.4f}, {rep['ci95_high']:+.4f}]")
