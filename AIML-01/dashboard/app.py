"""Analyst dashboard (presentation tier).

    streamlit run dashboard/app.py        # FRAUD_API_URL defaults to http://localhost:8000

Talks to the scoring API only (never to the database directly), which keeps
the three tiers independently deployable and lets the API enforce validation.
"""

from __future__ import annotations

import os
import time

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("FRAUD_API_URL", "http://localhost:8000")
STATUSES = ["open", "in_review", "confirmed_fraud", "false_positive"]

st.set_page_config(page_title="Fraud Ops Console", page_icon=":shield:", layout="wide")


@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=5.0)


def get(path: str, **params):
    try:
        r = client().get(path, params=params)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        st.error(f"API error on {path}: {exc}")
        return None


# ---------------------------------------------------------------- header
ready = get("/ready") or {}
stats = get("/v1/cases/stats") or {}
st.title("Fraud Operations Console")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Open alerts", stats.get("open", 0))
c2.metric("In review", stats.get("in_review", 0))
c3.metric("Confirmed fraud", stats.get("confirmed_fraud", 0))
c4.metric("False positives", stats.get("false_positive", 0))
c5.metric("p99 latency", f"{ready.get('p99_ms', 0):.1f} ms", help=f"model {ready.get('model_version', '?')}")

alerts_tab, case_tab, rings_tab, score_tab = st.tabs(["Alerts", "Case review", "Fraud rings", "Score a transaction"])

# ---------------------------------------------------------------- alerts
with alerts_tab:
    left, right = st.columns([1, 3])
    status = left.selectbox("Status", STATUSES, index=0)
    min_score = left.slider("Minimum score", 0.0, 1.0, 0.0, 0.05)
    auto = left.toggle("Auto-refresh (5s)", value=False)
    cases = get("/v1/cases", status=status, limit=200, min_score=min_score) or []
    if cases:
        df = pd.DataFrame(cases)
        df["top_reason"] = df["reasons"].map(lambda r: r[0]["feature"] if r else "")
        right.dataframe(
            df[["id", "transaction_id", "user_id", "amount", "fraud_score", "xgb_score",
                "graph_score", "decision", "top_reason", "created_at"]],
            column_config={"fraud_score": st.column_config.ProgressColumn("risk", min_value=0, max_value=1)},
            hide_index=True, use_container_width=True)
    else:
        right.info("No cases match the filter.")
    if auto:
        time.sleep(5)
        st.rerun()

# ---------------------------------------------------------------- case review
with case_tab:
    case_id = st.number_input("Case id", min_value=1, step=1)
    case = get(f"/v1/cases/{int(case_id)}") if case_id else None
    if case:
        a, b = st.columns(2)
        a.subheader(f"Case #{case['id']} - {case['transaction_id']}")
        a.write(f"**User** `{case['user_id']}`  |  **Amount** {case['amount']:,.2f}")
        a.write(f"**Score** {case['fraud_score']:.3f} (xgb {case['xgb_score']:.3f}, graph {case['graph_score']:.3f})")
        a.write("**Why flagged** (SHAP contributions, log-odds)")
        a.bar_chart(pd.DataFrame(case["reasons"]).set_index("feature") if case["reasons"] else pd.DataFrame())
        risk = get(f"/v1/users/{case['user_id']}/risk") or {}
        if risk.get("ring_id") is not None:
            a.warning(f"User belongs to fraud ring #{risk['ring_id']}")
        with b.form("label"):
            new_status = st.radio("Verdict", STATUSES, index=STATUSES.index(case["status"]))
            analyst = st.text_input("Analyst", value=case.get("analyst") or "")
            note = st.text_area("Note", value=case.get("note") or "")
            if st.form_submit_button("Save feedback label"):
                r = client().patch(f"/v1/cases/{case['id']}",
                                   json={"status": new_status, "analyst": analyst or None, "note": note or None})
                if r.is_success:
                    st.success("Saved - label will be used in the next retraining run")
                else:
                    st.error(r.text)

# ---------------------------------------------------------------- rings
with rings_tab:
    rings = get("/v1/rings", min_size=3) or []
    st.caption(f"{len(rings)} connected high-risk account clusters (GraphSAGE + connected components)")
    if rings:
        rdf = pd.DataFrame(rings)
        st.dataframe(rdf[["ring_id", "size", "mean_score", "max_score"]], hide_index=True)
        pick = st.selectbox("Inspect ring", rdf["ring_id"])
        members = rdf.set_index("ring_id").loc[pick, "members"]
        try:
            import networkx as nx
            import plotly.graph_objects as go

            g = nx.complete_graph(members)
            pos = nx.spring_layout(g, seed=1)
            ex, ey = [], []
            for u, v in g.edges():
                ex += [pos[u][0], pos[v][0], None]
                ey += [pos[u][1], pos[v][1], None]
            fig = go.Figure([
                go.Scatter(x=ex, y=ey, mode="lines", line={"width": 1, "color": "#999"}, hoverinfo="none"),
                go.Scatter(x=[pos[m][0] for m in members], y=[pos[m][1] for m in members],
                           mode="markers+text", text=members, textposition="top center",
                           marker={"size": 16, "color": "#d62728"})])
            fig.update_layout(showlegend=False, height=420, xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.write(members)

# ---------------------------------------------------------------- manual scoring
with score_tab, st.form("score"):
    cols = st.columns(4)
    txn = {
        "transaction_id": cols[0].text_input("Transaction id", f"MAN-{int(time.time())}"),
        "user_id": cols[1].text_input("User", "U1"),
        "card_id": cols[2].text_input("Card", "U1"),
        "device_id": cols[3].text_input("Device", "U1"),
        "merchant_id": cols[0].text_input("Merchant", "M1"),
        "location": cols[1].text_input("Location", "L1"),
        "amount": cols[2].number_input("Amount", min_value=0.01, value=120.0),
        "channel": cols[3].selectbox("Channel", ["CARD_PRESENT", "ONLINE", "TRANSFER", "CASH_OUT", "PAYMENT"]),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    if st.form_submit_button("Score"):
        r = client().post("/v1/score", json=txn)
        st.json(r.json())
