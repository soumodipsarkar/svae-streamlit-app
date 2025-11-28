# svae_app.py
# S.V.A.E. — Smart Volume Allocation Engine (Prototype)
# Manual-upload-only Streamlit MVP with compact weights + static Legend tab + About tab
#

import streamlit as st
import pandas as pd
import numpy as np
import io
import os

st.set_page_config(page_title="S.V.A.E. — Allocation Prototype", layout="wide")
st.title("S.V.A.E. — Smart Volume Allocation Engine (Prototype)")

# ---------------- Sidebar: Uploads ----------------
st.sidebar.header("Upload required files (manual only)")
uploaded_vol = st.sidebar.file_uploader("Upload volumes.csv (required)", type=["csv"])
uploaded_agents = st.sidebar.file_uploader("Upload agents.csv (required)", type=["csv"])
uploaded_history = st.sidebar.file_uploader("Upload history.csv (optional)", type=["csv"])

st.sidebar.markdown("---")
if st.sidebar.button("Clear uploaded files & results"):
    for k in ("alloc_results",):
        if k in st.session_state:
            st.session_state.pop(k)
    st.experimental_rerun()

# Helper to read uploaded CSVs
def read_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        return None

# --- Read uploads (kept read-only) ---
volumes_df = read_csv(uploaded_vol)
agents_df = read_csv(uploaded_agents)
history_df = read_csv(uploaded_history)

# ---------------- Tabs: Data / About / Legend ----------------
tab_data, tab_about, tab_legend = st.tabs(["Data & Preview", "About", "Legend"])

# ==========================================================
# Data & Preview 
## ==========================================================
with tab_data:
    st.header("Source Data Status & Preview")

    c1, c2, c3 = st.columns(3)

    def preview_or_hint(col, df, name, required=False):
        if isinstance(df, pd.DataFrame) and not df.empty:
            col.success(f"✅ {name} uploaded")
            col.dataframe(df.head(6))
        else:
            if required:
                col.warning(f"⚠️ {name} is required.")
            else:
                col.info(f"ℹ️ {name} not uploaded (optional).")

    preview_or_hint(c1, volumes_df, "Volumes (volumes.csv)", required=True)
    preview_or_hint(c2, agents_df, "Agents (agents.csv)", required=True)
    preview_or_hint(c3, history_df, "History (history.csv)", required=False)

    st.markdown("---")

    # Compact Weights section
    st.subheader("Allocation Weights")
    st.caption("Adjust weights before running allocation — weights are normalized internally so you can experiment freely.")

    defaults = {
        "w_alpha": 0.35, "w_beta": 0.25, "w_gamma": 0.15,
        "w_delta": 0.15, "w_epsilon": 0.7, "w_zeta": 0.1
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    cols = st.columns([1, 1, 1, 1, 1, 1])
    st.session_state["w_alpha"] = cols[0].number_input("α (SLA urgency)", value=float(st.session_state["w_alpha"]), step=0.05, min_value=0.0, max_value=5.0)
    st.session_state["w_beta"] = cols[1].number_input("β (Priority)", value=float(st.session_state["w_beta"]), step=0.05, min_value=0.0, max_value=5.0)
    st.session_state["w_gamma"] = cols[2].number_input("γ (Agent efficiency)", value=float(st.session_state["w_gamma"]), step=0.05, min_value=0.0, max_value=5.0)
    st.session_state["w_delta"] = cols[3].number_input("δ (Backlog penalty)", value=float(st.session_state["w_delta"]), step=0.05, min_value=0.0, max_value=5.0)
    st.session_state["w_epsilon"] = cols[4].number_input("ε (Complexity mismatch)", value=float(st.session_state["w_epsilon"]), step=0.05, min_value=0.0, max_value=10.0)
    st.session_state["w_zeta"] = cols[5].number_input("ζ (Skill bonus)", value=float(st.session_state["w_zeta"]), step=0.05, min_value=0.0, max_value=5.0)

# ==========================================================
# Functions / Parameters
# ==========================================================
def compute_complexity(row, history_df, max_tat=10.0):
    rc = row.get("refer_code", None)
    if pd.isna(rc) or rc is None or str(rc).strip() == "":
        return 0.3
    if history_df is None or history_df.empty:
        return 0.3
    rec = history_df[history_df["refer_code"] == rc]
    if not rec.empty:
        r = rec.iloc[0]
        try:
            norm_tat = min(float(r.get("avg_tat_days", 0.0)) / float(max_tat), 1.0)
            norm_breach = min(float(r.get("sla_breach_pct", 0.0)) / 100.0, 1.0)
            norm_exc = min(float(r.get("exception_rate", 0.0)), 1.0)
        except Exception:
            return 0.3
        comp = 0.5 * norm_tat + 0.3 * norm_breach + 0.2 * norm_exc
        return float(max(0.0, min(1.0, comp)))
    return 0.3

def sla_urgency_score(row):
    """
    Linear SLA urgency in days:
      sla_urgency = 1 - (remaining_days / total_window_days), clipped to [0,1]
    - If past due -> 1.0
    - If total_window_days <= 0 -> fallback behavior
    """
    try:
        due = pd.to_datetime(row["sla_due_date"])
        received = pd.to_datetime(row.get("received_date", pd.Timestamp.now()))
        now = pd.Timestamp.now()
        total_seconds = (due - received).total_seconds()
        remain_seconds = (due - now).total_seconds()

        total_days = total_seconds / 86400.0
        remain_days = remain_seconds / 86400.0

        # past due
        if remain_seconds <= 0:
            return 1.0

        if total_days > 0:
            val = 1.0 - (remain_days / total_days)
            return float(max(0.0, min(1.0, val)))

        return 0.5
    except Exception:
        try:
            received = pd.to_datetime(row.get("received_date", pd.Timestamp.now()))
            total = (pd.to_datetime(row.get("sla_due_date", pd.Timestamp.now())) - received).total_seconds()
            remain = (pd.to_datetime(row.get("sla_due_date", pd.Timestamp.now())) - pd.Timestamp.now()).total_seconds()
            if total <= 0:
                return 1.0
            return float(max(0.0, 1.0 - (remain / total)))
        except Exception:
            return 0.5

def priority_score(p):
    mapping = {1: 1.0, 2: 0.7, 3: 0.4}
    try:
        return mapping.get(int(p), 0.5)
    except Exception:
        return 0.5

def agent_efficiency(a_row, min_aht_observed=1.0):
    try:
        aht = float(a_row.get("avg_handle_time_mins", 30.0))
        if aht <= 0:
            return 0.0
        eff = min_aht_observed / aht
        return float(max(0.0, min(1.0, eff)))
    except Exception:
        return 0.0

def skill_match_bonus(case_row, agent_row):
    try:
        tags = [t.strip() for t in str(agent_row.get("skill_tags", "")).split(",") if t.strip()]
        refer = str(case_row.get("refer_code") or "").strip()
        ctype = str(case_row.get("case_type") or "").strip()
        bonus = 0.15 if (refer in tags or ctype in tags) else 0.0
        return float(max(0.0, min(1.0, bonus)))
    except Exception:
        return 0.0

def suitability_score(case_row, agent_row, history_df, min_aht_observed=1.0):
    comp = float(case_row.get("complexity", 0.3))
    sla = float(case_row.get("sla_urgency", 0.5))
    pri = float(case_row.get("priority_score", 0.5))

    comp = float(max(0.0, min(1.0, comp)))
    sla = float(max(0.0, min(1.0, sla)))
    pri = float(max(0.0, min(1.0, pri)))

    eff = agent_efficiency(agent_row, min_aht_observed=min_aht_observed)

    try:
        backlog_raw = float(agent_row.get("current_backlog", 0.0))
        capacity = max(1.0, float(agent_row.get("capacity_daily", 1.0)))
        backlog_norm = float(max(0.0, min(1.0, backlog_raw / capacity)))
    except Exception:
        backlog_norm = 0.0

    try:
        skill_level_raw = float(agent_row.get("skill_level", 3.0))
        skill_norm = float(max(0.0, min(1.0, skill_level_raw / 5.0)))
    except Exception:
        skill_norm = 0.6

    mismatch = max(0.0, comp - skill_norm)
    sbonus = skill_match_bonus(case_row, agent_row)

    w = st.session_state
    raw_weights = {
        "alpha": float(w.get("w_alpha", 0.0)),
        "beta": float(w.get("w_beta", 0.0)),
        "gamma": float(w.get("w_gamma", 0.0)),
        "delta": float(w.get("w_delta", 0.0)),
        "epsilon": float(w.get("w_epsilon", 0.0)),
        "zeta": float(w.get("w_zeta", 0.0)),
    }

    positive_sum = raw_weights["alpha"] + raw_weights["beta"] + raw_weights["gamma"] + raw_weights["zeta"]
    penalty_sum = raw_weights["delta"] + raw_weights["epsilon"]

    if positive_sum <= 0:
        positive_sum = 1.0
    if penalty_sum <= 0:
        penalty_sum = 1.0

    w_alpha = raw_weights["alpha"] / positive_sum
    w_beta = raw_weights["beta"] / positive_sum
    w_gamma = raw_weights["gamma"] / positive_sum
    w_zeta = raw_weights["zeta"] / positive_sum

    w_delta = raw_weights["delta"] / penalty_sum
    w_epsilon = raw_weights["epsilon"] / penalty_sum

    positive_contrib = (w_alpha * sla) + (w_beta * pri) + (w_gamma * eff) + (w_zeta * sbonus)
    penalty_contrib = (w_delta * backlog_norm) + (w_epsilon * mismatch)

    penalty_scale = min(1.0, penalty_sum / (positive_sum + 1e-9))
    raw_score = positive_contrib - (penalty_scale * penalty_contrib)
    final = float(max(0.0, min(1.0, raw_score)))

    breakdown = {
        "sla_norm": sla,
        "priority_norm": pri,
        "efficiency_norm": eff,
        "backlog_norm": backlog_norm,
        "skill_norm": skill_norm,
        "mismatch": mismatch,
        "skill_bonus": sbonus,
        "positive_contrib": positive_contrib,
        "penalty_contrib": penalty_contrib,
        "penalty_scale": penalty_scale,
        "raw_score": raw_score,
        "final_score": final
    }

    return final, breakdown

def run_greedy_allocation(vols, agents, history):
    if "priority" not in vols.columns:
        vols["priority"] = 3
    if "refer_code" not in vols.columns:
        vols["refer_code"] = ""
    if "received_date" not in vols.columns:
        vols["received_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    if "sla_due_date" not in vols.columns:
        vols["sla_due_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    if "avg_handle_time_mins" not in agents.columns:
        agents["avg_handle_time_mins"] = 20
    if "capacity_daily" not in agents.columns:
        agents["capacity_daily"] = 10
    if "current_backlog" not in agents.columns:
        agents["current_backlog"] = 0
    if "skill_level" not in agents.columns:
        agents["skill_level"] = 3

    vols["complexity"] = vols.apply(lambda r: compute_complexity(r, history), axis=1)
    vols["sla_urgency"] = vols.apply(lambda r: sla_urgency_score(r), axis=1)
    vols["priority_score"] = vols["priority"].apply(priority_score)
    agents["assigned_count"] = 0

    try:
        min_aht_observed = max(1.0, float(agents['avg_handle_time_mins'].replace(0, np.nan).min(skipna=True)))
        if np.isnan(min_aht_observed) or min_aht_observed <= 0:
            min_aht_observed = 1.0
    except Exception:
        min_aht_observed = 1.0

    results = []
    for _, case in vols.iterrows():
        best_agent = None
        best_score = -1.0
        best_breakdown = None
        for _, agent in agents.iterrows():
            try:
                if int(agent["assigned_count"]) >= int(agent["capacity_daily"]):
                    continue
            except Exception:
                pass
            score, breakdown = suitability_score(case, agent, history, min_aht_observed=min_aht_observed)
            if score > best_score:
                best_score = score
                best_agent = agent
                best_breakdown = breakdown
        if best_agent is not None:
            results.append({
                "case_id": case.get("case_id"),
                "assigned_agent": best_agent.get("agent_id"),
                "score": round(float(best_score), 3),
                "complexity": float(case.get("complexity")),
                "expected_handle_time_mins": best_agent.get("avg_handle_time_mins"),
                "assigned_at": pd.Timestamp.now().isoformat(),
                "sla_norm": round(float(case.get("sla_urgency")), 3),
                "priority_norm": round(float(case.get("priority_score")), 3),
                "efficiency_norm": round(float(best_breakdown.get("efficiency_norm", 0.0)), 3),
                "backlog_norm": round(float(best_breakdown.get("backlog_norm", 0.0)), 3),
                "skill_norm": round(float(best_breakdown.get("skill_norm", 0.0)), 3),
                "mismatch": round(float(best_breakdown.get("mismatch", 0.0)), 3),
                "skill_bonus": round(float(best_breakdown.get("skill_bonus", 0.0)), 3),
            })
            agents.loc[agents["agent_id"] == best_agent["agent_id"], "assigned_count"] += 1
        else:
            results.append({
                "case_id": case.get("case_id"),
                "assigned_agent": None,
                "score": None,
                "complexity": float(case.get("complexity")),
                "expected_handle_time_mins": None,
                "assigned_at": pd.Timestamp.now().isoformat(),
                "sla_norm": round(float(case.get("sla_urgency")), 3),
                "priority_norm": round(float(case.get("priority_score")), 3),
                "efficiency_norm": None,
                "backlog_norm": None,
                "skill_norm": None,
                "mismatch": None,
                "skill_bonus": None,
            })
    return pd.DataFrame(results)

# ==========================================================
# ---------- About tab: equation, metric definitions, interactive example, audit view ----------
with tab_about:
    st.markdown("## Allocation equation ")
    st.markdown(
        "**Score = α × SLA + β × Priority + γ × Efficiency − scaled_penalty × ( δ × Backlog + ε × ComplexityMismatch ) + ζ × SkillBonus**"
    )
    st.markdown("Weights (α..ζ) are set in the Weights panel. The engine normalizes positive and penalty groups so they are comparable.")

    st.markdown("---")
    st.markdown("## Metric definitions — what gives highest / lowest value")
    st.markdown(
        "- **SLA urgency (0.00 → 1.00):** 1.00 means at or past SLA breach; 0.00 means plenty of time left. (Calculated linearly in days: `1 - remaining_days / total_window_days`.)\n"
        "- **Priority (0.00 → 1.00):** mapped (VIP→1.00, Medium→0.70, Normal→0.40).\n"
        "- **Complexity (0.00 → 1.00):** derived from historical avg TAT, breach %, exceptions — 1.00 = very complex.\n"
        "- **Agent Efficiency (0.00 → 1.00):** fastest_AHT / agent_AHT — 1.00 = fastest agent.\n"
        "- **Backlog (0.00 → 1.00):** current_backlog / capacity_daily — 1.00 = at/over capacity.\n"
        "- **SkillBonus (small, e.g., 0.15):** added if tags match case refer_code or case_type.\n"
        "- **ComplexityMismatch (0.00 → 1.00):** max(0, complexity - skill_norm) — penalty when case > agent skill.\n"
    )

    st.markdown("---")
    st.header("Worked example — Case #1234 ")

    # Default example values
    ex_defaults = {
        "case_id": "1234",
        "refer_code": "RC_PAY",
        "case_type": "payment",
        "priority": 1,
        "complexity": None,  # computed from temp history if left None
        "agent_id": "Agent_A",
        "agent_aht": 20.0,
        "agent_capacity": 10,
        "agent_backlog": 2,
        "agent_skill_level": 4,
        "agent_skill_tags": "RC_PAY,payment",
        "fastest_aht_assumed": 10.0,
        "original_sla_window_days": 7.0
    }

    st.markdown("### Case details ")
    col1, col2 = st.columns(2)
    with col1:
        case_id_in = st.text_input("Case ID", value=ex_defaults["case_id"])
        refer_in = st.text_input("Refer code", value=ex_defaults["refer_code"])
        ctype_in = st.text_input("Case type", value=ex_defaults["case_type"])
        priority_in = st.selectbox("Priority (1=VIP)", options=[1,2,3], index=0)
    with col2:
        sla_remaining_days = st.number_input(
            "SLA due in (days from now) — remaining_days",
            value=1.0, min_value=0.0, step=0.01)
        original_window_days = st.number_input(
            "Original SLA window (days) — total_window_days (received → due)",
            value=float(ex_defaults["original_sla_window_days"]),
            min_value=0.01, step=0.01)
        complexity_manual = st.number_input("Complexity (leave 0 to compute from history)", value=0.60, min_value=0.0, max_value=1.0, step=0.01)

    st.markdown("### Agent details (editable for demo)")
    a1, a2, a3 = st.columns(3)
    with a1:
        agent_id_in = st.text_input("Agent ID", value=ex_defaults["agent_id"])
        aht_in = st.number_input("Agent AHT (mins)", value=float(ex_defaults["agent_aht"]), min_value=1.0)
    with a2:
        backlog_in = st.number_input("Agent backlog (cases)", value=int(ex_defaults["agent_backlog"]), min_value=0)
        capacity_in = st.number_input("Agent daily capacity", value=int(ex_defaults["agent_capacity"]), min_value=1)
    with a3:
        skill_level_in = st.slider("Agent skill level (1–5)", min_value=1, max_value=5, value=ex_defaults["agent_skill_level"])
        tags_in = st.text_input("Agent tags (comma separated)", value=ex_defaults["agent_skill_tags"])
        fastest_aht = st.number_input("Assumed fastest AHT in pool (mins)", value=float(ex_defaults["fastest_aht_assumed"]), min_value=1.0)

    recompute = st.button("Recompute example")

    # assemble local case & agent dicts (local only)
    # compute received_date such that total_window_days = original_window_days and remaining_days = sla_remaining_days
    # clamp remaining_days to [0, original_window_days]
    rem = float(max(0.0, sla_remaining_days))
    total_win = float(max(0.01, original_window_days))
    if rem > total_win:
        rem = total_win  # if remaining > original window, cap to original window (treated as just arrived)

    # received_date should be now - (total_win - rem) days
    received_dt = pd.Timestamp.now() - pd.Timedelta(days=(total_win - rem))
    due_dt = pd.Timestamp.now() + pd.Timedelta(days=rem)

    example_case_local = {
        "case_id": case_id_in,
        "refer_code": refer_in,
        "case_type": ctype_in,
        "priority": priority_in,
        "received_date": received_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "sla_due_date": due_dt.strftime("%Y-%m-%d %H:%M:%S")
    }
    example_agent_local = {
        "agent_id": agent_id_in,
        "avg_handle_time_mins": aht_in,
        "capacity_daily": capacity_in,
        "current_backlog": backlog_in,
        "skill_level": skill_level_in,
        "skill_tags": tags_in
    }

    # local history for computing complexity 
    local_history = pd.DataFrame([{
        "refer_code": example_case_local["refer_code"],
        "avg_tat_days": 3.0,
        "sla_breach_pct": 20.0,
        "exception_rate": 0.05
    }])

    try:
        if isinstance(agents_df, pd.DataFrame) and not agents_df.empty:
            min_aht_observed = max(1.0, float(agents_df['avg_handle_time_mins'].replace(0, np.nan).min(skipna=True)))
            if np.isnan(min_aht_observed) or min_aht_observed <= 0:
                min_aht_observed = float(fastest_aht)
        else:
            min_aht_observed = float(fastest_aht)
    except Exception:
        min_aht_observed = float(fastest_aht)

    # compute complexity 
    if complexity_manual is not None and complexity_manual > 0:
        example_case_local["complexity"] = float(max(0.0, min(1.0, complexity_manual)))
    else:
        example_case_local["complexity"] = compute_complexity(example_case_local, local_history, max_tat=10.0)

    # compute sla_urgency using the updated received_date/due_date
    example_case_local["sla_urgency"] = sla_urgency_score(example_case_local)
    example_case_local["priority_score"] = priority_score(example_case_local["priority"])

    # compute score and breakdown using exact allocator function (local only)
    example_score_val, example_breakdown = suitability_score(example_case_local, example_agent_local, local_history, min_aht_observed=min_aht_observed)

    # round for display
    disp = {k: (round(v,2) if isinstance(v, (int,float,np.floating)) else v) for k,v in example_breakdown.items()}
    final_score_display = round(disp["final_score"], 2)

   
    st.markdown("### Computation ")
    st.markdown(
        f"**Assigned Agent:** {example_agent_local['agent_id']}   \n"
        f"**Final suitability score:** **{final_score_display}**   \n"
        f"**Expected handle time:** {example_agent_local['avg_handle_time_mins']} mins"
    )

    st.markdown("---")
    st.markdown("### At-a-glance ")
    df_at_glance = pd.DataFrame({
        "Metric": ["Priority","SLA urgency","Complexity","Agent skill","Agent backlog","Efficiency","Final score","Expected handle time (mins)"],
        "Value": [
            f"{example_case_local['priority']} (→ {example_case_local['priority_score']:.2f})",
            f"{example_case_local['sla_urgency']:.2f}",
            f"{example_case_local['complexity']:.2f}",
            f"{example_agent_local['skill_level']}/5 (→ {disp['skill_norm']:.2f})",
            f"{example_agent_local['current_backlog']}/{example_agent_local['capacity_daily']} (→ {disp['backlog_norm']:.2f})",
            f"{disp['efficiency_norm']:.2f}",
            f"{final_score_display}",
            f"{example_agent_local['avg_handle_time_mins']}"
        ]
    })
    st.table(df_at_glance)

    # downloadable concise markdown
    example_md = f"""# S.V.A.E. Example — Case {example_case_local['case_id']}

Case:
- refer_code: {example_case_local['refer_code']}
- priority: {example_case_local['priority']} (→ {example_case_local['priority_score']:.2f})
- sla_urgency: {example_case_local['sla_urgency']:.2f}
- complexity: {example_case_local['complexity']:.2f}

Agent:
- id: {example_agent_local['agent_id']}
- aht: {example_agent_local['avg_handle_time_mins']} mins
- backlog: {example_agent_local['current_backlog']}/{example_agent_local['capacity_daily']} (→ {disp['backlog_norm']:.2f})
- skill: {example_agent_local['skill_level']}/5 (→ {disp['skill_norm']:.2f})
- efficiency: {disp['efficiency_norm']:.2f}
- skill_bonus: {disp['skill_bonus']:.2f}

Final suitability score: {disp['final_score']:.2f}
"""
    st.download_button("Download Example ", example_md, f"svae_example_{example_case_local['case_id']}.md", "text/markdown")

    # Detailed arithmetic & metric explanations
    with st.expander("Show full arithmetic & metric explanations (audit view)"):
        st.markdown("#### Full numeric breakdown ")
        st.write(f"- SLA (normalized): {disp['sla_norm']:.2f}")
        st.write(f"- Priority (normalized): {disp['priority_norm']:.2f}")
        st.write(f"- Efficiency (normalized): {disp['efficiency_norm']:.2f}")
        st.write(f"- Backlog (normalized): {disp['backlog_norm']:.2f}")
        st.write(f"- Skill level (normalized): {disp['skill_norm']:.2f}")
        st.write(f"- Complexity mismatch: {disp['mismatch']:.2f}")
        st.write(f"- Skill bonus: {disp['skill_bonus']:.2f}")
        st.write(f"- Positive contribution (weighted): {disp['positive_contrib']:.4f}")
        st.write(f"- Penalty contribution (weighted): {disp['penalty_contrib']:.4f}")
        st.write(f"- Penalty scale: {disp['penalty_scale']:.4f}")
        st.write(f"- Raw score: {disp['raw_score']:.4f}")
        st.write(f"- Final clipped score: {disp['final_score']:.4f} (→ {final_score_display})")

        st.markdown("---")
        st.markdown("#### Simple metric explanations")
        st.markdown("""
        - **SLA urgency:** linear in days: `1 - remaining_days / total_window_days`. Past-due => 1.00.
        - **Priority:** VIP→1.00, Medium→0.70, Normal→0.40.
        - **Efficiency:** fastest agent → 1.00; others = fastest / agent AHT.
        - **Backlog:** pending / capacity (protects agents).
        - **Skill norm:** skill level / 5.
        - **Complexity mismatch:** `max(0, complexity - skill_norm)` (penalty only when case harder than agent).
        - **Skill bonus:** small positive bump when tags match the refer_code or case_type.
        """)

    st.markdown("---")
    st.caption("Interactive example is local-only and will not modify uploaded data or app state.")
# ==========================================================

# ==========================================================
#TAB 3: Legend
#
# ==========================================================
with tab_legend:
    st.header("Legend — Reference")

    legend_md = (
        "# S.V.A.E. — Legend & Definitions\n\n"
        "**Complexity (0–1)**  \n"
        "Calculated per case from `history.csv` (refer_code stats).  \n"
        "Formula (shown as indented code block):\n\n"
        "    complexity = normalize(0.5*norm(avg_tat_days) + 0.3*norm(sla_breach_pct) + 0.2*exception_rate)\n\n"
        "- `norm(avg_tat_days)` = avg_tat_days / max_tat (clipped)\n"
        "- If refer_code not found, defaults to 0.3.\n\n"
        "**SLA urgency (0-1)**  \n"
        "`sla_urgency = 1 - (remaining_days / total_window_days)` clipped to [0,1]. Past-due => 1.00.\n\n"
        "**Priority score**  \n"
        "Mapped: priority 1 → 1.0, 2 → 0.7, 3 → 0.4.\n\n"
        "**Agent efficiency**  \n"
        "`efficiency = min_aht / agent_aht` (fastest agent => 1.0). Clipped to [0,1].\n\n"
        "**Suitability Score (higher is better)**  \n"
        "    Score = normalized_positive - scaled_penalty + skill_bonus\n\n"
        "- `Agent_backlog_norm = current_backlog / capacity_daily`  (0..1)\n"
        "- `Complexity_mismatch = max(0, complexity - agent_skill_norm)` (0..1)\n"
        "- `Skill_match_bonus` small bonus when agent skill_tags match case refer or type (0..1).\n\n"
        "**Expected handle time (mins)**  \n"
        "Taken from agent's `avg_handle_time_mins` in agents.csv (used for capacity tracking).\n"
    )

    st.markdown(legend_md)
    st.download_button(
        label="Download Legend (Markdown)",
        data=legend_md,
        file_name="svae_legend.md",
        mime="text/markdown"
    )

# ==========================================================
#Allocation Run / UI
# ==========================================================
st.markdown("---")
st.header("Allocation Run")

if volumes_df is None or volumes_df.empty:
    st.warning("Please upload **volumes.csv** first.")
elif agents_df is None or agents_df.empty:
    st.warning("Please upload **agents.csv** next.")
else:
    if st.button("Run greedy allocation"):
        st.session_state["alloc_results"] = run_greedy_allocation(volumes_df.copy(), agents_df.copy(), history_df)
        st.success("Allocation complete!")

if st.session_state.get("alloc_results") is not None:
    res = st.session_state["alloc_results"]
    st.subheader("Allocation Results")
    st.dataframe(res)
    st.metric("Assigned", int(res["assigned_agent"].notnull().sum()))
    st.metric("Unassigned", int(res["assigned_agent"].isnull().sum()))
    csv_buf = io.StringIO()
    res.to_csv(csv_buf, index=False)
    st.download_button("Download allocation.csv", csv_buf.getvalue(), "allocation.csv", "text/csv")
else:
    st.info("No allocation results yet. Upload CSVs and click Run.")

st.caption("S.V.A.E. prototype — manual upload only. ")
