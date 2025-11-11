# svae_app.py
# S.V.A.E. — Smart Volume Allocation Engine (Prototype)
# Manual-upload-only Streamlit MVP with compact weights + static Legend tab + About tab (executive-friendly)

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

volumes_df = read_csv(uploaded_vol)
agents_df = read_csv(uploaded_agents)
history_df = read_csv(uploaded_history)

# ---------------- Tabs: Data / About / Legend ----------------
tab_data, tab_about, tab_legend = st.tabs(["Data & Preview", "About", "Legend"])

# ==========================================================
# 🧩 TAB 1: Data & Preview + Compact Weight Inputs
# ==========================================================
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
    st.caption("Adjust weights before running allocation — keep them short and business-aligned")

    defaults = {
        "w_alpha": 0.35, "w_beta": 0.25, "w_gamma": 0.15,
        "w_delta": 0.15, "w_epsilon": 0.7, "w_zeta": 0.1
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    cols = st.columns([1, 1, 1, 1, 1, 1])
    st.session_state["w_alpha"] = cols[0].number_input("α (SLA urgency)", value=float(st.session_state["w_alpha"]), step=0.05, min_value=0.0, max_value=2.0)
    st.session_state["w_beta"] = cols[1].number_input("β (Priority)", value=float(st.session_state["w_beta"]), step=0.05, min_value=0.0, max_value=2.0)
    st.session_state["w_gamma"] = cols[2].number_input("γ (Agent efficiency)", value=float(st.session_state["w_gamma"]), step=0.05, min_value=0.0, max_value=2.0)
    st.session_state["w_delta"] = cols[3].number_input("δ (Backlog penalty)", value=float(st.session_state["w_delta"]), step=0.05, min_value=0.0, max_value=2.0)
    st.session_state["w_epsilon"] = cols[4].number_input("ε (Complexity mismatch)", value=float(st.session_state["w_epsilon"]), step=0.05, min_value=0.0, max_value=5.0)
    st.session_state["w_zeta"] = cols[5].number_input("ζ (Skill bonus)", value=float(st.session_state["w_zeta"]), step=0.05, min_value=0.0, max_value=2.0)

# ==========================================================
# 📘 TAB 2: About (Consulting-style explainer + worked example)
# ==========================================================
with tab_about:
   
   

    st.markdown("### What we measure")
    st.markdown(
        "- **SLA Urgency (0–1):** how close a case is to breaching its SLA.\n"
        "- **Priority (0–1):** business importance (VIP → 1.0, medium → 0.7, normal → 0.4).\n"
        "- **Complexity (0–1):** expected difficulty from historical TAT, breach %, exception rate.\n"
        "- **Agent Efficiency:** speed indicator derived from AHT (normalized as 1 / AHT).\n"
        "- **Backlog Penalty:** current work / daily capacity (keeps load balanced).\n"
        "- **Skill Match / Level:** agent competence and small bonus if tags match."
    )

    st.markdown("### The simple math")
    st.markdown(
        "For each case–agent pair the engine computes a single Suitability Score:\n\n"
        "Score = α × SLA + β × Priority + γ × Efficiency − δ × Backlog − ε × ComplexityMismatch + ζ × SkillBonus\n\n"
        "Every weight (α..ζ) is tunable by business to reflect priorities (e.g., emphasize SLA over load balancing or vice versa)."
    )

    st.markdown("### Worked example")
    

    st.markdown("**Scenario (single urgent VIP case):**")
    st.markdown(
        "- SLA Urgency = 0.8 (close to breach)\n"
        "- Priority = 1.0 (VIP)\n"
        "- Complexity = 0.6 (moderately complex)"
    )

    st.markdown("**Agent A (Fast & Skilled):**")
    st.markdown(
        "- AHT = 20 mins → Efficiency = 1 / 20 = 0.05\n"
        "- Backlog = 30% → BacklogPenalty = 0.3\n"
        "- SkillLevel = 0.8 → ComplexityMismatch = max(0, 0.6 − 0.8) = 0\n"
        "- SkillBonus = 0.1 (tags match)"
    )

    st.markdown("**Weights used (default):** α=0.35, β=0.25, γ=0.15, δ=0.15, ε=0.7, ζ=0.1")

    st.markdown("**Calculation**")
    st.markdown(
        "α × SLA = 0.35 × 0.8 = 0.28\n"
        "β × Priority = 0.25 × 1.0 = 0.25\n"
        "γ × Efficiency = 0.15 × 0.05 = 0.0075\n"
        "− δ × Backlog = −0.15 × 0.3 = −0.045\n"
        "− ε × Mismatch = −0.7 × 0 = 0\n"
        "+ ζ × SkillBonus = 0.1 × 0.1 = 0.01\n\n"
        "Final Score = 0.28 + 0.25 + 0.0075 − 0.045 + 0 + 0.01 = 0.5025"
    )

    
    st.success(
        "This agent scores ~0.50 because they are fast, skilled, and available enough — the score is a transparent sum of urgency, priority, speed, workload, and skill match."
    )

    st.markdown("### Quick rebuttals")
    st.markdown(
        "- Q: Why not just give the case to the fastest agent?  \n"
        "A: Because the fastest agent may be overloaded (backlog) or not skilled for complex cases. The score balances multiple factors.\n\n"
        "- Q: Why use 1/AHT instead of RPH?  \n"
        "A: 1/AHT is a normalized efficiency index that keeps the speed term in the same numeric range as other terms, so no single metric dominates unless you explicitly choose that via weights.\n\n"
        "- Q: What if an agent is near capacity?  \n"
        "A: The backlog penalty reduces their score so assignments stay fair and SLA risk is minimized across the team."
    )



# ==========================================================
# 📘 TAB 3: Legend (Static Info) — no fenced backticks inside string
# ==========================================================
with tab_legend:
    st.header("Legend — Reference")

    # Using indented code-style blocks (4-space) instead of triple-backtick fences
    legend_md = (
        "# S.V.A.E. — Legend & Definitions\n\n"
        "**Complexity (0–1)**  \n"
        "Calculated per case from `history.csv` (refer_code stats).  \n"
        "Formula (shown as indented code block):\n\n"
        "    complexity = normalize(0.5*norm(avg_tat_days) + 0.3*norm(sla_breach_pct) + 0.2*exception_rate)\n\n"
        "- `norm(avg_tat_days)` = avg_tat_days / max_tat (clipped)\n"
        "- If refer_code not found, defaults to 0.3.\n\n"
        "**SLA urgency (0–1)**  \n"
        "`sla_urgency = 1 - (remaining_time / total_window)` (clipped). Higher = closer to SLA.\n\n"
        "**Priority score**  \n"
        "Mapped: priority 1 → 1.0, 2 → 0.7, 3 → 0.4.\n\n"
        "**Agent efficiency**  \n"
        "`efficiency = 1 / avg_handle_time_mins` (faster = higher).\n\n"
        "**Suitability Score (higher is better)**  \n"
        "```\n"
        "Score = α*SLA_urgency + β*Priority + γ*Agent_efficiency\n"
        "        - δ*Agent_backlog_norm - ε*Complexity_mismatch + ζ*Skill_match_bonus\n"
        "```\n"
        "- `Agent_backlog_norm = current_backlog / capacity_daily`\n"
        "- `Complexity_mismatch = max(0, complexity - agent_skill_norm)`\n"
        "- `Skill_match_bonus` small bonus when agent skill_tags match case refer or type.\n\n"
        "**Expected handle time (mins)**  \n"
        "Taken from agent's `avg_handle_time_mins` in agents.csv.\n"
    )

    st.markdown(legend_md)
    st.download_button(
        label="Download Legend (MD)",
        data=legend_md,
        file_name="svae_legend.md",
        mime="text/markdown"
    )

# ==========================================================
# ⚙️ Core Functions (unchanged)
# ==========================================================
def compute_complexity(row, history_df, max_tat=10.0):
    rc = row.get("refer_code", None)
    if pd.isna(rc) or rc is None:
        return 0.3
    if history_df is None or history_df.empty:
        return 0.3
    rec = history_df[history_df["refer_code"] == rc]
    if not rec.empty:
        r = rec.iloc[0]
        try:
            norm_tat = min(float(r["avg_tat_days"]) / max_tat, 1.0)
            norm_breach = min(float(r["sla_breach_pct"]) / 100.0, 1.0)
            norm_exc = min(float(r["exception_rate"]), 1.0)
        except Exception:
            return 0.3
        comp = 0.5 * norm_tat + 0.3 * norm_breach + 0.2 * norm_exc
        return float(min(max(comp, 0.0), 1.0))
    return 0.3

def sla_urgency_score(row):
    try:
        due = pd.to_datetime(row["sla_due_date"])
        now = pd.Timestamp.now()
        total = (due - pd.to_datetime(row["received_date"])).days
        remain = (due - now).days
        if total <= 0: return 1.0
        return float(max(0.0, 1.0 - (remain / total)))
    except Exception:
        return 0.5

def priority_score(p):
    mapping = {1: 1.0, 2: 0.7, 3: 0.4}
    try: return mapping.get(int(p), 0.5)
    except Exception: return 0.5

def agent_efficiency(a_row):
    try: return 1.0 / float(a_row.get("avg_handle_time_mins", 30))
    except Exception: return 0.03

def skill_match_bonus(case_row, agent_row):
    tags = [t.strip() for t in str(agent_row.get("skill_tags", "")).split(",") if t.strip()]
    refer = str(case_row.get("refer_code") or "")
    ctype = str(case_row.get("case_type") or "")
    return 0.15 if (refer in tags or ctype in tags) else 0.0

def suitability_score(case_row, agent_row, history_df):
    comp = float(case_row.get("complexity", 0.3))
    sla = float(case_row.get("sla_urgency", 0.5))
    pri = float(case_row.get("priority_score", 0.5))
    eff = agent_efficiency(agent_row)
    backlog = float(agent_row.get("current_backlog", 0)) / max(1.0, float(agent_row.get("capacity_daily", 1)))
    skill_bonus = skill_match_bonus(case_row, agent_row)
    mismatch = max(0.0, comp - (float(agent_row.get("skill_level", 3)) / 5.0))
    w = st.session_state
    return (w["w_alpha"]*sla + w["w_beta"]*pri + w["w_gamma"]*eff -
            w["w_delta"]*backlog - w["w_epsilon"]*mismatch + w["w_zeta"]*skill_bonus)

def run_greedy_allocation(vols, agents, history):
    # ensure columns exist safely
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

    results = []
    for _, case in vols.iterrows():
        best_agent = None
        best_score = -1
        for _, agent in agents.iterrows():
            try:
                if int(agent["assigned_count"]) >= int(agent["capacity_daily"]):
                    continue
            except Exception:
                pass
            score = suitability_score(case, agent, history)
            if score > best_score:
                best_score = score
                best_agent = agent
        if best_agent is not None:
            results.append({
                "case_id": case.get("case_id"),
                "assigned_agent": best_agent.get("agent_id"),
                "score": round(best_score, 3),
                "complexity": case.get("complexity"),
                "expected_handle_time_mins": best_agent.get("avg_handle_time_mins"),
                "assigned_at": pd.Timestamp.now().isoformat()
            })
            agents.loc[agents["agent_id"] == best_agent["agent_id"], "assigned_count"] += 1
        else:
            results.append({
                "case_id": case.get("case_id"),
                "assigned_agent": None,
                "score": None,
                "complexity": case.get("complexity"),
                "expected_handle_time_mins": None,
                "assigned_at": pd.Timestamp.now().isoformat()
            })
    return pd.DataFrame(results)

# ==========================================================
# 🚀 Allocation Run / UI
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
