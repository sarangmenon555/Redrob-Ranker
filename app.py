import json
import time
import os
import streamlit as st
import pandas as pd
from src.ranker import (
    compute_score, score_skills, score_career, score_behavioral,
    score_location, ml_career_evidence, is_honeypot,
    JD_MUST_SKILLS, JD_GOOD_SKILLS, JD_NEG_SKILLS,
    HARD_NON_ENGINEERING, ENGINEERING_TITLE_KW,
)

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #0f1d2e;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .score-high { color: #00BFA5; font-weight: 700; }
    .score-mid  { color: #FFB300; font-weight: 700; }
    .score-low  { color: #EF5350; font-weight: 700; }
    .tag {
        display: inline-block;
        background: #0f2747;
        border: 1px solid #1e3a5f;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        margin: 2px;
        color: #90caf9;
    }
    .tag-match {
        background: #0d3327;
        border-color: #00BFA5;
        color: #00BFA5;
    }
    .tag-neg {
        background: #3e1010;
        border-color: #ef5350;
        color: #ef5350;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "candidates.jsonl")


@st.cache_data(show_spinner=False)
def load_candidates():
    candidates = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return candidates


@st.cache_data(show_spinner=False)
def run_full_ranking():
    candidates = load_candidates()
    results = []
    for c in candidates:
        score, reasoning = compute_score(c)
        p = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        sk, matched = score_skills(c)
        ca = score_career(c)
        be = score_behavioral(sig)
        lo = score_location(c, sig)
        results.append({
            "candidate_id": c["candidate_id"],
            "score": score,
            "reasoning": reasoning,
            "title": p.get("current_title", ""),
            "company": p.get("current_company", ""),
            "yoe": p.get("years_of_experience", 0),
            "location": p.get("location", ""),
            "country": p.get("country", ""),
            "skill_score": round(sk, 3),
            "career_score": round(ca, 3),
            "behavioral_score": round(be, 3),
            "location_score": round(lo, 3),
            "matched_skills": matched,
            "open_to_work": sig.get("open_to_work_flag", False),
            "response_rate": sig.get("recruiter_response_rate", None),
            "notice_days": sig.get("notice_period_days", None),
            "last_active": sig.get("last_active_date", ""),
            "honeypot": is_honeypot(c),
        })
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results, candidates


def score_color_class(score):
    if score >= 0.75:
        return "score-high"
    elif score >= 0.50:
        return "score-mid"
    return "score-low"


def render_score_bar(label, value, color):
    pct = int(value * 100)
    st.markdown(f"""
    <div style="margin-bottom:6px;">
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px;">
            <span style="color:#94a3b8;">{label}</span>
            <span style="color:{color}; font-weight:600;">{pct}%</span>
        </div>
        <div style="background:#1e3a5f; border-radius:4px; height:6px;">
            <div style="background:{color}; width:{pct}%; height:6px; border-radius:4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


st.sidebar.title("Redrob Ranker")
st.sidebar.caption("Intelligent candidate discovery for the Senior AI Engineer role")
st.sidebar.divider()

page = st.sidebar.radio("View", ["Full Rankings", "Candidate Detail", "Score Explorer", "Stats"], label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown("**Scoring Weights**")
st.sidebar.markdown("""
- Skill Matching: **40%**
- Career Quality: **30%**
- Behavioral Signals: **20%**
- Location Fit: **10%**
""")

with st.spinner("Loading and scoring 100,000 candidates..."):
    results, candidates = run_full_ranking()

cand_map = {c["candidate_id"]: c for c in candidates}

if page == "Full Rankings":
    st.title("Candidate Rankings")
    st.caption(f"Scored {len(results):,} candidates | {sum(1 for r in results if r['honeypot']):,} honeypots filtered")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Candidates", f"{len(results):,}")
    col2.metric("Honeypots Filtered", f"{sum(1 for r in results if r['honeypot']):,}")
    col3.metric("India-based (Top 100)", f"{sum(1 for r in results[:100] if r['country'] == 'India')}")
    col4.metric("Top Score", f"{results[0]['score']:.4f}")

    st.divider()

    with st.expander("Filter Options"):
        fc1, fc2, fc3 = st.columns(3)
        min_score = fc1.slider("Minimum Score", 0.0, 1.0, 0.0, 0.01)
        min_yoe = fc2.slider("Minimum Years of Experience", 0, 20, 0)
        country_filter = fc3.selectbox("Country", ["All"] + sorted(set(r["country"] for r in results[:500] if r["country"])))
        show_n = st.slider("Show Top N", 10, 200, 100, 10)

    filtered = [
        r for r in results
        if r["score"] >= min_score
        and r["yoe"] >= min_yoe
        and (country_filter == "All" or r["country"] == country_filter)
        and not r["honeypot"]
    ][:show_n]

    df = pd.DataFrame([{
        "Rank": r["rank"],
        "ID": r["candidate_id"],
        "Title": r["title"],
        "Company": r["company"],
        "YOE": r["yoe"],
        "Country": r["country"],
        "Score": r["score"],
        "Skill": r["skill_score"],
        "Career": r["career_score"],
        "Behavioral": r["behavioral_score"],
        "Open to Work": "Yes" if r["open_to_work"] else "No",
    } for r in filtered])

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.4f"),
            "Skill": st.column_config.ProgressColumn("Skill", min_value=0, max_value=1, format="%.3f"),
            "Career": st.column_config.ProgressColumn("Career", min_value=0, max_value=1, format="%.3f"),
            "Behavioral": st.column_config.ProgressColumn("Behavioral", min_value=0, max_value=1, format="%.3f"),
        }
    )

    st.download_button(
        "Download Top 100 CSV",
        data="\n".join(
            ["candidate_id,rank,score,reasoning"] +
            [f"{r['candidate_id']},{r['rank']},{r['score']:.6f},{r['reasoning']}" for r in results[:100]]
        ),
        file_name="team_devnira.csv",
        mime="text/csv",
    )


elif page == "Candidate Detail":
    st.title("Candidate Detail")

    search_input = st.text_input("Enter Candidate ID (e.g. CAND_0046064)")

    rank_lookup = {r["candidate_id"]: r for r in results}

    if search_input and search_input in cand_map:
        c = cand_map[search_input]
        r = rank_lookup.get(search_input, {})
        p = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        career = c.get("career_history", [])
        skills = c.get("skills", [])

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader(p.get("current_title", "Unknown Title"))
            st.caption(f"{p.get('current_company', '')} | {p.get('location', '')} | {p.get('years_of_experience', 0)} years")
            st.write(p.get("summary", ""))

        with col_right:
            score = r.get("score", 0)
            cls = score_color_class(score)
            st.markdown(f"<div class='metric-card'><div style='font-size:13px; color:#94a3b8;'>Overall Score</div><div class='{cls}' style='font-size:2.5rem;'>{score:.4f}</div><div style='font-size:12px; color:#64748b;'>Rank #{r.get('rank', '?')} of 100,000</div></div>", unsafe_allow_html=True)

        st.divider()

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Skill Match", f"{r.get('skill_score', 0):.1%}")
        sc2.metric("Career Quality", f"{r.get('career_score', 0):.1%}")
        sc3.metric("Behavioral", f"{r.get('behavioral_score', 0):.1%}")
        sc4.metric("Location", f"{r.get('location_score', 0):.1%}")

        render_score_bar("Skill Match (40%)", r.get("skill_score", 0), "#00BFA5")
        render_score_bar("Career Quality (30%)", r.get("career_score", 0), "#4FC3F7")
        render_score_bar("Behavioral Signals (20%)", r.get("behavioral_score", 0), "#FFB300")
        render_score_bar("Location Fit (10%)", r.get("location_score", 0), "#CE93D8")

        st.divider()

        tab1, tab2, tab3 = st.tabs(["Skills", "Career History", "Behavioral Signals"])

        with tab1:
            matched = r.get("matched_skills", [])
            listed_matches = [s for s in matched if not s.startswith("~")]
            text_matches = [s.replace("~", "") for s in matched if s.startswith("~")]

            if listed_matches:
                st.markdown("**Listed skills matched to JD:**")
                st.markdown(" ".join(f"<span class='tag tag-match'>{s}</span>" for s in listed_matches), unsafe_allow_html=True)

            if text_matches:
                st.markdown("**Skills found in career text:**")
                st.markdown(" ".join(f"<span class='tag'>{s}</span>" for s in text_matches[:10]), unsafe_allow_html=True)

            st.markdown("**All listed skills:**")
            all_skill_names = [s["name"] for s in skills]
            neg_skill_names = [k for k in JD_NEG_SKILLS if any(k in sn.lower() for sn in all_skill_names)]
            tags = []
            for s in skills:
                nm = s["name"]
                nm_l = nm.lower()
                is_match = any(nm_l == m for m in listed_matches)
                is_neg = any(neg in nm_l for neg in JD_NEG_SKILLS)
                css = "tag-match" if is_match else ("tag-neg" if is_neg else "")
                tags.append(f"<span class='tag {css}'>{nm} ({s.get('proficiency','?')}, {s.get('duration_months',0)}mo)</span>")
            st.markdown(" ".join(tags), unsafe_allow_html=True)

        with tab2:
            for job in career:
                with st.container():
                    st.markdown(f"**{job.get('title', '')}** at {job.get('company', '')} — {job.get('duration_months', 0)} months")
                    st.caption(f"{job.get('start_date', '')[:7]} to {job.get('end_date', 'present')[:7] if job.get('end_date') else 'present'}")
                    st.write(job.get("description", ""))
                    st.divider()

        with tab3:
            b1, b2, b3 = st.columns(3)
            b1.metric("Open to Work", "Yes" if sig.get("open_to_work_flag") else "No")
            b2.metric("Recruiter Response Rate", f"{sig.get('recruiter_response_rate', 0):.0%}" if sig.get("recruiter_response_rate") is not None else "N/A")
            b3.metric("Notice Period", f"{sig.get('notice_period_days', '?')} days")

            b4, b5, b6 = st.columns(3)
            b4.metric("Last Active", sig.get("last_active_date", "")[:10])
            b5.metric("GitHub Activity", f"{sig.get('github_activity_score', 0):.0f}/100")
            b6.metric("Interview Completion", f"{sig.get('interview_completion_rate', 0):.0%}" if sig.get("interview_completion_rate") is not None else "N/A")

            sal = sig.get("expected_salary_range_inr_lpa", {})
            if isinstance(sal, dict) and sal:
                st.metric("Expected Salary (LPA)", f"{sal.get('min','?')} - {sal.get('max','?')}")

    elif search_input:
        st.warning("Candidate not found. Check the ID.")
    else:
        st.info("Enter a candidate ID above to see their full scoring breakdown.")
        st.markdown("**Top 10 candidates:**")
        for r in results[:10]:
            if st.button(f"#{r['rank']} {r['candidate_id']} — {r['title']} ({r['score']:.4f})", key=r["candidate_id"]):
                st.session_state["selected"] = r["candidate_id"]


elif page == "Score Explorer":
    st.title("Score Explorer")
    st.caption("Understand how the scoring model works")

    st.subheader("Must-Have Skills")
    skill_df = pd.DataFrame([
        {"Skill": k, "Weight": v, "Category": "Must-Have"}
        for k, v in sorted(JD_MUST_SKILLS.items(), key=lambda x: -x[1])[:20]
    ] + [
        {"Skill": k, "Weight": v, "Category": "Nice-to-Have"}
        for k, v in sorted(JD_GOOD_SKILLS.items(), key=lambda x: -x[1])
    ] + [
        {"Skill": k, "Weight": abs(v), "Category": "Negative Signal"}
        for k, v in sorted(JD_NEG_SKILLS.items(), key=lambda x: x[1])[:10]
    ])

    st.dataframe(
        skill_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=15, format="%.0f"),
        }
    )

    st.divider()
    st.subheader("Career Scoring Logic")
    st.markdown("""
    **Experience band scoring:**
    - 5 to 9 years: +25 points (ideal)
    - 4 to 5 years or 9 to 11 years: +16 to +18 points
    - 3 to 4 years or 11 to 14 years: +8 points
    - Over 14 years: +3 points

    **Title penalties:**
    - Non-engineering titles (Customer Support, HR, Sales, Mechanical Engineer, Civil Engineer): -45 points
    - Ambiguous titles: -12 points

    **Computer vision title penalty:** -20 additional points

    **Services company penalty:** -18 points for TCS/Infosys/Wipro-only careers

    **Tenure stability:**
    - Average tenure under 10 months: -12 points
    - Average tenure under 18 months: -5 points
    - Average tenure over 30 months: +6 points
    """)

    st.divider()
    st.subheader("Honeypot Detection Rules")
    st.markdown("""
    A candidate is flagged as a honeypot (score = 0) if any of the following are true:

    1. Any job has an end date before its start date
    2. Any job has a start year before 1970 or after next year
    3. Five or more skills marked advanced or expert with 0 months of experience
    4. Total skill-months exceeds 9 times career years (mathematically impossible)
    """)


elif page == "Stats":
    st.title("Dataset Stats")

    top100 = [r for r in results[:100]]
    all_results = results

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution (top 100)")
        score_df = pd.DataFrame({"Score": [r["score"] for r in top100]})
        st.bar_chart(score_df["Score"].value_counts(bins=20).sort_index())

    with col2:
        st.subheader("Country Distribution (top 100)")
        from collections import Counter
        country_counts = Counter(r["country"] for r in top100)
        st.bar_chart(pd.DataFrame.from_dict(country_counts, orient="index", columns=["Count"]))

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("YOE Distribution (top 100)")
        yoe_df = pd.DataFrame({"YOE": [r["yoe"] for r in top100]})
        st.bar_chart(yoe_df["YOE"].value_counts(bins=15).sort_index())

    with col4:
        st.subheader("Honeypot Rate")
        hp_total = sum(1 for r in all_results if r["honeypot"])
        clean_total = len(all_results) - hp_total
        hp_df = pd.DataFrame({"Count": [clean_total, hp_total]}, index=["Clean", "Honeypot"])
        st.bar_chart(hp_df)

    st.divider()
    st.subheader("Open to Work (top 100)")
    otw_count = sum(1 for r in top100 if r["open_to_work"])
    st.metric("Actively open to work", f"{otw_count} / 100")

    st.subheader("Response Rate Distribution (top 100)")
    rr_vals = [r["response_rate"] for r in top100 if r["response_rate"] is not None]
    if rr_vals:
        rr_df = pd.DataFrame({"Response Rate": rr_vals})
        st.bar_chart(rr_df["Response Rate"].value_counts(bins=10).sort_index())
