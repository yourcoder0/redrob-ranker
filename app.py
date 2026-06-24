"""
Redrob Hackathon — Interactive Sandbox
Demonstrates the ranker on a small candidate sample.
Deploy on HuggingFace Spaces / Streamlit Cloud.
"""

import streamlit as st
import json
import csv
import io
import sys
import os

# Import ranker (rank.py must be in same directory)
sys.path.insert(0, os.path.dirname(__file__))
from rank import score_candidate, detect_honeypot, hard_disqualifier

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Redrob Intelligent Candidate Ranker")
st.markdown(
    "**Senior AI Engineer (Founding Team) @ Redrob AI** — "
    "Upload a JSONL file of candidates (or paste JSON) to see ranked results."
)

# ---------------------------------------------------------------------------
# Sidebar — scoring explainer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Scoring Architecture")
    st.markdown("""
**Component Scores (max 100)**
| Component | Max |
|-----------|-----|
| Core Skills Match | 35 |
| Career Trajectory | 25 |
| Experience Years | 10 |
| Location/Availability | 10 |
| Education | 5 |
| Bonus Skills | 5 |

**× Behavioral Modifier (0.3–1.0)**
- Last active date
- Open-to-work flag
- Recruiter response rate
- Notice period
- Interview completion rate
- GitHub activity

**Trap Avoidance**
- Honeypot detection (impossible profiles)
- Hard disqualifiers (consulting-only, non-technical, CV/robotics-only)
- Keyword stuffer detection (trust multiplier on skills)
    """)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.subheader("Input Candidates")

tab1, tab2 = st.tabs(["📁 Upload JSONL File", "📝 Paste JSON"])

candidates = []

with tab1:
    uploaded = st.file_uploader("Upload a .jsonl file (one candidate per line)", type=["jsonl", "json"])
    if uploaded:
        content = uploaded.read().decode("utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    # Handle both JSONL (one obj per line) and JSON array
                    obj = json.loads(line)
                    if isinstance(obj, list):
                        candidates.extend(obj)
                    else:
                        candidates.append(obj)
                except json.JSONDecodeError:
                    pass
        st.success(f"Loaded {len(candidates)} candidates from file.")

with tab2:
    sample_json = json.dumps([
        {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Alex K.",
                "headline": "Senior NLP/Search Engineer",
                "summary": "5 years building production retrieval systems with embeddings and vector databases.",
                "location": "Bangalore, Karnataka",
                "country": "India",
                "years_of_experience": 5,
                "current_title": "ML Engineer",
                "current_company": "Startup Co",
                "current_company_size": "51-200",
                "current_industry": "Technology"
            },
            "career_history": [
                {
                    "company": "Startup Co",
                    "title": "ML Engineer",
                    "start_date": "2022-01-01",
                    "end_date": None,
                    "duration_months": 29,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "51-200",
                    "description": "Built and deployed semantic search pipeline using sentence-transformers and FAISS. Implemented BM25 + dense retrieval hybrid. Improved NDCG@10 by 18%."
                }
            ],
            "education": [
                {
                    "institution": "IIT Bombay",
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science",
                    "start_year": 2016,
                    "end_year": 2020,
                    "grade": "8.5",
                    "tier": "tier_1"
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 45, "duration_months": 60},
                {"name": "Sentence Transformers", "proficiency": "advanced", "endorsements": 12, "duration_months": 30},
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 8, "duration_months": 24},
                {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
                {"name": "BM25", "proficiency": "intermediate", "endorsements": 5, "duration_months": 18},
                {"name": "RAG", "proficiency": "intermediate", "endorsements": 4, "duration_months": 12}
            ],
            "certifications": [],
            "languages": [{"language": "English", "proficiency": "professional"}],
            "redrob_signals": {
                "profile_completeness_score": 92,
                "signup_date": "2023-06-01",
                "last_active_date": "2025-05-20",
                "open_to_work_flag": True,
                "profile_views_received_30d": 28,
                "applications_submitted_30d": 3,
                "recruiter_response_rate": 0.82,
                "avg_response_time_hours": 4,
                "skill_assessment_scores": {"Python": 88, "NLP": 79},
                "connection_count": 410,
                "endorsements_received": 84,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 28, "max": 38},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 72,
                "search_appearance_30d": 55,
                "saved_by_recruiters_30d": 9,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.75,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True
            }
        }
    ], indent=2)

    pasted = st.text_area("Paste candidate JSON (array or JSONL)", value=sample_json, height=200)
    if st.button("Load from paste"):
        try:
            parsed = json.loads(pasted)
            if isinstance(parsed, list):
                candidates = parsed
            else:
                candidates = [parsed]
            st.success(f"Loaded {len(candidates)} candidates.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

# ---------------------------------------------------------------------------
# Rank button
# ---------------------------------------------------------------------------
if candidates:
    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        n_show = st.number_input("Show top N", min_value=1, max_value=min(100, len(candidates)),
                                  value=min(10, len(candidates)))
    with col2:
        run = st.button("🚀 Rank Candidates", type="primary")

    if run:
        with st.spinner(f"Scoring {len(candidates)} candidates..."):
            scored = []
            for c in candidates:
                cid = c.get("candidate_id", "UNKNOWN")
                is_hp = detect_honeypot(c)
                is_dq, dq_reason = hard_disqualifier(c)
                score, reasoning = score_candidate(c)
                scored.append({
                    "candidate_id": cid,
                    "score": round(score / 100.0, 4),
                    "raw_score": round(score, 2),
                    "reasoning": reasoning,
                    "is_honeypot": is_hp,
                    "is_disqualified": is_dq,
                    "dq_reason": dq_reason,
                    "name": c.get("profile", {}).get("anonymized_name", ""),
                    "title": c.get("profile", {}).get("current_title", ""),
                    "company": c.get("profile", {}).get("current_company", ""),
                    "yoe": c.get("profile", {}).get("years_of_experience", 0),
                    "location": c.get("profile", {}).get("location", ""),
                })

            scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))

        st.subheader(f"Top {n_show} Candidates")

        # Stats
        hp_count = sum(1 for s in scored if s["is_honeypot"])
        dq_count = sum(1 for s in scored[:n_show] if s["is_disqualified"])
        st.info(f"Total scored: {len(scored)} | Honeypots detected: {hp_count} | "
                f"Disqualified in top {n_show}: {dq_count}")

        for rank, s in enumerate(scored[:n_show], 1):
            badge = ""
            if s["is_honeypot"]:
                badge = " 🚨 HONEYPOT"
            elif s["is_disqualified"]:
                badge = " ❌ DISQUALIFIED"

            score_pct = int(s["score"] * 100)
            bar = "█" * (score_pct // 5) + "░" * (20 - score_pct // 5)

            with st.expander(
                f"#{rank} — {s['name']} | {s['title']} @ {s['company']} "
                f"| Score: {s['score']:.4f}{badge}"
            ):
                cols = st.columns([2, 1, 1, 1])
                cols[0].metric("Candidate ID", s["candidate_id"])
                cols[1].metric("Score", f"{s['score']:.4f}")
                cols[2].metric("Experience", f"{s['yoe']:.0f} yrs")
                cols[3].metric("Location", s["location"] or "—")

                st.markdown(f"**Score bar:** `{bar}` {score_pct}%")
                st.markdown(f"**Reasoning:** {s['reasoning']}")
                if s["is_disqualified"]:
                    st.error(f"Disqualification reason: {s['dq_reason']}")

        # CSV Download
        st.divider()
        st.subheader("Download Submission CSV")
        top = scored[:min(100, len(scored))]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, s in enumerate(top, 1):
            writer.writerow([s["candidate_id"], rank, s["score"], s["reasoning"]])
        st.download_button(
            "⬇️ Download submission.csv",
            data=buf.getvalue(),
            file_name="submission.csv",
            mime="text/csv",
        )
