#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranker v3
Senior AI Engineer (Founding Team) @ Redrob AI

UPGRADE over v2: Adds LLM reasoning layer using Claude API.

Pipeline:
  Stage 1 — Semantic + Rule scoring on ALL 100K candidates (fast, ~2-3 min)
             → produces preliminary top 500

  Stage 2 — LLM reasoning on top 500 candidates (smart, ~5-10 min)
             → Claude reads each candidate and produces:
               a) A refined fit score (0-10)
               b) A 1-2 sentence reasoning explanation
               c) Flags any disqualifiers or concerns

  Stage 3 — Final hybrid score = 0.6 * stage1_score + 0.4 * llm_score
             → Sort, take top 100, output CSV

Why this works:
  - Stage 1 filters 99.5% of the pool quickly
  - Stage 2 applies genuine intelligence to the remaining candidates
  - LLM catches nuanced fit that rules and embeddings miss
    (e.g. "built retrieval system at Google" >> "listed FAISS as a skill")

Requirements:
  pip install sentence-transformers groq

Usage:
  python rank_v3.py --candidates candidates.jsonl --out submission_v3.csv

  # Skip embeddings (faster, less accurate):
  python rank_v3.py --candidates candidates.jsonl --out submission_v3.csv --no-embeddings

  # Skip LLM (no API key):
  python rank_v3.py --candidates candidates.jsonl --out submission_v3.csv --no-llm

  # Full pipeline:
  python rank_v3.py --candidates candidates.jsonl --out submission_v3.csv
"""

import json
import csv
import math
import re
import sys
import os
import time
import argparse
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# JD Text
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior AI Engineer (Founding Team) at Redrob AI

We are building the next generation of AI-powered recruitment infrastructure.
As a founding AI engineer, you will own the candidate ranking and retrieval system
end to end — from embedding pipelines to evaluation frameworks.

What you will do:
- Design and build production retrieval systems using dense embeddings and vector databases
- Implement hybrid search combining BM25 sparse retrieval with dense vector search
- Build and maintain evaluation pipelines measuring NDCG, MRR, MAP for ranking quality
- Fine-tune embedding models (sentence-transformers, BGE, E5) for domain-specific retrieval
- Integrate vector databases: Pinecone, Qdrant, Weaviate, Milvus, FAISS, Elasticsearch
- Develop reranking models using cross-encoders and learning-to-rank approaches
- Build RAG pipelines for candidate matching
- Write production Python code that ships to real users at scale
- Set up A/B testing and online evaluation for search quality

What we need:
- 5-9 years of hands-on ML/AI engineering experience
- Production experience with semantic search and embedding systems
- Strong Python skills — production code, not just notebooks
- Experience with at least one vector database
- Solid understanding of information retrieval: BM25, NDCG, MRR
- Track record of shipping ML systems to real users
- Startup comfort — fast-moving, small team

Not a fit:
- Pure researchers with no production deployments
- Consulting-only backgrounds with no product experience
- CV/speech specialists with no NLP/IR experience
- Non-coders who call themselves architects
- Candidates outside India (no visa sponsorship)
"""

LLM_SYSTEM_PROMPT = """You are a senior technical recruiter and ML engineer evaluating candidates for a Senior AI Engineer role at an AI startup.

The role requires:
1. Production experience with embeddings and semantic search (MUST HAVE)
2. Vector database experience — Pinecone, Qdrant, FAISS, Elasticsearch etc (MUST HAVE)
3. Strong Python skills with production deployments (MUST HAVE)
4. IR evaluation knowledge — NDCG, MRR, BM25 (MUST HAVE)
5. 5-9 years experience sweet spot
6. India-based (no visa sponsorship)

DISQUALIFIERS (score 0-2):
- Consulting-only career (TCS, Infosys, Wipro, Accenture, Cognizant etc) with no product company
- CV/speech/robotics specialist with zero NLP/IR
- Non-technical role (marketing, HR, sales, design)
- Outside India with no relocation willingness
- Pure researcher, no shipped products

STRONG POSITIVE SIGNALS:
- Built and shipped retrieval/search/ranking systems to real users
- Vector DB + embeddings in production (not just tutorials)
- Startup experience
- Open source contributions
- GitHub activity
- Fast recruiter response, low notice period

You will receive a candidate profile. Respond ONLY with a JSON object like this:
{
  "score": 7.5,
  "reasoning": "One to two sentence explanation of why this candidate fits or doesn't fit.",
  "red_flags": "Any concerns, or empty string if none."
}

Score guide:
0-2: Hard disqualified (wrong domain, consulting only, non-technical)
3-4: Weak fit (some relevant skills but missing core requirements)
5-6: Moderate fit (decent background but gaps in key areas)
7-8: Strong fit (solid relevant experience, most requirements met)
9-10: Exceptional fit (exactly what we need, shipped relevant systems)

Be strict. A candidate who merely lists "embeddings" as a skill scores lower than one who has descriptions of building and deploying embedding systems. Depth > breadth."""


# ---------------------------------------------------------------------------
# All rule-based + embedding code from v2 (copy-paste to keep single file)
# ---------------------------------------------------------------------------

CORE_REQUIRED = {
    "sentence transformers", "sentence-transformers", "embeddings", "embedding",
    "semantic search", "dense retrieval", "bi-encoder", "cross-encoder",
    "bge", "e5", "openai embeddings", "text embeddings", "vector search",
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
    "opensearch", "chroma", "pgvector", "vespa", "annoy", "hybrid search",
    "vector database", "vector db", "vector store",
    "ndcg", "mrr", "map", "mean average precision", "bm25", "learning to rank",
    "reranking", "re-ranking", "ranking system", "retrieval system",
    "information retrieval", "search ranking", "candidate ranking",
    "python",
}

BONUS_SKILLS = {
    "rag", "retrieval augmented generation", "llm", "large language model",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "xgboost", "lightgbm", "learning to rank", "lambdamart",
    "a/b testing", "ab testing", "online evaluation", "offline evaluation",
    "transformers", "hugging face", "huggingface", "bert", "roberta",
    "nlp", "natural language processing", "text classification",
    "recommendation system", "recommender system",
    "mlflow", "weights & biases", "wandb", "experiment tracking",
    "docker", "kubernetes", "fastapi", "flask",
    "distributed systems", "large scale inference",
    "open source", "github",
}

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree",
}

CV_SPEECH_TERMS = {
    "computer vision", "image classification", "object detection", "yolo",
    "speech recognition", "asr", "text-to-speech", "tts", "voice recognition",
    "robotics", "ros", "slam", "autonomous vehicles",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurugram", "gurgaon",
    "hyderabad", "mumbai", "bangalore", "bengaluru",
}

DISQUALIFIER_TITLES = {
    "marketing manager", "hr manager", "sales manager", "content writer",
    "graphic designer", "product manager", "business analyst",
    "data entry", "customer success", "account manager",
    "ux designer", "ui designer", "scrum master",
}


def normalize_text(text: str) -> str:
    return text.lower().strip()


def build_candidate_text(candidate: dict) -> str:
    parts = []
    p = candidate.get("profile", {})
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(f"Current role: {p.get('current_title', '')} at {p.get('current_company', '')}")
    for job in candidate.get("career_history", []):
        parts.append(f"{job.get('title','')} at {job.get('company','')}: {job.get('description','')}")
    skill_names = [s.get("name", "") for s in candidate.get("skills", [])
                   if s.get("proficiency") in ("advanced", "expert")]
    if skill_names:
        parts.append("Key skills: " + ", ".join(skill_names[:20]))
    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))
    return " ".join(p for p in parts if p).strip()


def full_text_lower(candidate: dict) -> str:
    return build_candidate_text(candidate).lower()


def detect_honeypot(candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    yoe = profile.get("years_of_experience", 0)

    for job in career:
        dur = job.get("duration_months", 0)
        start_str = job.get("start_date", "")
        if start_str:
            try:
                if int(start_str[:4]) >= 2022 and yoe >= 8 and dur >= 60:
                    return True
            except ValueError:
                pass

    if sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0) >= 8:
        return True

    total_career = sum(j.get("duration_months", 0) for j in career)
    total_skill = sum(s.get("duration_months", 0) for s in skills)
    if total_career > 0 and total_skill > total_career * 15:
        return True

    if career and yoe > 0:
        career_span = total_career / 12
        if career_span > 0 and abs(yoe - career_span) > 10:
            return True

    return False


def hard_disqualifier(candidate: dict) -> tuple:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    ftext = full_text_lower(candidate)

    current_title = normalize_text(profile.get("current_title", ""))
    if any(dt in current_title for dt in DISQUALIFIER_TITLES):
        ml_career = sum(1 for j in career if any(
            t in normalize_text(j.get("title", ""))
            for t in ["ml", "machine learning", "ai ", "data scientist", "nlp", "engineer"]))
        if ml_career == 0:
            return True, f"Non-technical title: {profile.get('current_title')}"

    if career:
        non_consulting = sum(1 for j in career if not any(
            cc in normalize_text(j.get("company", "")) for cc in CONSULTING_COMPANIES))
        if non_consulting == 0 and len(career) >= 2:
            return True, "Consulting-only career"

    cv_count = sum(1 for t in CV_SPEECH_TERMS if t in ftext)
    nlp_count = sum(1 for t in ["nlp", "information retrieval", "ranking", "search",
                                  "embeddings", "retrieval", "recommendation"] if t in ftext)
    if cv_count >= 4 and nlp_count == 0:
        return True, "CV/speech/robotics only"

    tech_titles = {"engineer", "developer", "scientist", "researcher", "architect",
                   "analyst", "programmer", "technical"}
    if career and not any(any(t in normalize_text(j.get("title", "")) for t in tech_titles) for j in career):
        return True, "No technical roles"

    return False, ""


def skill_trust_score(skill_obj: dict) -> float:
    prof = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}.get(
        skill_obj.get("proficiency", "beginner"), 0.5)
    endo = min(1.0, math.log1p(skill_obj.get("endorsements", 0)) / math.log1p(50))
    dur = skill_obj.get("duration_months", 0)
    dur_w = min(1.0, dur / 36) if dur > 0 else 0.2
    return prof * 0.4 + endo * 0.3 + dur_w * 0.3


def score_core_skills(candidate: dict) -> float:
    skills = candidate.get("skills", [])
    ftext = full_text_lower(candidate)
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})
    skill_map = {normalize_text(s.get("name", "")): s for s in skills}

    core_score = 0.0
    for kw in CORE_REQUIRED:
        if kw in skill_map:
            tm = skill_trust_score(skill_map[kw])
            for akey, aval in assessment_scores.items():
                if kw in akey.lower():
                    tm = min(1.0, tm + aval / 200)
            core_score += tm
        elif kw in ftext:
            core_score += 0.4

    clusters = [
        any(kw in skill_map or kw in ftext for kw in [
            "embeddings", "sentence transformers", "semantic search", "dense retrieval", "bge", "e5"]),
        any(kw in skill_map or kw in ftext for kw in [
            "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
            "opensearch", "chroma", "pgvector", "vector database", "vector db"]),
        any(kw in skill_map or kw in ftext for kw in [
            "ndcg", "mrr", "map", "bm25", "ranking system", "retrieval system",
            "information retrieval", "learning to rank", "reranking"]),
        "python" in skill_map or "python" in ftext,
    ]
    return min(35.0, min(15.0, core_score * 1.5) + sum(clusters) * 5.0)


def score_career(candidate: dict) -> float:
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    if not career:
        return 0.0

    consulting_industries = {"it services", "consulting", "outsourcing", "staffing", "bpo"}
    ai_ml_titles = {
        "ml engineer", "machine learning engineer", "ai engineer", "applied scientist",
        "data scientist", "nlp engineer", "search engineer", "research engineer",
        "ranking engineer", "software engineer", "backend engineer", "platform engineer",
        "senior engineer", "staff engineer", "principal engineer",
    }
    shipped_terms = [
        "deployed", "production", "shipped", "built", "launched", "real users",
        "at scale", "end-to-end", "led", "owned", "retrieval", "ranking",
        "recommendation", "search", "embedding", "vector", "pipeline", "index",
    ]

    score = 0.0
    product_months = total_months = title_score = 0

    for job in career:
        dur = job.get("duration_months", 0)
        total_months += dur
        industry = normalize_text(job.get("industry", ""))
        company = normalize_text(job.get("company", ""))
        title = normalize_text(job.get("title", ""))
        desc = normalize_text(job.get("description", ""))

        is_consulting = (any(ct in industry for ct in consulting_industries) or
                         any(cc in company for cc in CONSULTING_COMPANIES))
        if not is_consulting:
            product_months += dur

        if any(t in title for t in ai_ml_titles):
            title_score += min(4.0, dur / 12)

        score += min(3.0, sum(1 for t in shipped_terms if t in desc) * 0.3)

    if total_months > 0:
        score += (product_months / total_months) * 8.0
    score += min(8.0, title_score)

    if len(career) > 1:
        avg = total_months / len(career)
        score *= 0.7 if avg < 12 else (1.1 if avg > 24 else 1.0)

    if profile.get("current_company_size", "") in ("51-200", "201-500", "501-1000"):
        score += 1.5

    return min(25.0, score)


def score_experience_years(candidate: dict) -> float:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if 5 <= yoe <= 9: return 10.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11: return 7.5
    elif 3 <= yoe < 4 or 11 < yoe <= 13: return 5.0
    elif yoe >= 13: return 3.0
    elif yoe >= 2: return 2.0
    return 0.0


def score_location(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    location = normalize_text(profile.get("location", ""))
    country = normalize_text(profile.get("country", ""))
    relocate = signals.get("willing_to_relocate", False)

    if any(city in location for city in PREFERRED_LOCATIONS):
        return 10.0
    elif country in ("india", "in") or "india" in location:
        return min(10.0, 6.0 + (2.0 if relocate else 0.0))
    return min(3.0, 2.0 if relocate else 0.0)


def score_education(candidate: dict) -> float:
    edu = candidate.get("education", [])
    if not edu:
        return 2.0
    relevant = {"computer science", "cs", "software engineering", "electrical engineering",
                "electronics", "information technology", "machine learning",
                "artificial intelligence", "data science", "statistics", "mathematics", "physics"}
    score = 0.0
    for e in edu:
        field = normalize_text(e.get("field_of_study", ""))
        tier = {"tier_1": 2.0, "tier_2": 1.5, "tier_3": 1.0, "tier_4": 0.5, "unknown": 0.8}.get(
            e.get("tier", "unknown"), 0.5)
        degree = normalize_text(e.get("degree", ""))
        deg_s = 1.5 if any(d in degree for d in ["b.e", "b.tech", "m.tech", "m.s", "phd", "ms"]) else 0.5
        score += any(r in field for r in relevant) * 1.0 + tier * 0.5 + deg_s * 0.3
    return min(5.0, score)


def score_bonus_skills(candidate: dict) -> float:
    ftext = full_text_lower(candidate)
    skill_map = {normalize_text(s.get("name", "")): s for s in candidate.get("skills", [])}
    bonus = 0.0
    for kw in BONUS_SKILLS:
        if kw in skill_map:
            bonus += 0.8 if skill_map[kw].get("duration_months", 0) > 6 else 0.4
        elif kw in ftext:
            bonus += 0.25
    return min(10.0, bonus)


def behavioral_modifier(candidate: dict) -> float:
    signals = candidate.get("redrob_signals", {})
    score = 1.0

    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            days_inactive = (date(2025, 6, 1) -
                             datetime.strptime(last_active_str, "%Y-%m-%d").date()).days
            if days_inactive > 180: score *= 0.5
            elif days_inactive > 90: score *= 0.75
            elif days_inactive > 30: score *= 0.9
        except ValueError:
            pass

    if not signals.get("open_to_work_flag", False): score *= 0.8
    rrr = signals.get("recruiter_response_rate", 0.5)
    if rrr < 0.1: score *= 0.5
    elif rrr < 0.3: score *= 0.75
    elif rrr >= 0.7: score *= 1.05

    notice = signals.get("notice_period_days", 90)
    if notice <= 30: score *= 1.05
    elif notice > 90: score *= 0.80

    icr = signals.get("interview_completion_rate", 0.5)
    if icr < 0.3: score *= 0.75
    elif icr >= 0.8: score *= 1.03

    gh = signals.get("github_activity_score", -1)
    if gh > 60: score *= 1.05
    elif 0 <= gh < 20: score *= 0.95

    pcs = signals.get("profile_completeness_score", 70)
    if pcs < 50: score *= 0.85
    elif pcs >= 85: score *= 1.02

    if signals.get("verified_email") and signals.get("verified_phone"):
        score *= 1.02

    return max(0.3, min(1.0, score))


def compute_stage1_score(candidate: dict, semantic_score: float) -> float:
    """Stage 1: fast hybrid score for all 100K candidates."""
    if detect_honeypot(candidate):
        return 0.0
    is_dq, _ = hard_disqualifier(candidate)
    if is_dq:
        return 1.0

    s_semantic = semantic_score * 25.0
    s_core = score_core_skills(candidate) * (25 / 35)
    s_career = score_career(candidate) * (20 / 25)
    s_yoe = score_experience_years(candidate)
    s_location = score_location(candidate)
    s_edu = score_education(candidate)
    s_bonus = score_bonus_skills(candidate) * 0.5

    raw = s_semantic + s_core + s_career + s_yoe + s_location + s_edu + s_bonus
    return raw * behavioral_modifier(candidate)


# ---------------------------------------------------------------------------
# LLM reasoning layer
# ---------------------------------------------------------------------------

def build_llm_prompt(candidate: dict) -> str:
    """Build a SHORT candidate summary for the LLM — minimizes token usage."""
    p = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    # Top 6 skills by proficiency + endorsements
    skills = candidate.get("skills", [])
    top_skills = sorted(skills, key=lambda s: (
        {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}.get(s.get("proficiency", ""), 0),
        s.get("endorsements", 0)
    ), reverse=True)
    skill_str = ", ".join(f"{s['name']}({s.get('duration_months',0)}mo)"
                          for s in top_skills[:6])

    # Top 3 career roles, description trimmed to 120 chars
    career_parts = []
    for job in candidate.get("career_history", [])[:3]:
        desc = job.get("description", "")[:120]
        career_parts.append(f"{job.get('title')}@{job.get('company')}({job.get('duration_months',0)}mo): {desc}")
    career_str = " | ".join(career_parts)

    gh = signals.get("github_activity_score", -1)
    gh_str = "none" if gh == -1 else f"{gh:.0f}"

    return (
        f"ID:{candidate.get('candidate_id')} {p.get('current_title')}@{p.get('current_company')} "
        f"YOE:{p.get('years_of_experience')} LOC:{p.get('location')},{p.get('country')} "
        f"SKILLS:{skill_str} "
        f"CAREER:{career_str} "
        f"OPEN:{signals.get('open_to_work_flag')} NOTICE:{signals.get('notice_period_days')}d "
        f"GH:{gh_str} RR:{signals.get('recruiter_response_rate',0):.0%} "
        f"Evaluate for Senior AI Engineer role. Respond JSON only."
    )


def call_llm_api(prompt: str, system: str, retries: int = 3) -> dict:
    """Call Groq API (free) and return parsed JSON response. Auto-retries on rate limit."""
    try:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    except ImportError:
        return None

    for attempt in range(retries):
        try:
            message = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                max_tokens=200,   # reduced to save tokens
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = message.choices[0].message.content.strip()
            response_text = re.sub(r"```json\s*|\s*```", "", response_text).strip()
            return json.loads(response_text)

        except json.JSONDecodeError:
            return None

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str:
                wait_match = re.search(r"try again in (\d+)m([\d.]+)s", err_str)
                wait_secs = (int(wait_match.group(1)) * 60 + float(wait_match.group(2)) + 5) if wait_match else 180
                print(f"\n    Rate limited. Waiting {wait_secs:.0f}s then retrying (attempt {attempt+1}/{retries})...")
                time.sleep(wait_secs)
                continue
            else:
                print(f"    API error: {e}")
                return None

    return None  # all retries exhausted


def llm_score_candidates(candidates_with_scores: list, top_n: int = 200) -> dict:
    """
    Run LLM scoring on top N candidates.
    Returns dict: candidate_id -> {score, reasoning, red_flags}
    """
    try:
        from groq import Groq
    except ImportError:
        print("WARNING: groq package not installed. Run: pip install groq")
        print("Skipping LLM layer.")
        return {}

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("WARNING: GROQ_API_KEY not set.")
        print("Set it with: set GROQ_API_KEY=gsk_your-key-here  (Windows)")
        print("Skipping LLM layer.")
        return {}

    print(f"\nLLM reasoning layer: scoring top {top_n} candidates with Claude...")
    results = {}
    errors = 0

    for i, (candidate, stage1_score) in enumerate(candidates_with_scores[:top_n]):
        cid = candidate.get("candidate_id", "")

        if i % 50 == 0:
            print(f"  LLM scoring {i}/{top_n}... ({errors} errors so far)")

        prompt = build_llm_prompt(candidate)
        result = call_llm_api(prompt, LLM_SYSTEM_PROMPT)

        if result and "score" in result:
            results[cid] = {
                "llm_score": float(result.get("score", 5.0)) / 10.0,  # normalize to 0-1
                "reasoning": result.get("reasoning", ""),
                "red_flags": result.get("red_flags", ""),
            }
        else:
            errors += 1
            # Fallback: use stage1 score normalized
            results[cid] = {
                "llm_score": min(1.0, stage1_score / 100.0),
                "reasoning": "Scored by hybrid semantic+rule system.",
                "red_flags": "",
            }

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    print(f"LLM scoring complete. {len(results)} scored, {errors} errors.")
    return results


def build_final_reasoning(candidate: dict, stage1_score: float,
                           llm_result: dict) -> str:
    """Build the final reasoning string for the CSV."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    country = profile.get("country", "")
    notice = signals.get("notice_period_days", "?")

    ftext = full_text_lower(candidate)
    top_skills = [kw for kw in [
        "embeddings", "sentence transformers", "faiss", "pinecone", "qdrant",
        "elasticsearch", "bm25", "ndcg", "mrr", "semantic search",
        "rag", "fine-tuning", "python", "nlp", "learning to rank"
    ] if kw in ftext][:3]
    skill_str = ", ".join(top_skills) if top_skills else "general ML"

    loc_str = f"{location}, {country}" if location else country

    if llm_result and llm_result.get("reasoning"):
        # Use LLM reasoning — it's better
        base = llm_result["reasoning"]
        red_flags = llm_result.get("red_flags", "")
        flag_str = f" Flag: {red_flags}" if red_flags else ""
        return f"{base}{flag_str} [{title} @ {company}, {yoe:.0f}yr, {loc_str}, {notice}d notice]"
    else:
        # Fallback to rule-based reasoning
        concerns = []
        if isinstance(notice, int) and notice > 60:
            concerns.append(f"{notice}d notice")
        if signals.get("recruiter_response_rate", 1) < 0.3:
            concerns.append("low response rate")
        if not signals.get("open_to_work_flag", False):
            concerns.append("not open-to-work")
        concern_str = ("; concern: " + ", ".join(concerns)) if concerns else ""
        return (f"{title} at {company} ({yoe:.0f}yr, {loc_str}); "
                f"key skills: {skill_str}{concern_str}.")


# ---------------------------------------------------------------------------
# Embedding (same as v2)
# ---------------------------------------------------------------------------

def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        print("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.")
        return model, np
    except Exception as e:
        print(f"WARNING: Could not load embedding model: {e}")
        return None, None


def compute_semantic_scores(candidates: list, model, np) -> list:
    print("Building candidate texts...")
    texts = [build_candidate_text(c)[:2000] for c in candidates]

    print("Embedding JD...")
    jd_emb = model.encode([JD_TEXT[:2000]], batch_size=1,
                           show_progress_bar=False, normalize_embeddings=True)

    print(f"Embedding {len(texts)} candidates...")
    BATCH_SIZE = 256
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embs = model.encode(batch, batch_size=BATCH_SIZE,
                            show_progress_bar=False, normalize_embeddings=True)
        all_embs.append(embs)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}...")

    all_embs = np.vstack(all_embs)
    sims = (all_embs @ jd_emb.T).flatten()
    sim_min, sim_max = float(sims.min()), float(sims.max())
    if sim_max > sim_min:
        normalized = (sims - sim_min) / (sim_max - sim_min)
    else:
        normalized = sims
    return [float(s) for s in normalized]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker v3 (Semantic + Rules + LLM)")
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out", default="submission_v3.csv")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-top-n", type=int, default=200,
                        help="How many top candidates to send to LLM (default 500)")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    start_time = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Load + semantic score ──────────────────────────────────────
    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates.")

    semantic_scores = [0.5] * len(candidates)
    if not args.no_embeddings:
        model, np = load_embedding_model()
        if model is not None:
            semantic_scores = compute_semantic_scores(candidates, model, np)
            print(f"Semantic range: {min(semantic_scores):.3f} – {max(semantic_scores):.3f}")
    else:
        print("Skipping embeddings.")

    print("Stage 1: computing hybrid scores for all candidates...")
    stage1 = []
    for i, (c, sem) in enumerate(zip(candidates, semantic_scores)):
        if i % 10000 == 0 and i > 0:
            print(f"  {i}/{len(candidates)}...")
        stage1.append((c, compute_stage1_score(c, sem)))

    stage1.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))
    print(f"Stage 1 done. Top score: {stage1[0][1]:.2f}")

    elapsed1 = time.time() - start_time
    print(f"Stage 1 elapsed: {elapsed1:.0f}s")

    # ── Stage 2: LLM reasoning on top N ─────────────────────────────────────
    llm_results = {}
    if not args.no_llm:
        llm_results = llm_score_candidates(stage1, top_n=args.llm_top_n)

    # ── Stage 3: Final hybrid score ──────────────────────────────────────────
    print("Stage 3: computing final scores...")
    final_scored = []

    for candidate, s1_score in stage1[:max(args.llm_top_n, args.top_n * 3)]:
        cid = candidate.get("candidate_id", "")
        llm_result = llm_results.get(cid)

        if llm_result:
            # Hybrid: 60% stage1 (normalized) + 40% LLM
            s1_norm = min(1.0, s1_score / 100.0)
            llm_score = llm_result["llm_score"]
            final = 0.6 * s1_norm + 0.4 * llm_score
        else:
            final = min(1.0, s1_score / 100.0)

        reasoning = build_final_reasoning(candidate, s1_score, llm_result)
        final_scored.append((cid, final, reasoning))

    final_scored.sort(key=lambda x: (-x[1], x[0]))
    top_n = final_scored[:args.top_n]

    # Normalize scores to ensure clean 0-1 range, non-increasing
    max_score = top_n[0][1]
    min_score = top_n[-1][1]
    score_range = max_score - min_score if max_score > min_score else 1.0

    def norm(s):
        return round(0.5 + 0.5 * (s - min_score) / score_range, 4)

    # Write CSV
    print(f"Writing top {args.top_n} to {out_path}...")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning) in enumerate(top_n, 1):
            writer.writerow([cid, rank, norm(score), reasoning])

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {out_path}")
    print(f"Top 3:")
    for cid, score, reasoning in top_n[:3]:
        print(f"  {cid} ({norm(score)}) — {reasoning[:100]}")
    print(f"\nNext: python validate_submission.py {out_path}")


if __name__ == "__main__":
    main()