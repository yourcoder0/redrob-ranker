# Redrob Hackathon — Intelligent Candidate Ranker v3

**Challenge:** Intelligent Candidate Discovery & Ranking  
**Role:** Senior AI Engineer (Founding Team) @ Redrob AI  
**Dataset:** 100,000 candidates (JSONL)  
**Output:** Top 100 candidates, ranked best-fit first  

---

## Architecture — 3-Stage Hybrid Pipeline

```
100,000 candidates
       │
       ▼
┌─────────────────────────────────────┐
│  STAGE 1: Semantic + Rule Scoring   │  ~2 hrs on CPU
│  • sentence-transformers embeddings │
│  • Cosine similarity vs JD          │
│  • Rule-based component scores      │
│  • Behavioral signal modifier       │
│  → Top 200 candidates shortlisted   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  STAGE 2: LLM Reasoning (Groq)      │  ~15 min
│  • LLaMA 3.1 8B scores each of     │
│    top 200 candidates               │
│  • Genuine fit assessment           │
│  • Catches nuance rules miss        │
│  → LLM score (0-10) per candidate   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  STAGE 3: Final Hybrid Score        │  instant
│  • 60% Stage 1 + 40% LLM score     │
│  • Sort, take top 100               │
│  → submission.csv                   │
└─────────────────────────────────────┘
```

---

## Scoring Components

### Stage 1 — Semantic + Rule Hybrid (max 100 pts × behavioral modifier)

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Semantic similarity** | 25 pts | Cosine similarity between candidate text and JD via `all-MiniLM-L6-v2` |
| **Core Skills Match** | 25 pts | Embeddings, vector DB, ranking/eval, Python — with trust multiplier |
| **Career Trajectory** | 20 pts | Product companies, shipped systems, tenure stability |
| **Experience Years** | 10 pts | Sweet spot 5–9 years per JD |
| **Location/Availability** | 10 pts | India-based preferred, relocation willingness |
| **Education** | 5 pts | Field relevance + institution tier |
| **Bonus Skills** | 5 pts | RAG, fine-tuning, LTR, A/B testing, open source |

**× Behavioral Modifier (0.3–1.0)** — answers *"Is this person actually hirable right now?"*
- Last active date (stale >180 days → 0.5×)
- `open_to_work_flag` (false → 0.8×)
- `recruiter_response_rate` (<10% → 0.5×, >70% → 1.05×)
- Notice period (≤30d → 1.05×, >90d → 0.8×)
- `interview_completion_rate`, `github_activity_score`, profile completeness

### Stage 2 — LLM Reasoning

Top 200 candidates from Stage 1 are evaluated by **LLaMA 3.1 8B via Groq API** (free tier).

The LLM receives a structured candidate summary and returns:
```json
{
  "score": 8.5,
  "reasoning": "Built and shipped production semantic search at Zomato with FAISS...",
  "red_flags": ""
}
```

Score guide the LLM follows:
- **0-2**: Hard disqualified (wrong domain, consulting-only, non-technical)
- **3-4**: Weak fit (some skills but missing core requirements)
- **5-6**: Moderate fit (decent background, gaps in key areas)
- **7-8**: Strong fit (solid relevant experience, most requirements met)
- **9-10**: Exceptional fit (shipped relevant systems, exactly what we need)

### Stage 3 — Final Score

```
final_score = 0.60 × stage1_normalized + 0.40 × llm_score_normalized
```

---

## Trap Avoidance

### Honeypot Detection (`detect_honeypot()`)
- Company founded after 2022 but candidate claims 8+ years there
- Expert proficiency on 8+ skills with 0 duration months each
- Total skill duration >> 15× total career months
- Years of experience inconsistent with career history by >10 years

### Hard Disqualifiers (`hard_disqualifier()`)
- Non-technical current title (marketing, HR, design etc.) with no prior ML career
- Entire career at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra) with no product company
- CV/speech/robotics specialist with zero NLP/IR exposure
- No technical roles anywhere in career history

### Keyword Stuffer Detection
Skills receive a **trust multiplier**:
```
trust = proficiency_weight × 0.4 + log(endorsements+1) × 0.3 + duration_weight × 0.3
```
A skill listed with 0 months duration and 0 endorsements gets **0.2× weight** — keyword stuffing cannot dominate the score.

### Skill Cluster Logic
4 clusters must each be present for maximum core score:
1. **Embeddings/retrieval** — sentence-transformers, BGE, E5, semantic search, bi-encoder
2. **Vector DB** — Pinecone, Qdrant, Milvus, FAISS, Elasticsearch, Weaviate, pgvector
3. **Ranking/eval** — NDCG, MRR, MAP, BM25, LTR, reranking, information retrieval
4. **Python** — explicitly required by JD

Missing any cluster caps the score even if many other skills match.

---

## Reproduce

### Requirements

```
pip install sentence-transformers groq
```

Set your free Groq API key (get one at console.groq.com):
```bash
# Windows
set GROQ_API_KEY=gsk_your-key-here

# Mac/Linux
export GROQ_API_KEY=gsk_your-key-here
```

### Run full pipeline

```bash
python rank_v3.py --candidates candidates.jsonl --out submission.csv
```

Runtime breakdown:
- Stage 1 (embeddings): ~2 hours on CPU for 100K candidates
- Stage 2 (LLM): ~15 minutes for top 200 candidates  
- Stage 3 (final scoring): instant

### Run without LLM (faster, less accurate)

```bash
python rank_v3.py --candidates candidates.jsonl --out submission.csv --no-llm
```

~2 hours total, no API key needed.

### Run without embeddings (fastest, least accurate)

```bash
python rank_v3.py --candidates candidates.jsonl --out submission.csv --no-embeddings --no-llm
```

~40 seconds total, pure rule-based.

---

## Files

```
rank_v3.py                    — Main ranker (single file)
app.py                        — Streamlit sandbox UI
README.md                     — This file
requirements.txt              — sentence-transformers, groq, streamlit
submission_metadata.yaml      — Submission metadata
```

---

## Why This Approach Beats Pure Keyword Matching

The challenge says: *"not by matching keywords, but by actually understanding who fits the role."*

Three layers of understanding:

**1. Semantic embeddings** understand that *"built dense retrieval system for e-commerce search"* is relevant even if it doesn't contain the word "FAISS."

**2. Trust multiplier** understands that listing "embeddings" as a skill with 0 months used and 0 endorsements is very different from listing it with 36 months and 45 endorsements.

**3. LLM reasoning** understands that *"Senior ML Engineer at Zomato who built production semantic search serving 10M users"* is a stronger signal than someone who merely listed all the right keywords.

Rules handle the filtering. Embeddings handle the semantic gap. LLM handles the nuance. Together they approximate what a great recruiter does.
