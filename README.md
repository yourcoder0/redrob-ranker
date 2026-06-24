# Redrob Hackathon — Intelligent Candidate Ranker

**Challenge:** Intelligent Candidate Discovery & Ranking  
**Role being ranked for:** Senior AI Engineer (Founding Team) @ Redrob AI  
**Dataset:** 100,000 candidates (JSONL)  
**Output:** Top 100 candidates, ranked best-fit first

---

## Architecture

A **multi-component rule-based ranker** with explicit reasoning capture. No GPU, no network, no LLM APIs. Runs in ~40 seconds on CPU for 100K candidates.

### Scoring Components (max 100 pts)

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Core Skills Match | 35 pts | Embeddings, vector DB, ranking/eval, Python — with trust multiplier |
| Career Trajectory | 25 pts | Product companies, shipped systems, tenure quality |
| Experience Years | 10 pts | Sweet spot 5–9 years per JD |
| Location/Availability | 10 pts | India-based preferred, relocation willingness |
| Education | 5 pts | Field relevance + institution tier |
| Bonus Skills | up to 5 pts | RAG, fine-tuning, LTR, A/B testing, etc. |

**Final score = component total × behavioral modifier (0.3–1.0)**

### Behavioral Signal Modifier

Derived from `redrob_signals` to answer: *"Is this person actually hirable right now?"*

- Last active date (stale profiles penalized up to 50%)
- `open_to_work_flag`
- `recruiter_response_rate` (< 10% → 50% modifier)
- Notice period (≤30d bonus, >90d penalty)
- `interview_completion_rate`
- `github_activity_score`
- Profile completeness + verified contact

### Trap Avoidance

**Honeypot detection** (`detect_honeypot()`):
- Company founding date vs claimed tenure mismatch
- Expert proficiency with 0-duration months on 8+ skills
- Total skill duration >> plausible career months
- Years of experience wildly inconsistent with career history

**Hard disqualifiers** (`hard_disqualifier()`):
- Non-technical current title with no prior technical career
- Entire career at known consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) with no product company experience
- Primary expertise is CV/speech/robotics with no NLP/IR exposure
- No technical roles in career history

**Keyword-stuffer trap**: Skills receive a trust multiplier = weighted combination of proficiency level, endorsement count (log-scaled), and duration in months. Skills listed with 0 months duration and 0 endorsements get 0.2× weight — they can't dominate.

### Skill Cluster Logic

Core skills are organized into 4 clusters that must each be present for maximum score:
1. **Embeddings/retrieval** — sentence-transformers, BGE, E5, semantic search, dense retrieval
2. **Vector DB / hybrid search** — Pinecone, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch
3. **Ranking/eval** — NDCG, MRR, MAP, BM25, LTR, reranking, information retrieval
4. **Python** — explicitly required by JD

Each cluster contributes 5 pts. Missing clusters cap the score even if a candidate has many other relevant skills.

---

## Reproduce

### Requirements

```
python >= 3.10
```

No external packages required — pure stdlib.

### Run

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runtime: ~40 seconds on a modern CPU for 100K candidates.  
Memory: <2 GB.  
No network access required.

---

## Files

```
rank.py                       — Main ranker (single file, no deps)
README.md                     — This file
submission_metadata.yaml      — Submission metadata
requirements.txt              — Empty (stdlib only)
```

---

## Design Notes

### Why rule-based over embeddings/LLM?

The compute constraint (5 min CPU, no GPU, no network) rules out:
- Per-candidate LLM calls (too slow, network required)
- Local LLM inference (too slow on CPU)
- Real-time embedding generation for 100K candidates (marginal benefit, much slower)

The JD explicitly says the right answer involves understanding "the gap between what the JD says and what it means." A well-specified rule-based system with a careful trust multiplier is more interpretable and more defensible in the Stage 5 interview than a black-box embedding similarity score.

### Why not BM25 / TF-IDF?

Pure keyword matching is exactly the trap the challenge warns about. A candidate whose summary says "I specialize in embeddings and retrieval" scores the same as one who listed those as skills with 0 months used and 0 endorsements. Our trust multiplier is the fix.

### Why is the career trajectory component worth 25 pts?

The JD explicitly disqualifies:
- People with 0 production deployments
- Consulting-only careers
- Non-coders who moved into "architecture"

No skill list tells you if someone actually shipped to production. The career component reads descriptions for shipped-system evidence (`deployed`, `production`, `real users`, `end-to-end`) and penalizes short-tenure title-chasers.
