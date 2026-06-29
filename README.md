# Redrob Intelligent Candidate Ranker

**Challenge:** India Runs Data and AI Challenge — Intelligent Candidate Discovery & Ranking
**Team:** Devnira

## What This Is

A multi-signal candidate ranking system that scores 100,000 candidates in under 60 seconds on CPU, without vector embeddings or external APIs.

## Architecture

4 scoring dimensions weighted as a composite:

- **Skill Matching (40%)** — 30+ must-have keywords with proficiency and duration multipliers. Text-fallback matching against career descriptions. Negative signals for wrong domains (computer vision, speech, robotics).
- **Career Quality (30%)** — Production deployment evidence, product vs services company ratio, tenure stability, non-engineering title penalty, recent AI/ML role signals.
- **Behavioral Signals (20%)** — Last active date, recruiter response rate, interview completion, notice period, GitHub activity, offer acceptance rate.
- **Location Fit (10%)** — India-first scoring with city-level preference (Pune, Noida, Delhi, Hyderabad, Mumbai, Bangalore).

## Honeypot Detection

Flags candidates with impossible profiles before scoring:
- End date before start date or impossible year values
- Five or more advanced skills with zero months of experience
- Total skill-months exceeding 9x career years

Result: 5,858 honeypots flagged and scored zero. Zero honeypots in top 100.

## Running Locally

```bash
pip install -r requirements.txt
python src/ranker.py
streamlit run app.py
```

## Submission Output

`output/team_devnira.csv` — top 100 ranked candidates with score and reasoning.

Passes the official validator: `python validate_submission.py output/team_devnira.csv`

## Design Decisions

**Why not vector embeddings?** The 5-minute CPU-only constraint rules out loading a sentence-transformer and embedding 100K candidates. Structured feature scoring achieves strong precision with millisecond-per-candidate latency.

**Why 40% skill weight?** Skills are the hard gate. A candidate without vector DB and retrieval experience cannot do this job regardless of behavioral signals.

**Why penalize services companies softly?** We apply a -18 point penalty rather than a hard filter because prior product company experience redeems a current services role.

**Why the non-engineering title penalty is so strong (-45 pts)?** The dataset contains candidates titled Customer Support, Mechanical Engineer, and Civil Engineer with perfect AI skill lists. Their career history shows zero months of ML work. The penalty correctly buries them regardless of their skills section.
