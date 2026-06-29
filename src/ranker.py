import json
import math
import time
import csv
import os
from datetime import datetime, date
from collections import defaultdict
from typing import Optional

JD_MUST_SKILLS = {
    "embedding": 14, "embeddings": 14, "sentence-transformers": 12, "sentence transformer": 12,
    "bge": 9, "e5": 7, "dense retrieval": 10, "bi-encoder": 10, "cross-encoder": 8,
    "faiss": 12, "pinecone": 11, "weaviate": 11, "qdrant": 11, "milvus": 11,
    "opensearch": 10, "elasticsearch": 10, "vector search": 12, "vector database": 12,
    "vector store": 10, "ann": 8, "hnsw": 8,
    "hybrid search": 12, "semantic search": 11, "retrieval": 9, "bm25": 10,
    "information retrieval": 10, "reranking": 11, "re-ranking": 11, "re-rank": 10,
    "ndcg": 11, "mrr": 10, "mean average precision": 8, "ranking evaluation": 9,
    "a/b testing": 8, "ab testing": 8, "offline evaluation": 8, "online evaluation": 8,
    "python": 5, "pytorch": 7, "tensorflow": 6, "transformers": 7,
    "huggingface": 7, "scikit-learn": 5, "sklearn": 5,
    "llm": 9, "large language model": 9, "rag": 12, "retrieval augmented": 11,
    "fine-tuning": 8, "fine-tune": 8, "lora": 7, "qlora": 7, "peft": 7,
    "learning to rank": 11, "ltr": 9, "xgboost": 6, "lightgbm": 6,
    "recommendation system": 8, "recommender": 8, "ranking system": 9,
    "search infrastructure": 10, "search engine": 7,
}

JD_GOOD_SKILLS = {
    "distributed systems": 4, "kafka": 3, "spark": 3, "ray": 4,
    "hr tech": 5, "recruiting": 4, "talent": 3,
    "open source": 3, "kaggle": 2,
    "llamaindex": 3, "langchain": 1,
    "mlops": 4, "model serving": 4, "inference": 3,
}

JD_NEG_SKILLS = {
    "computer vision": -8, "opencv": -6, "yolo": -6, "object detection": -6,
    "image classification": -5, "image segmentation": -5,
    "speech recognition": -7, "asr": -7, "tts": -7, "text-to-speech": -7,
    "robotics": -8, "autonomous driving": -8, "slam": -6,
    "photoshop": -2, "illustrator": -2, "graphic design": -3,
    "digital marketing": -4, "seo": -3, "social media": -3,
    "sales": -3, "crm": -2, "salesforce crm": -3,
    "six sigma": -2, "autocad": -3,
}

ENGINEERING_TITLE_KW = [
    "engineer", "scientist", "developer", "architect", "researcher",
    "ml ", "ai ", "nlp", "data scientist", "machine learning", "deep learning",
    "search", "retrieval", "ranking", "recommendation", "applied",
    "backend", "software", "platform", "infrastructure", "tech lead",
    "principal", "staff", "distinguished",
]

HARD_NON_ENGINEERING = [
    "customer support", "hr manager", "hr executive", "recruiter",
    "content writer", "content writing", "digital marketing", "marketing manager",
    "sales manager", "sales executive", "account manager", "business analyst",
    "product manager", "operations manager", "finance", "accountant",
    "teacher", "professor", "lawyer", "legal",
    "mechanical engineer", "civil engineer", "structural engineer",
    "electrical engineer", "chemical engineer",
    "brand designer", "graphic designer", "ux designer",
]

SERVICES_COS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl technologies", "hcltech", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree",
}

ML_WORK_KEYWORDS = [
    "trained", "deployed", "fine-tuned", "embedding", "retrieval", "ranking",
    "recommendation", "search", "inference", "model", "vector", "semantic",
    "nlp", "transformer", "bert", "gpt", "llm", "rag", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "pytorch", "tensorflow", "huggingface", "scikit", "ndcg", "recall",
    "precision", "a/b test", "online eval", "offline eval", "rerank",
    "bm25", "hybrid search", "dense retrieval", "bi-encoder",
]

NON_ML_CAREER_KW = [
    "content writing", "seo strategy", "brand design", "creative direction",
    "customer support", "sales quota", "fulfillment operations", "warehouses",
    "mechanical design", "dfm", "dfma", "autocad", "solidworks",
    "civil engineering", "structural", "construction",
    "digital marketing", "social media", "graphic design",
]

INDIA_CITIES = {
    "pune", "noida", "delhi", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "gurgaon", "gurugram", "chennai", "kolkata", "ahmedabad", "jaipur",
    "kochi", "coimbatore", "indore", "trivandrum", "vizag", "nagpur",
    "chandigarh", "lucknow", "bhopal", "surat", "vadodara",
}


def today() -> date:
    return date.today()


def days_since(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        d = datetime.fromisoformat(date_str[:10]).date()
        return (today() - d).days
    except Exception:
        return None


def career_text(candidate: dict) -> str:
    parts = []
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", "").lower())
        parts.append(job.get("description", "").lower())
        parts.append(job.get("company", "").lower())
    return " ".join(parts)


def full_text(candidate: dict) -> str:
    p = candidate.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        career_text(candidate),
    ]
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(parts).lower()


def is_honeypot(candidate: dict) -> bool:
    today_year = today().year
    career = candidate.get("career_history", [])

    for job in career:
        start = job.get("start_date", "")
        end = job.get("end_date")
        dur = job.get("duration_months", 0)
        try:
            sy = int(start[:4])
        except Exception:
            continue
        if sy < 1970 or sy > today_year + 1:
            return True
        if dur < 0:
            return True
        if end:
            try:
                ey = int(end[:4])
                if ey > today_year + 1 or sy > ey:
                    return True
            except Exception:
                pass

    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    skills = candidate.get("skills", [])

    zero_month_expert = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced") and s.get("duration_months", 1) == 0
    ]
    if len(zero_month_expert) >= 4:
        return True

    total_skill_months = sum(s.get("duration_months", 0) for s in skills)
    if yoe > 0 and total_skill_months > yoe * 12 * 9:
        return True

    return False


def ml_career_evidence(candidate: dict) -> float:
    ctext = career_text(candidate)

    ml_hits = sum(1 for kw in ML_WORK_KEYWORDS if kw in ctext)
    non_ml_hits = sum(1 for kw in NON_ML_CAREER_KW if kw in ctext)

    raw = ml_hits - non_ml_hits * 3
    return max(0.0, min(1.0, raw / 15.0))


def score_skills(candidate: dict) -> tuple:
    skills_list = candidate.get("skills", [])
    skill_map = {s["name"].lower(): s for s in skills_list}
    ftext = full_text(candidate)
    ctext = career_text(candidate)

    career_ev = ml_career_evidence(candidate)

    total = 0.0
    matched = []

    for skill_key, weight in JD_MUST_SKILLS.items():
        if skill_key in skill_map:
            s = skill_map[skill_key]
            prof_mult = {"expert": 1.3, "advanced": 1.1, "intermediate": 0.9, "beginner": 0.55}.get(
                s.get("proficiency", ""), 0.75
            )
            dur = s.get("duration_months", 0)
            dur_mult = 0.7 + 0.3 * min(1.0, dur / 24.0)
            total += weight * prof_mult * dur_mult
            matched.append(skill_key)
        elif skill_key in ctext:
            total += weight * 0.55 * career_ev
            matched.append("~" + skill_key)
        elif skill_key in ftext:
            total += weight * 0.2 * career_ev

    for skill_key, weight in JD_GOOD_SKILLS.items():
        if skill_key in skill_map:
            total += weight * 0.8
        elif skill_key in ctext:
            total += weight * 0.35 * career_ev

    for skill_key, penalty in JD_NEG_SKILLS.items():
        if skill_key in skill_map:
            total += penalty * 1.5
        elif skill_key in ftext:
            total += penalty

    max_possible = sum(w * 1.3 for w in JD_MUST_SKILLS.values()) * 0.22
    normalized = max(0.0, min(1.0, total / max_possible))
    return normalized, matched


def score_career(candidate: dict) -> float:
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    ftext = full_text(candidate)
    ctext = career_text(candidate)
    title_lower = p.get("current_title", "").lower()

    score = 0.0

    is_hard_non_eng = any(kw in title_lower for kw in HARD_NON_ENGINEERING)
    is_eng = any(kw in title_lower for kw in ENGINEERING_TITLE_KW)

    if is_hard_non_eng:
        score -= 45
    elif not is_eng:
        score -= 12

    yoe = p.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        score += 25
    elif 4 <= yoe < 5:
        score += 18
    elif 9 < yoe <= 11:
        score += 16
    elif 3 <= yoe < 4 or 11 < yoe <= 14:
        score += 8
    elif yoe > 14:
        score += 3
    else:
        score += max(0, 4 - abs(yoe - 6))

    career_ev = ml_career_evidence(candidate)
    score += career_ev * 28

    prod_kw = [
        "production", "deployed", "shipped", "real users", "at scale",
        "million users", "serving", "api endpoint", "launched", "live system",
        "production traffic", "production inference",
    ]
    research_only_kw = [
        "research lab only", "academic only", "no production", "phd research only",
    ]
    prod_count = sum(1 for kw in prod_kw if kw in ftext)
    research_pen = sum(1 for kw in research_only_kw if kw in ftext)
    score += min(18, prod_count * 2.5) - research_pen * 4

    product_months = 0
    services_months = 0
    all_services = True
    for job in career:
        co = job.get("company", "").lower()
        ind = job.get("industry", "").lower()
        dur = job.get("duration_months", 0)
        is_svc = any(sc in co for sc in SERVICES_COS) or "it services" in ind
        if is_svc:
            services_months += dur
        else:
            product_months += dur
            all_services = False

    if all_services and len(career) > 0:
        score -= 18
    total_months = product_months + services_months
    if total_months > 0:
        score += (product_months / total_months) * 18

    if len(career) >= 2:
        avg_tenure = sum(j.get("duration_months", 0) for j in career) / len(career)
        if avg_tenure < 10:
            score -= 12
        elif avg_tenure < 18:
            score -= 5
        elif avg_tenure >= 30:
            score += 6

    if career:
        recent = career[0]
        recent_text = (recent.get("title", "") + " " + recent.get("description", "")).lower()
        ai_hits = sum(1 for kw in [
            "ml", "ai", "llm", "nlp", "embedding", "retrieval", "ranking",
            "model", "vector", "search", "recommendation", "transformer",
        ] if kw in recent_text)
        score += min(15, ai_hits * 2.5)

    cv_dom = sum(1 for kw in ["computer vision", "opencv", "yolo", "resnet", "cnn", "object detection", "image classification", "speech recognition", "tts", "robotics", "asr"] if kw in ftext)
    nlp_dom = sum(1 for kw in ["nlp", "information retrieval", "ranking", "retrieval", "search", "embedding", "vector", "semantic", "recommendation", "rag", "llm"] if kw in ftext)
    if cv_dom > nlp_dom:
        score -= 18
    if "computer vision" in title_lower or "cv engineer" in title_lower:
        score -= 20

    if title_lower in ["cto", "chief technology officer", "vp of engineering", "director of engineering"]:
        score -= 10

    return max(0.0, min(1.0, score / 100.0))


def score_location(candidate: dict, signals: dict) -> float:
    p = candidate.get("profile", {})
    location = p.get("location", "").lower()
    country = p.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)

    if country == "india":
        if any(city in location for city in INDIA_CITIES):
            return 1.0
        elif relocate:
            return 0.82
        else:
            return 0.65
    else:
        if relocate:
            return 0.38
        return 0.18


def score_behavioral(signals: dict) -> float:
    score = 0.0
    weight_total = 0.0

    last_active = days_since(signals.get("last_active_date"))
    if last_active is not None:
        if last_active <= 7:
            act = 20
        elif last_active <= 30:
            act = 16
        elif last_active <= 60:
            act = 11
        elif last_active <= 90:
            act = 6
        elif last_active <= 180:
            act = 2
        else:
            act = -6
        score += act
    weight_total += 20

    otw = signals.get("open_to_work_flag", False)
    score += 10 if otw else 0
    weight_total += 10

    rr = signals.get("recruiter_response_rate", -1)
    if rr >= 0:
        score += rr * 16
    weight_total += 16

    rt = signals.get("avg_response_time_hours", -1)
    if rt >= 0:
        score += max(0, 10 - rt / 4.0)
    weight_total += 10

    icr = signals.get("interview_completion_rate", -1)
    if icr >= 0:
        score += icr * 10
    weight_total += 10

    oar = signals.get("offer_acceptance_rate", -1)
    if oar >= 0:
        score += oar * 7
    weight_total += 7

    notice = signals.get("notice_period_days", -1)
    if notice >= 0:
        if notice <= 15:
            score += 9
        elif notice <= 30:
            score += 7
        elif notice <= 60:
            score += 4
        elif notice <= 90:
            score += 1
        else:
            score -= 3
    weight_total += 9

    gh = signals.get("github_activity_score", -1)
    if gh >= 0:
        score += (gh / 100.0) * 9
    weight_total += 9

    pc = signals.get("profile_completeness_score", 0)
    score += (pc / 100.0) * 5
    weight_total += 5

    sal = signals.get("expected_salary_range_inr_lpa", {})
    if isinstance(sal, dict):
        sal_min = sal.get("min", 0)
        if 15 <= sal_min <= 80:
            score += 3
        elif sal_min > 120:
            score -= 2
    weight_total += 3

    wm = signals.get("preferred_work_mode", "")
    if wm in ("hybrid", "flexible", "onsite"):
        score += 3
    weight_total += 3

    sv = signals.get("saved_by_recruiters_30d", 0)
    if sv >= 5:
        score += 3
    elif sv >= 2:
        score += 1
    weight_total += 3

    if weight_total == 0:
        return 0.5
    return max(0.0, min(1.0, score / weight_total))


def compute_score(candidate: dict) -> tuple:
    if is_honeypot(candidate):
        return 0.0, "HONEYPOT: Impossible profile — scored zero"

    signals = candidate.get("redrob_signals", {})

    skill_score, matched = score_skills(candidate)
    career_score = score_career(candidate)
    location_score = score_location(candidate, signals)
    behavioral_score = score_behavioral(signals)

    composite = (
        0.40 * skill_score +
        0.30 * career_score +
        0.20 * behavioral_score +
        0.10 * location_score
    )

    if skill_score < 0.04:
        composite *= 0.25

    if career_score < 0.15:
        composite *= 0.60

    otw = signals.get("open_to_work_flag", False)
    rr = signals.get("recruiter_response_rate", 0)
    if otw and rr > 0.70:
        composite = min(1.0, composite * 1.06)

    last_active = days_since(signals.get("last_active_date"))
    if last_active is not None and last_active > 270:
        composite *= 0.80

    reasoning = build_reasoning(candidate, signals, skill_score, career_score, behavioral_score, location_score, matched)
    return round(composite, 6), reasoning


def build_reasoning(candidate, signals, skill_score, career_score, behavioral_score, location_score, matched) -> str:
    p = candidate.get("profile", {})
    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    loc = p.get("location", "")
    country = p.get("country", "")

    listed = [s for s in matched if not s.startswith("~")][:3]
    text_match = [s.replace("~", "") for s in matched if s.startswith("~")][:3]
    display = listed if listed else text_match
    skills_str = ", ".join(display[:3]) if display else "limited core AI/ML skills"

    rr = signals.get("recruiter_response_rate", -1)
    notice = signals.get("notice_period_days", -1)
    last_active = days_since(signals.get("last_active_date"))
    otw = signals.get("open_to_work_flag", False)

    part1 = f"{title} at {company}, {yoe:.1f}yrs exp; core skills: {skills_str}."

    notes = []
    if otw:
        notes.append("actively open to work")
    if rr >= 0 and rr > 0.65:
        notes.append(f"responsive ({rr:.0%} reply rate)")
    if notice >= 0 and notice <= 30:
        notes.append(f"short notice ({notice}d)")
    if country.lower() != "india":
        notes.append(f"outside India ({country})")
    if last_active is not None and last_active > 180:
        notes.append(f"inactive {last_active}d")
    if rr >= 0 and rr < 0.30:
        notes.append(f"low response rate ({rr:.0%})")
    if notice >= 0 and notice > 90:
        notes.append(f"long notice ({notice}d)")
    if career_score < 0.30:
        notes.append("limited production ML deployment evidence")

    part2 = "; ".join(notes[:3]) if notes else f"location: {loc}"
    return f"{part1} {part2}".strip()


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "candidates.jsonl")
    out_path = os.path.join(os.path.dirname(__file__), "..", "output", "team_devnira.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Scoring 100K candidates...")
    t0 = time.time()

    scored = []
    total = 0
    honeypots = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            score, reasoning = compute_score(c)
            if score == 0.0 and "HONEYPOT" in reasoning:
                honeypots += 1
            scored.append((c["candidate_id"], score, reasoning))

            if total % 10000 == 0:
                print(f"  {total:,} done in {time.time()-t0:.1f}s")

    print(f"Total: {total:,} | Honeypots: {honeypots}")

    scored.sort(key=lambda x: (-x[1], x[0]))
    top100 = scored[:100]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning) in enumerate(top100, 1):
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s -> {out_path}")
    print("Top 5:")
    for i, (cid, score, reasoning) in enumerate(top100[:5], 1):
        print(f"  {i}. {cid} | {score:.4f} | {reasoning[:90]}")


if __name__ == "__main__":
    main()
