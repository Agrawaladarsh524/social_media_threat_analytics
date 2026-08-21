"""
Local NLP + feature engineering. No API calls — spaCy's pretrained
en_core_web_sm model runs entirely on-device for named-entity recognition;
TF-IDF + cosine similarity (scikit-learn) for semantic content matching;
everything else is plain regex/statistics.

Produces two things per profile:
1. A flat feature dict consumed by risk_engine.score_*_profile().
2. `inference_rows` — real extracted entities with the source text as
   evidence, replacing the old OpenAI-generated PII inference table.
"""

from __future__ import annotations

import json
import re
from typing import Any

import spacy

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_sm', disable=['lemmatizer'])
    return _nlp


EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')

SENSITIVE_KEYWORDS: dict[str, list[str]] = {
    'travel': ['flying to', 'landed in', 'airport', 'vacation', 'trip to', 'on my way to', 'just arrived'],
    'home': ['my house', 'my apartment', 'my home', 'moved to', 'new place', 'my neighborhood'],
    'family': ['my wife', 'my husband', 'my kids', 'my son', 'my daughter', 'my family', 'my kid'],
    'finance': ['just closed', 'raised a round', 'my salary', 'bought a', 'net worth', 'investment portfolio'],
    'schedule': ['every morning', 'every day at', 'my routine', 'usually at', 'every week'],
}
_ALL_KEYWORDS = [(kw, cat) for cat, kws in SENSITIVE_KEYWORDS.items() for kw in kws]

# ── Semantic content scoring (TF-IDF cosine similarity) ─────────────────────
# The keyword dictionary above can't tell "I visited my parents last year"
# (past, low operational exposure) from "I am flying home tomorrow" (near-term,
# specific, real exposure) — same keyword family, very different risk. This
# runs alongside the keyword dictionary, not instead of it, using TF-IDF
# vectorization + cosine similarity (scikit-learn, no neural model, no API
# call, no labeled dataset) to catch near-term/specific disclosures the
# keyword list misses.
SEMANTIC_EXEMPLARS: dict[str, list[str]] = {
    'travel': [
        'I am flying to Tokyo tomorrow',
        'Landing in San Francisco this afternoon',
        'On my way to the airport right now',
        'Heading to New York next week for a conference',
        'Just booked my flight for Friday',
    ],
    'home': [
        'This is my house in the photo',
        'Just moved into my new apartment downtown',
        'Working from my house today',
        'My home address is',
    ],
    'family': [
        'My wife and kids are visiting this weekend',
        'My daughter just started school nearby',
        'Family dinner with my husband tonight',
        "My son's birthday party is this weekend",
    ],
    'finance': [
        'We just closed our funding round',
        'My salary this year increased significantly',
        'I just bought a new house',
        'My net worth this quarter was',
    ],
    'schedule': [
        'I go to the gym every morning at 6am',
        'My daily routine starts with coffee at this cafe',
        'Every Monday I work from this coworking space',
    ],
}
# TF-IDF cosine similarities between short texts are generally lower than
# dense-embedding similarities, so the threshold is calibrated accordingly.
# With bigram features and sublinear TF scaling, 0.25 surfaces real
# disclosures the keyword dictionary misses while filtering noise. This is
# deliberately tuned for recall over precision: this is a *supplementary*
# signal whose evidence string always carries the similarity score, so a
# human reviewing the itemized breakdown can judge each hit rather than
# trusting it blindly.
SEMANTIC_SIMILARITY_THRESHOLD = 0.25

_URL_RE = re.compile(r'https?://\S+')
_RT_PREFIX_RE = re.compile(r'^RT @\w+:\s*')
_MENTION_RE = re.compile(r'@\w+')


def _clean_for_matching(text: str) -> str:
    """Strips retweet prefixes, URLs and @mentions before similarity matching.
    These carry no content signal but dilute TF-IDF cosine similarity —
    removing them improves match quality measurably."""
    text = _RT_PREFIX_RE.sub('', text)
    text = _URL_RE.sub('', text)
    text = _MENTION_RE.sub('', text)
    return ' '.join(text.split())

_exemplar_texts: list[str] = []
_exemplar_categories: list[str] = []


def _build_exemplar_lists() -> None:
    global _exemplar_texts, _exemplar_categories
    if _exemplar_texts:
        return
    for cat, sentences in SEMANTIC_EXEMPLARS.items():
        for s in sentences:
            _exemplar_texts.append(s)
            _exemplar_categories.append(cat)


def semantic_content_hits(texts: list[str]) -> tuple[int, list[dict]]:
    """
    Returns (hit_count, [{category, text, similarity}, ...]). Vectorizes each
    text and each exemplar sentence with TF-IDF (bigrams, sublinear TF), then
    scores every text against its single closest exemplar by cosine similarity
    — no neural model, no training, no labels, just scikit-learn's
    TfidfVectorizer over a fixed set of hand-written example sentences per
    category.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Keep the original text for evidence/reporting, clean the version used for matching.
    pairs = [(t, _clean_for_matching(t)) for t in texts if t and t.strip()]
    pairs = [(orig, cleaned) for orig, cleaned in pairs if cleaned]
    if not pairs:
        return 0, []

    _build_exemplar_lists()

    cleaned_texts = [c for _, c in pairs]

    # Fit on exemplars + input together so IDF weights reflect the full corpus.
    # Bigrams capture short phrases ("flying to", "my house") that unigrams miss.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    all_texts = _exemplar_texts + cleaned_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    n_exemplars = len(_exemplar_texts)
    sims = cosine_similarity(tfidf_matrix[n_exemplars:], tfidf_matrix[:n_exemplars])

    hits: list[dict] = []
    seen: set[str] = set()
    for i, (original, _cleaned) in enumerate(pairs):
        best_j = int(sims[i].argmax())
        best_score = float(sims[i][best_j])
        if best_score >= SEMANTIC_SIMILARITY_THRESHOLD and original not in seen:
            hits.append({
                'category': _exemplar_categories[best_j],
                'text': original[:120],
                'similarity': round(best_score, 3),
            })
            seen.add(original)
    return len(hits), hits

LOCATION_LABELS = {'GPE', 'LOC'}
ORG_LABELS = {'ORG'}


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Returns [(label, entity_text), ...] via spaCy NER."""
    if not text or not text.strip():
        return []
    doc = get_nlp()(text[:2000])
    return [(ent.label_, ent.text) for ent in doc.ents]


def count_sensitive_keyword_hits(texts: list[str]) -> tuple[int, list[tuple[str, str]]]:
    """Returns (hit_count, [(keyword, source_text_snippet), ...])."""
    hits: list[tuple[str, str]] = []
    for text in texts:
        low = (text or '').lower()
        for kw, _cat in _ALL_KEYWORDS:
            if kw in low:
                hits.append((kw, text[:120]))
    return len(hits), hits


def posting_hour_histogram(dates: list[str]) -> list[int]:
    counts = [0] * 24
    for d in dates:
        if not d:
            continue
        try:
            # Handles 'YYYY-MM-DDTHH:MM:SS' and 'YYYY-MM-DD HH:MM:SS' without
            # pulling in a full ISO/timezone parser — this dataset has neither
            # timezone suffixes nor exotic formats.
            hour = int(d[11:13])
            if 0 <= hour <= 23:
                counts[hour] += 1
        except (ValueError, IndexError):
            continue
    return counts


def _safe_json(raw: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


# ── Twitter ──────────────────────────────────────────────────────────────────

def extract_twitter_features(tweets: list[dict], profile_meta: dict) -> tuple[dict, list[dict]]:
    """
    tweets: list of {text, created_at, ...}
    profile_meta: unpacked people_profile_json for this person (organization,
    bio, email, linkedin_url, followers_count, following_count).
    """
    from .risk_engine import posting_entropy as _entropy

    texts = [t.get('text', '') for t in tweets if t.get('text')]
    dates = [t.get('created_at', '') for t in tweets]

    location_hits = 0
    inference_rows: list[dict] = []
    seen_entities: set[tuple[str, str]] = set()
    for t in tweets[:60]:  # cap NER cost per profile
        text = t.get('text', '')
        for label, ent_text in extract_entities(text):
            key = (label, ent_text.lower())
            if label in LOCATION_LABELS:
                location_hits += 1
            if key in seen_entities:
                continue
            seen_entities.add(key)
            if label in LOCATION_LABELS:
                inference_rows.append({
                    'targetInfo': 'Location', 'inferredValue': ent_text,
                    'evidence': f'"{text[:100]}" (spaCy GPE/LOC)',
                })
            elif label in ORG_LABELS:
                inference_rows.append({
                    'targetInfo': 'Organization mention', 'inferredValue': ent_text,
                    'evidence': f'"{text[:100]}" (spaCy ORG)',
                })

    kw_count, kw_hits = count_sensitive_keyword_hits(texts)
    for kw, snippet in kw_hits[:5]:
        inference_rows.append({
            'targetInfo': 'Sensitive topic', 'inferredValue': kw,
            'evidence': f'"{snippet}" (keyword match)',
        })

    sem_count, sem_hits = semantic_content_hits(texts)
    for hit in sem_hits[:5]:
        inference_rows.append({
            'targetInfo': f'Sensitive topic ({hit["category"]}, semantic)', 'inferredValue': hit['text'],
            'evidence': f'similarity={hit["similarity"]} to a near-term "{hit["category"]}" disclosure example',
        })

    # Union by matched text, not sum — a tweet flagged by both keyword and
    # semantic similarity is one real signal, not two.
    flagged_texts = {snippet for _, snippet in kw_hits} | {h['text'] for h in sem_hits}
    combined_hit_count = len(flagged_texts)

    entropy = _entropy(posting_hour_histogram(dates))

    features = {
        'has_email': bool(profile_meta.get('email')),
        'has_linkedin': bool(profile_meta.get('linkedin_url')),
        'has_org': bool(profile_meta.get('organization')),
        'location_entity_count': location_hits,
        'posting_entropy': entropy,
        'sensitive_keyword_hits': combined_hit_count,
        'keyword_only_hits': kw_count,  # kept for before/after comparison, not scored separately
        'semantic_only_hits': sem_count,
        'followers_count': int(profile_meta.get('followers_count') or 0),
        'following_count': int(profile_meta.get('following_count') or 0),
        'total_tweets': len(tweets),
    }
    return features, inference_rows


def unpack_twitter_profile_meta(people_profile_json: Any) -> dict:
    data = _safe_json(people_profile_json) or {}
    return {
        'organization': data.get('organization') or '',
        'bio': data.get('bio') or '',
        'email': data.get('email'),
        'linkedin_url': data.get('linkedin_url'),
        'followers_count': data.get('followers_count') or 0,
        'following_count': data.get('following_count') or 0,
    }


# ── LinkedIn ─────────────────────────────────────────────────────────────────

def extract_linkedin_features(row: dict) -> tuple[dict, list[dict]]:
    """row: one raw CSV row dict from linkedin_export_1000.csv."""
    inference_rows: list[dict] = []

    city = (row.get('location_city') or '').strip()
    state = (row.get('location_state') or '').strip()
    country = (row.get('location_country') or '').strip()
    if city:
        granularity, loc_val = 3, city
    elif state:
        granularity, loc_val = 2, state
    elif country:
        granularity, loc_val = 1, country
    else:
        granularity, loc_val = 0, ''
    if loc_val:
        inference_rows.append({
            'targetInfo': 'Location', 'inferredValue': row.get('location_text') or loc_val,
            'evidence': 'Disclosed directly in profile (location_city/state/country)',
        })

    experiences = _safe_json(row.get('experiences_json')) or []
    current_employer = ''
    if isinstance(experiences, list) and experiences:
        first = experiences[0]
        if isinstance(first, dict):
            current_employer = first.get('company_name') or ''
        if current_employer:
            inference_rows.append({
                'targetInfo': 'Current employer', 'inferredValue': current_employer,
                'evidence': 'experiences_json[0].company_name',
            })
    employer_count = len({
        e.get('company_name') for e in experiences
        if isinstance(e, dict) and e.get('company_name')
    }) if isinstance(experiences, list) else 0

    educations = _safe_json(row.get('educations_json')) or []
    education_disclosed = bool(educations)
    if education_disclosed and isinstance(educations, list) and educations:
        school = educations[0].get('school_name') if isinstance(educations[0], dict) else None
        if school:
            inference_rows.append({
                'targetInfo': 'Education', 'inferredValue': school,
                'evidence': 'educations_json[0].school_name',
            })

    skills = _safe_json(row.get('profile_skills_json')) or []
    skill_count = len(skills) if isinstance(skills, list) else 0

    headline = row.get('headline') or ''
    about = row.get('about') or ''
    bio_entities = extract_entities(f'{headline}. {about}')
    bio_entity_count = len({(l, t.lower()) for l, t in bio_entities})
    for label, ent_text in bio_entities[:5]:
        if label in ORG_LABELS or label in LOCATION_LABELS:
            inference_rows.append({
                'targetInfo': f'{"Organization" if label in ORG_LABELS else "Location"} mention (headline/about)',
                'inferredValue': ent_text,
                'evidence': f'spaCy {label} in headline/about',
            })

    def _bool(v: Any) -> bool:
        return str(v).strip().lower() in {'1', 'true', 'yes'}

    # How much real career history we actually have, independent of what it says.
    # A profile with one listed employer because they genuinely only had one job
    # looks identical, field-wise, to one where the export just didn't capture
    # more — this can't distinguish those cases, but it can at least avoid
    # treating a near-empty profile with the same certainty as a fully detailed one.
    field_completeness = round(sum([
        employer_count > 0,
        education_disclosed,
        skill_count > 0,
    ]) / 3, 2)

    features = {
        'location_disclosed': granularity,
        'current_employer': current_employer,
        'employer_count': employer_count,
        'education_disclosed': education_disclosed,
        'bio_entity_count': bio_entity_count,
        'field_completeness': field_completeness,
        'open_to_work': _bool(row.get('open_to_work')),
        'is_hiring': _bool(row.get('is_hiring')),
        'connections_count': int(row.get('connections_count') or 0),
        'follower_count': int(row.get('follower_count') or 0),
        'skill_count': skill_count,
    }
    return features, inference_rows
