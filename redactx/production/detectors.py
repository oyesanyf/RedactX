"""
Production detectors.

  RedactXDetector  - the fine-tuned VaultGemma engine, with long-document windowing and batched scoring.
  PresidioDetector - Microsoft Presidio (presidio_analyzer + spaCy), the rule/NER baseline.
  HybridDetector   - union of both: a character is flagged if EITHER detector flags it. Two detectors with
                     different failure modes give higher recall than either alone, which is the right
                     trade-off for redaction (a miss is a leak; a false positive is an over-redaction).

All detectors return `Detection` objects with findings in ORIGINAL-text character offsets.
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from redactx.production.chunking import Window, chunk_text
from redactx.production.hipaa import Category, is_hipaa_identifier, to_category


@dataclass
class Finding:
    start: int
    end: int
    text: str
    category: Category
    score: float
    sources: Tuple[str, ...]

    @property
    def is_hipaa_identifier(self) -> bool:
        return is_hipaa_identifier(self.category)

    def to_dict(self, include_text: bool = True) -> Dict:
        d = {"start": self.start, "end": self.end, "category": self.category.value,
             "hipaa_identifier": self.is_hipaa_identifier, "score": round(float(self.score), 4),
             "sources": list(self.sources)}
        if include_text:
            d["text"] = self.text
        return d


@dataclass
class Detection:
    contains_phi: bool
    findings: List[Finding]
    doc_score: Optional[float] = None            # RedactX P(PHI), max over windows; None if not computed
    unlocalized_phi: bool = False                # model says PHI but no span could be localized
    windows: int = 1
    detectors: List[str] = field(default_factory=list)


# Prefer the more specific category when findings from several detectors overlap.
_SOURCE_PRIORITY = {"presidio": 0, "redactx": 1}


def merge_findings(text: str, findings: Sequence[Finding]) -> List[Finding]:
    """Union of overlapping/adjacent findings. Keeps max score, all sources, most specific category."""
    ordered = sorted(findings, key=lambda f: (f.start, -f.end))
    groups: List[List[Finding]] = []
    for f in ordered:
        if f.end <= f.start:
            continue
        if groups and f.start <= max(g.end for g in groups[-1]):
            groups[-1].append(f)
        else:
            groups.append([f])
    merged: List[Finding] = []
    for g in groups:
        s, e = min(x.start for x in g), max(x.end for x in g)
        best = sorted(g, key=lambda x: (_SOURCE_PRIORITY.get(x.sources[0], 9),
                                        x.category == Category.UNKNOWN, -(x.end - x.start)))[0]
        sources = tuple(sorted({src for x in g for src in x.sources}))
        merged.append(Finding(s, e, text[s:e], best.category, max(x.score for x in g), sources))
    return merged


class RedactXDetector:
    name = "redactx"

    def __init__(self, engine, max_chars: int = 600, overlap: int = 150, batch_size: int = 8):
        self.engine = engine
        self.max_chars = max_chars
        self.overlap = overlap
        self.batch_size = batch_size
        self._lock = threading.Lock()  # one inference at a time per model instance (GPU memory)

    @property
    def has_span_head(self) -> bool:
        return getattr(self.engine, "span_locator", None) is not None

    @property
    def doc_threshold(self) -> float:
        return float(getattr(self.engine, "doc_threshold", 0.5))

    def detect_many(self, texts: Sequence[str]) -> List[Detection]:
        plan: List[Tuple[int, Window]] = []
        for di, t in enumerate(texts):
            for w in chunk_text(t, self.max_chars, self.overlap):
                plan.append((di, w))
        window_texts = [w.slice(texts[di]) for di, w in plan]
        with self._lock:
            scored = self.engine.score_texts(window_texts, batch_size=self.batch_size) if window_texts else []

        per_doc_scores: Dict[int, List[float]] = {i: [] for i in range(len(texts))}
        per_doc_findings: Dict[int, List[Finding]] = {i: [] for i in range(len(texts))}
        for (di, w), res in zip(plan, scored):
            per_doc_scores[di].append(res["p_phi"])
            for sp in res["spans"]:
                s, e = w.start + sp.start, w.start + sp.end
                per_doc_findings[di].append(Finding(s, e, texts[di][s:e], to_category(sp.category),
                                                    float(sp.confidence), (self.name,)))
        out: List[Detection] = []
        for di, t in enumerate(texts):
            findings = merge_findings(t, per_doc_findings[di])
            doc_score = max(per_doc_scores[di]) if per_doc_scores[di] else 0.0
            flagged = doc_score >= self.doc_threshold
            out.append(Detection(
                contains_phi=flagged or bool(findings),
                findings=findings,
                doc_score=round(doc_score, 6),
                unlocalized_phi=flagged and not findings,
                windows=sum(1 for d, _ in plan if d == di),
                detectors=[self.name],
            ))
        return out

    def detect(self, text: str) -> Detection:
        return self.detect_many([text])[0]


class PresidioDetector:
    name = "presidio"

    def __init__(self, analyzer=None, score_threshold: float = 0.35, language: str = "en",
                 entities: Optional[List[str]] = None):
        if analyzer is None:
            from presidio_analyzer import AnalyzerEngine  # spaCy en_core_web_lg by default
            analyzer = AnalyzerEngine()
        self.analyzer = analyzer
        self.score_threshold = score_threshold
        self.language = language
        self.entities = entities

    def detect(self, text: str) -> Detection:
        results = self.analyzer.analyze(text=text, language=self.language, entities=self.entities,
                                        score_threshold=self.score_threshold)
        findings = merge_findings(text, [
            Finding(r.start, r.end, text[r.start:r.end], to_category(r.entity_type), float(r.score), (self.name,))
            for r in results
        ])
        return Detection(contains_phi=bool(findings), findings=findings, detectors=[self.name])

    def detect_many(self, texts: Sequence[str]) -> List[Detection]:
        return [self.detect(t) for t in texts]


class HybridDetector:
    name = "hybrid"

    def __init__(self, redactx: RedactXDetector, presidio: PresidioDetector):
        self.redactx = redactx
        self.presidio = presidio

    def detect_many(self, texts: Sequence[str]) -> List[Detection]:
        rx = self.redactx.detect_many(texts)
        out: List[Detection] = []
        for t, r in zip(texts, rx):
            p = self.presidio.detect(t)
            findings = merge_findings(t, list(r.findings) + list(p.findings))
            out.append(Detection(
                contains_phi=r.contains_phi or p.contains_phi,
                findings=findings,
                doc_score=r.doc_score,
                unlocalized_phi=r.unlocalized_phi,
                windows=r.windows,
                detectors=[self.redactx.name, self.presidio.name],
            ))
        return out

    def detect(self, text: str) -> Detection:
        return self.detect_many([text])[0]
