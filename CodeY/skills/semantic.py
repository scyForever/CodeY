"""Dense-vector semantic routing over the discoverable Skill catalog."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass

from ..providers.embeddings import normalize_embedding_batch

DEFAULT_SEMANTIC_MIN_SIMILARITY = 0.55
DEFAULT_SEMANTIC_MIN_MARGIN = 0.05
DEFAULT_EMBEDDING_BATCH_SIZE = 128


@dataclass(frozen=True)
class SemanticSkillScore:
    skill_name: str
    score: float
    rank: int

    def to_dict(self):
        return {
            "skill_name": self.skill_name,
            "score": self.score,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class SemanticSkillMatch:
    skill_name: str = ""
    score: float = 0.0
    runner_up_score: float | None = None
    margin: float | None = None
    accepted: bool = False
    status: str = "not_run"
    reason: str = ""
    model: str = ""
    index_fingerprint: str = ""
    dimensions: int = 0
    index_rebuilt: bool = False
    scores: tuple[SemanticSkillScore, ...] = ()

    def to_dict(self):
        return {
            "skill_name": self.skill_name,
            "score": self.score,
            "runner_up_score": self.runner_up_score,
            "margin": self.margin,
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "model": self.model,
            "index_fingerprint": self.index_fingerprint,
            "dimensions": self.dimensions,
            "index_rebuilt": self.index_rebuilt,
            "scores": [score.to_dict() for score in self.scores],
        }


class SkillSemanticIndex:
    """A lazily built exact-cosine index for a workspace's Skill descriptions."""

    def __init__(
        self,
        embedding_client,
        *,
        min_similarity=DEFAULT_SEMANTIC_MIN_SIMILARITY,
        min_margin=DEFAULT_SEMANTIC_MIN_MARGIN,
        batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
    ):
        if embedding_client is None:
            raise ValueError("embedding_client is required")
        self.embedding_client = embedding_client
        self.min_similarity = float(min_similarity)
        self.min_margin = float(min_margin)
        self.batch_size = int(batch_size)
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("semantic minimum similarity must be between 0 and 1")
        if not 0.0 <= self.min_margin <= 1.0:
            raise ValueError("semantic minimum margin must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("embedding batch size must be at least 1")

        self._lock = threading.Lock()
        self._fingerprint = ""
        self._vectors = ()
        self._model = ""
        self._dimensions = 0

    def select(self, request, skills, *, excluded_skill_names=()):
        request_text = str(request or "").strip()
        if not request_text:
            raise ValueError("semantic routing request must not be empty")
        skill_items = tuple(skills)
        if not skill_items:
            return SemanticSkillMatch(
                status="no_skills",
                reason="No Skills are available for semantic routing.",
            )

        fingerprint, vectors, model, dimensions, rebuilt = self._ensure_index(skill_items)
        query_batch = normalize_embedding_batch(
            self.embedding_client.embed((request_text,)),
            expected_count=1,
        )
        if query_batch.model != model:
            raise ValueError("embedding model changed between Skill indexing and query")
        query = _unit_vector(query_batch.vectors[0])
        if len(query) != dimensions:
            raise ValueError("query embedding dimension does not match the Skill index")

        excluded = {str(name).strip().casefold() for name in excluded_skill_names}
        ranked = sorted(
            (
                (skill.name, max(-1.0, min(1.0, _dot(query, vector))))
                for skill, vector in zip(skill_items, vectors, strict=True)
                if skill.name.casefold() not in excluded
            ),
            key=lambda item: (-item[1], item[0].casefold()),
        )
        scores = tuple(
            SemanticSkillScore(skill_name=name, score=score, rank=index + 1)
            for index, (name, score) in enumerate(ranked)
        )
        if not ranked:
            return SemanticSkillMatch(
                status="near_miss_excluded",
                reason="All semantic candidates were vetoed by explicit near-miss rules.",
                model=model,
                index_fingerprint=fingerprint,
                dimensions=dimensions,
                index_rebuilt=rebuilt,
                scores=scores,
            )

        skill_name, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else None
        margin = top_score - runner_up if runner_up is not None else None
        if top_score < self.min_similarity:
            status = "below_similarity_threshold"
            accepted = False
            reason = (
                f"Top cosine similarity {top_score:.4f} is below the configured "
                f"threshold {self.min_similarity:.4f}."
            )
        elif margin is not None and margin < self.min_margin:
            status = "ambiguous_margin"
            accepted = False
            reason = (
                f"Top-two cosine margin {margin:.4f} is below the configured "
                f"margin {self.min_margin:.4f}."
            )
        else:
            status = "accepted"
            accepted = True
            if margin is None:
                reason = f"Semantic route accepted at cosine similarity {top_score:.4f}."
            else:
                reason = (
                    f"Semantic route accepted at cosine similarity {top_score:.4f} "
                    f"with top-two margin {margin:.4f}."
                )

        return SemanticSkillMatch(
            skill_name=skill_name,
            score=top_score,
            runner_up_score=runner_up,
            margin=margin,
            accepted=accepted,
            status=status,
            reason=reason,
            model=model,
            index_fingerprint=fingerprint,
            dimensions=dimensions,
            index_rebuilt=rebuilt,
            scores=scores,
        )

    def _ensure_index(self, skills):
        documents = tuple(_skill_semantic_text(skill) for skill in skills)
        identity = str(
            getattr(
                self.embedding_client,
                "identity",
                self.embedding_client.__class__.__qualname__,
            )
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {"embedding_identity": identity, "documents": documents},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self._lock:
            if fingerprint == self._fingerprint:
                return (
                    fingerprint,
                    self._vectors,
                    self._model,
                    self._dimensions,
                    False,
                )

            vectors = []
            model = ""
            dimensions = 0
            for offset in range(0, len(documents), self.batch_size):
                batch = normalize_embedding_batch(
                    self.embedding_client.embed(documents[offset : offset + self.batch_size]),
                    expected_count=min(self.batch_size, len(documents) - offset),
                )
                if model and batch.model != model:
                    raise ValueError("embedding model changed while building the Skill index")
                model = batch.model
                for raw_vector in batch.vectors:
                    vector = _unit_vector(raw_vector)
                    if dimensions and len(vector) != dimensions:
                        raise ValueError("Skill embeddings have inconsistent dimensions")
                    dimensions = len(vector)
                    vectors.append(vector)

            self._fingerprint = fingerprint
            self._vectors = tuple(vectors)
            self._model = model
            self._dimensions = dimensions
            return fingerprint, self._vectors, model, dimensions, True


def _skill_semantic_text(skill):
    positive_description = str(skill.description).split(
        " It should not activate for ", 1
    )[0].strip()
    lines = [
        f"Skill name: {skill.name}",
        f"Purpose: {positive_description}",
        "Activation examples: " + "; ".join(skill.activation_phrases),
    ]
    for route in skill.routes:
        route_parts = [route.label, *route.triggers]
        lines.append("Route intent: " + "; ".join(part for part in route_parts if part))
    return "\n".join(lines)


def _unit_vector(vector):
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding vector must have a finite non-zero norm")
    return tuple(value / norm for value in values)


def _dot(left, right):
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))
