import json

import pytest

from CodeY import CodeYAgent, EmbeddingBatch, FakeModelClient, SessionStore, WorkspaceContext


class RecordingEmbeddingClient:
    model = "recording-embedding"
    identity = "recording-embedding:v1"

    def __init__(self, query_vectors=None, fail_queries=False):
        self.query_vectors = dict(query_vectors or {})
        self.fail_queries = fail_queries
        self.calls = []

    def embed(self, texts):
        texts = tuple(texts)
        self.calls.append(texts)
        vectors = []
        for text in texts:
            if text.startswith("Skill name: data-systems"):
                vectors.append((1.0, 0.0))
            elif text.startswith("Skill name: security-systems"):
                vectors.append((0.0, 1.0))
            elif self.fail_queries:
                raise RuntimeError("embedding backend unavailable")
            else:
                vectors.append(self.query_vectors[text])
        return EmbeddingBatch(
            vectors=tuple(vectors),
            model=self.model,
            metadata={},
        )


def write_skill(root, name, purpose, phrases, near_misses, route_triggers):
    skill = root / "skills" / name
    (skill / "rules").mkdir(parents=True)
    (skill / "rules" / "core.md").write_text(f"{name} core\n", encoding="utf-8")
    description = (
        f"This skill should be used when the user's primary objective is {purpose} and the "
        f'request is best characterized as "{phrases[0]}" or "{phrases[1]}". '
        f"It should not activate for {near_misses[0]}, {near_misses[1]}, or {near_misses[2]}."
    )
    routes = {
        "tasks": [
            {
                "id": "primary",
                "label": f"{name} primary",
                "triggers": list(route_triggers),
                "reads": [],
            },
            {"id": "other", "label": "Other", "triggers": [], "reads": []},
        ]
    }
    (skill / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                "<always-applicable>",
                "Always Read:",
                "- rules/core.md",
                "</always-applicable>",
                "<task-routing>",
                "```json",
                json.dumps(routes, ensure_ascii=False, separators=(",", ":")),
                "```",
                "</task-routing>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_agent(root, embedder, selector_outputs=(), **kwargs):
    model = FakeModelClient(selector_outputs)
    agent = CodeYAgent(
        model_client=model,
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".codey" / "sessions"),
        approval_policy="auto",
        skill_embedding_client=embedder,
        **kwargs,
    )
    return agent, model


@pytest.fixture
def skill_workspace(tmp_path):
    write_skill(
        tmp_path,
        "data-systems",
        "repairing data pipelines and storage workflows",
        ("database pipeline repair", "storage workflow diagnosis"),
        ("generic database question", "unrelated UI styling", "product roadmap planning"),
        ("pipeline", "storage"),
    )
    write_skill(
        tmp_path,
        "security-systems",
        "hardening access controls and investigating security failures",
        ("access control hardening", "security incident diagnosis"),
        ("generic login question", "unrelated data cleanup", "visual design review"),
        ("access", "security"),
    )
    return tmp_path


def test_semantic_vector_selects_skill_and_reuses_index(skill_workspace):
    embedder = RecordingEmbeddingClient(
        {
            "repair broken ETL rows": (1.0, 0.0),
            "investigate suspicious permissions": (0.0, 1.0),
        }
    )
    agent, selector_model = build_agent(skill_workspace, embedder)

    first = agent.route_task("repair broken ETL rows", run_id="run-1", task_id="task-1")
    second = agent.route_task(
        "investigate suspicious permissions",
        run_id="run-2",
        task_id="task-2",
    )

    assert first.skill_name == "data-systems"
    assert first.selection_source == "semantic_vector"
    assert second.skill_name == "security-systems"
    assert second.selection_source == "semantic_vector"
    assert len(embedder.calls) == 3
    assert len(embedder.calls[0]) == 2
    assert embedder.calls[1] == ("repair broken ETL rows",)
    assert embedder.calls[2] == ("investigate suspicious permissions",)
    assert "generic database question" not in embedder.calls[0][0]
    assert selector_model.prompts == []
    status = agent.route_status()["semantic_route"]
    assert status["status"] == "accepted"
    assert status["index_rebuilt"] is False
    assert status["model"] == "recording-embedding"
    assert len(status["index_fingerprint"]) == 64


def test_ambiguous_vector_route_falls_back_to_description_model(skill_workspace):
    embedder = RecordingEmbeddingClient({"uncertain maintenance request": (1.0, 1.0)})
    selector = json.dumps(
        {
            "skill_name": "security-systems",
            "confidence": 0.91,
            "reason": "The request is about access-control maintenance.",
        }
    )
    agent, selector_model = build_agent(skill_workspace, embedder, [selector])

    route = agent.route_task(
        "uncertain maintenance request",
        run_id="run-ambiguous",
        task_id="task-ambiguous",
    )

    assert route.skill_name == "security-systems"
    assert route.selection_source == "description_model"
    assert "Description-model fallback" in route.selector_reason
    assert agent.last_semantic_skill_routing["status"] == "ambiguous_margin"
    assert len(selector_model.prompts) == 1


def test_near_miss_is_a_hard_veto_for_vector_and_model(skill_workspace):
    embedder = RecordingEmbeddingClient({"generic database question": (1.0, 0.0)})
    selector = json.dumps(
        {
            "skill_name": "data-systems",
            "confidence": 0.99,
            "reason": "Select data systems.",
        }
    )
    agent, _ = build_agent(skill_workspace, embedder, [selector])

    route = agent.route_task(
        "generic database question",
        run_id="run-near-miss",
        task_id="task-near-miss",
    )

    assert route.skill_name == ""
    assert route.selection_source == "description_model_rejected"
    assert agent.last_semantic_skill_routing["status"] == "below_similarity_threshold"
    assert [item["skill_name"] for item in agent.last_semantic_skill_routing["scores"]] == [
        "security-systems"
    ]


def test_embedding_failure_falls_back_without_exposing_error_text(skill_workspace):
    embedder = RecordingEmbeddingClient(fail_queries=True)
    selector = json.dumps(
        {
            "skill_name": "",
            "confidence": 0.8,
            "reason": "No Skill applies.",
        }
    )
    agent, _ = build_agent(skill_workspace, embedder, [selector])

    route = agent.route_task(
        "unclassified operation",
        run_id="run-error",
        task_id="task-error",
    )

    assert route.skill_name == ""
    assert route.selection_source == "description_model"
    assert agent.last_semantic_skill_routing == {
        "status": "error",
        "enabled": True,
        "accepted": False,
        "error_type": "RuntimeError",
    }


def test_explicit_skill_invocation_bypasses_embeddings(skill_workspace):
    embedder = RecordingEmbeddingClient()
    agent, _ = build_agent(skill_workspace, embedder)

    route = agent.route_task(
        "/security-systems inspect this",
        run_id="run-explicit",
        task_id="task-explicit",
    )

    assert route.skill_name == "security-systems"
    assert route.selection_source == "explicit"
    assert embedder.calls == []
    assert agent.last_semantic_skill_routing == {
        "status": "not_invoked",
        "enabled": True,
    }


def test_exclusive_description_phrase_bypasses_embeddings(skill_workspace):
    embedder = RecordingEmbeddingClient()
    agent, _ = build_agent(skill_workspace, embedder)

    route = agent.route_task(
        "please perform database pipeline repair",
        run_id="run-phrase",
        task_id="task-phrase",
    )

    assert route.skill_name == "data-systems"
    assert route.selection_source == "description_phrase"
    assert embedder.calls == []
    assert agent.last_semantic_skill_routing["status"] == "not_invoked"


def test_configured_semantic_threshold_is_authoritative(skill_workspace):
    embedder = RecordingEmbeddingClient({"weak but explicit semantic signal": (0.4, -0.916515)})
    agent, _ = build_agent(
        skill_workspace,
        embedder,
        skill_semantic_min_similarity=0.35,
    )

    route = agent.route_task(
        "weak but explicit semantic signal",
        run_id="run-low-threshold",
        task_id="task-low-threshold",
    )

    assert route.skill_name == "data-systems"
    assert route.selection_source == "semantic_vector"
    assert route.selector_confidence == pytest.approx(0.4, abs=1e-6)
