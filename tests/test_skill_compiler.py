from __future__ import annotations

from dataclasses import dataclass

import pytest

from aeloon.plugins.SkillGraph.compiler import SkillCompilerRequest, compile_skill_to_workspace


@dataclass(frozen=True)
class _FakePackage:
    slug: str


@dataclass(frozen=True)
class _FakeApi:
    build_skill_package: object
    compile_skill: object


def test_compile_skill_to_workspace_writes_into_compiled_skills(tmp_path, monkeypatch) -> None:
    skills_dir = tmp_path / "skills" / "demo"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Demo", encoding="utf-8")

    def _fake_build_skill_package(skill_path):
        assert skill_path == skills_dir.resolve()
        return _FakePackage(slug="demo")

    def _fake_compile_skill(**kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            'SKILL_META = {"name": "demo", "description": "Demo workflow", "global_inputs": []}\n'
            "class _Graph:\n"
            "    def invoke(self, state):\n"
            "        return state\n"
            "def build_graph():\n"
            "    return _Graph()\n",
            encoding="utf-8",
        )
        output_path.with_suffix(".manifest.json").write_text(
            '{"dependencies": []}', encoding="utf-8"
        )
        sandbox = output_path.with_suffix(".sandbox")
        (sandbox / "skill").mkdir(parents=True, exist_ok=True)
        (sandbox / "bootstrap.json").write_text(
            '{"status": "ready", "checks": [], "env": {}}', encoding="utf-8"
        )
        (sandbox / "env.json").write_text("{}", encoding="utf-8")
        config_path = output_path.parent / "skill_config.json"
        config_path.write_text(
            '{"runtime": {"api_key": "", "base_url": "https://example.com", "model": "runtime-model"}}',
            encoding="utf-8",
        )
        kwargs["report_path"].write_text('{"ok": true}', encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "aeloon.plugins.SkillGraph.compiler._load_skillgraph_api",
        lambda: _FakeApi(_fake_build_skill_package, _fake_compile_skill),
    )

    provider = type("Provider", (), {"api_key": "key", "api_base": "https://example.com"})()
    result = compile_skill_to_workspace(
        workspace=tmp_path,
        provider=provider,
        default_model="default-model",
        request=SkillCompilerRequest(skill_path="skills/demo", runtime_model="runtime-model"),
    )

    assert result.workflow_name == "demo"
    assert result.output_path == tmp_path / "compiled_skills" / "demo_workflow.py"
    assert result.manifest_path.exists()
    assert result.sandbox_path.exists()
    assert result.report_path.exists()
    assert result.config_path.exists()


def test_compile_skill_to_workspace_raises_when_skillgraph_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aeloon.plugins.SkillGraph.compiler._load_skillgraph_api",
        lambda: (_ for _ in ()).throw(RuntimeError("skillgraph is not available")),
    )

    provider = type("Provider", (), {"api_key": "key", "api_base": "https://example.com"})()
    with pytest.raises(RuntimeError, match="skillgraph is not available"):
        compile_skill_to_workspace(
            workspace=tmp_path,
            provider=provider,
            default_model="default-model",
            request=SkillCompilerRequest(skill_path="skills/demo"),
        )
