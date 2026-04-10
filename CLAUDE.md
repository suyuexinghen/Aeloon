# Aeloon — Agent Development Guide

This file is loaded automatically by Claude Code. Read it before making any changes.

---

## Code Style

### Tooling

| Tool | Command | Purpose |
|------|---------|---------|
| ruff | `ruff check --fix .` | Lint + auto-fix |
| ruff | `ruff format .` | Format |
| pytest | `pytest` | Run tests |

**Always run both before committing.** CI enforces `ruff==0.15.6` — use the same version locally.

### Key Rules

- Line length: 100 (formatter enforced, E501 silenced)
- Imports: sorted by ruff (isort-compatible)
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` module-level constants only
- Type hints: required on all public functions and class methods
- Async: prefer `async/await` for all I/O; keep sync wrappers thin
- Never rename camelCase args that mirror external API contracts — add `# noqa: N803` instead

### Commit Format (Conventional Commits)

```
<type>(<scope>): <short summary>
```

Types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `perf`, `ci`

```
feat(channels): add Discord channel support
fix(cron): prevent duplicate job on restart
chore: bump ruff to 0.15.7
```

---

## Project Layout

```
aeloon/
├── agent/          # Core agent loop, context builder, memory, subagents
│   └── tools/      # Tool implementations (filesystem, web, exec, cron, mcp)
├── channels/       # One file per platform (slack.py, telegram.py, ...)
├── cli/            # Typer CLI commands — thin wrappers only
├── config/         # Pydantic schema + loader + path helpers
├── providers/      # LLM provider abstractions + registry
├── skills/         # Built-in skill directories (each has SKILL.md)
└── templates/      # Runtime templates (SOUL.md, AGENTS.md, USER.md, ...)
bridge/             # WhatsApp bridge (Node.js / TypeScript — separate process)
tests/              # pytest test suite
docs/               # Team documentation
```

---

## Extension Patterns

### Add a Channel

```python
# aeloon/channels/myplatform.py
from aeloon.channels.base import BaseChannel

class MyPlatformChannel(BaseChannel):
    name = "myplatform"
    display_name = "My Platform"

    async def start(self) -> None:
        # connect and listen; call self._handle_message() on each event
        ...

    async def stop(self) -> None:
        ...

    async def send(self, msg) -> None:
        ...
```

Auto-discovered — no registration needed. Drop the file in `aeloon/channels/` and it works.

### Add an LLM Provider

```python
# aeloon/providers/myprovider.py
from aeloon.providers.base import LLMProvider, LLMResponse

class MyProvider(LLMProvider):
    async def chat(self, messages, tools=None, model=None, **kwargs) -> LLMResponse:
        ...
        return LLMResponse(content=text, tool_calls=[], finish_reason="stop")

    def get_default_model(self) -> str:
        return "my-model-v1"
```

Then add a `ProviderSpec` entry in `aeloon/providers/registry.py`.

### Add a Tool

```python
# aeloon/agent/tools/mytool.py
from aeloon.agent.tools.base import Tool

class MyTool(Tool):
    @property
    def name(self) -> str: return "my_tool"

    @property
    def description(self) -> str: return "Does something useful."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }

    async def execute(self, input: str, **kwargs) -> str:
        return f"result: {input}"
```

Register it in `AgentLoop._register_default_tools()`.

### Add a Skill

Create `aeloon/skills/myskill/SKILL.md`:

```markdown
---
name: myskill
description: "One-line description"
metadata: {"aeloon": {"emoji": "🔧", "requires": {"bins": ["mytool"]}}}
---

# My Skill

Instructions for the agent on how to use this skill...
```

No code required — skills are markdown instructions loaded into the system prompt.

---

## Testing

```bash
pytest                    # run all tests
pytest tests/test_foo.py  # run specific file
pytest -k "slack"         # run matching tests
```

- `asyncio_mode = "auto"` — all async tests work without decoration
- Mock external services; do not make real API calls in tests
- One test file per module (`tests/test_<module>.py`)

---

## What NOT to Do

- Don't import channel-specific packages at module top-level (use lazy imports)
- Don't add logic to `cli/` — keep commands thin; put logic in `core/` or the relevant module
- Don't hardcode paths — use `config.workspace_path` or `Path.home()`
- Don't silently swallow exceptions in tool `execute()` — return `"Error: ..."` strings
- Don't use `asyncio.run()` inside async code
