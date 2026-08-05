# Adding a new agent to fintel

Step-by-step guide to wire a custom agent onto an **existing strategy package** and run simulations / backtests. 

Companion: [architecture.md](architecture.md) (L5 agents, environment boundary, Harbor-style factory).

---

## 0. Mental model


| Piece                                     | Owns                                                    |
| ----------------------------------------- | ------------------------------------------------------- |
| **Strategy package** (`packages/<name>/`) | Mission, schedule, universe, **which data kinds exist** |
| **Platform** (`simulate/`)                | Cells, PIT clamp, concurrency, nerve/TUI, scoring       |
| **Adapter** (`fintel/agents/…`)           | How *your* agent gets data and returns views            |


Contract:

```text
decide(env: Environment) -> AgentResponse
```

Hard rules:

1. **Never import** `market/` **or** `pit/`**.** Only `env.access` / `env.tools` / `env.evidence`.
2. **Declare** `pit_enforcement`: `"access"` (in-process) or `"cli_deny"` (subprocess CLI).
3. **No symbol fan-out in the adapter.** One cell → decide once. The platform fans out.
4. Return `AgentResponse` with `views` / `outcome` / `usage` / optional `trace`. Do not invent a side channel.

---



## 1. Pick the host


| Host                     | When                                         | Examples                       |
| ------------------------ | -------------------------------------------- | ------------------------------ |
| **In-process**           | Agent is a Python library / LangGraph / desk | `llm`, `optimized`, `scripted` |
| **Subprocess CLI + MCP** | Agent is an external CLI                     | `openclaw`, `claude-code`      |


Most custom research agents are **in-process**. CLI agents subclass the installed host in `fintel/agents/adapters/base.py` and differ mainly in argv / config isolation.

---



## 2. Choose how data arrives (channel)

The strategy binds kinds. You choose the delivery shape:


| Channel                    | Use when                                                    |
| -------------------------- | ----------------------------------------------------------- |
| `env.access.read(kind, …)` | You build your own packs / prompts (OptimizedAgent pattern) |
| `env.tools`                | Tool-loop agent that asks for data                          |
| `env.evidence` **/ pack**  | One-shot LLM over a rendered dossier                        |


Same PIT and audit for all three. For role-split desks (quant vs qual), **bind the full surface in the strategy**, then **partition in the adapter** with selective `access.read` — do not ask the platform for specialist roles.

---



## 3. Check the strategy package

Open `packages/<strategy>/strategy.toml`:

- `[decision].scope` — `single_name` vs portfolio (cells differ; adapter API does not)
- `[[data]]` — every kind your agent needs must be listed (source + lookbacks)
- `mission.md` / `output_schema.json` — passed into the adapter as `mission_text` / `output_schema_text` when the platform builds you

If a kind is missing, **extend the package’s** `[[data]]`, don’t bypass access.

Smoke later with a tiny override: `--universe AAPL,MSFT --dates 2025-01-01`.

---



## 4. Write one adapter file

Convention: **one file per agent** under `fintel/agents/adapters/<name>.py` (fintel hosts / wrappers around external agents) or `fintel/agents/installed/<name>.py` (native in-process agent logic). See `installed/llm_agent.py`, `installed/optimized_agent.py` + `adapters/optimized.py`, `adapters/openclaw.py`.

Minimal skeleton:

```python
from dataclasses import dataclass
from typing import ClassVar

from fintel.agents.pit_policy import PitEnforcement
from fintel.environment import Environment
from fintel.models.decision import AgentResponse, View
from fintel.models.trace import ReasoningTrace, Usage

@dataclass
class MyAgent:
    name: str = "myagent"
    version: str = "0.1.0"
    model: str = "…"
    # Platform injects these from the package:
    mission_text: str = ""
    output_schema_text: str = ""

    pit_enforcement: ClassVar[PitEnforcement] = "access"

    @staticmethod
    def preflight_checks(**params) -> list[str]:
        # e.g. missing API keys → list of problem strings
        return []

    def decide(self, env: Environment) -> AgentResponse:
        # 1) Read only via env.access / tools / evidence
        # 2) Call your agent
        # 3) Map to fintel View(s)
        # 4) Return AgentResponse(views=..., outcome="ok"|"abstained"|..., usage=...)
        ...
```

Map your native output → `View` (`symbol`, `score` ∈ [-1, 1], `conviction`, `time_horizon`, `rationale`, `key_factors`, optional `sources_cited`).

Only cite sources the agent **actually retrieved**. If a synthesizer never called access, leave `sources_cited` empty — do not invent citations.

---



## 5. Optional but recommended adapter polish


| Feature           | How                                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Live TUI stages   | `env.nerve.emit("agent_stage", cell=env.cell.name, decision_date=…, stage="<your_label>")` — TUI shows the string as-is |
| Persist internals | Write under `env.session.path` (e.g. `dossier.json`)                                                                    |
| Usage / cost      | Fill `Usage` (`n_llm_calls`, tokens, `cost_usd`, `basis`)                                                               |
| Knobs             | Dataclass fields → CLI `--agent-opt key=value` (values arrive as strings; coerce in `__post_init__`)                    |


Do **not** patch shared platform modules for agent-specific tracing. Keep hooks in the adapter (or your agent package).

---



## 6. Register (or don’t)

**One-off / WIP** — no factory edit:

```bash
--agent my_pkg.adapters.fintel:MyAgent
```

**First-class builtin** — add one line to `fintel/agents/factory.py`:

```python
"myagent": "fintel.agents.adapters.myagent:MyAgent",
```

Then: `--agent myagent`.

---



## 7. Runtime / deps

- In-process agents that need LangGraph / langchain / etc.: run fintel with that **venv**, and set `PYTHONPATH` to include both repos if the agent code lives outside fintel.
- Entry that works without `python -m fintel`:

```bash
python -c 'from fintel.cli.main import main; raise SystemExit(main())' simulation …
```

Secrets: simulation bootstraps from fintel’s local `.env/keys.env`
(gitignored; see `.env.example`). Shell exports still win if already set.

---



## 8. Smoke, then scale

**Smoke (cheap):**

```bash
fintel simulation packages/<strategy> \
  --agent myagent \
  --model <id> \
  --agent-opt <key>=<value> \
  --universe AAPL,MSFT \
  --dates 2025-01-01 \
  --cache-root <shared-cache> \
  --k 1 \
  --job-id smoke-myagent-0001
```

**Concurrency knobs:**

- `--max-concurrent` — parallel K repeats (runs)
- omit `--cell-concurrency` — auto = universe size (cells per date)
- `--trial-concurrency` — parallel dates (default 1)

**Check:**

- `runs/<job>/health.json` — access PIT / kinds
- `runs/<job>/r*/trials/<date>/cells/<sym>.json` — views
- `runs/<job>/r*/sessions/.../access.jsonl` — what was read
- optional session files (e.g. `dossier.json`) and nerve stages in `run.log` + live TUI

**Then** drop `--universe` / `--dates` overrides to use the package schedule for a real backtest; raise `--k` if you want repeat noise.

---



## 9. Checklist before you call it “wired”

- [ ] `decide(env) → AgentResponse` only
- [ ] `pit_enforcement` set; preflight clean
- [ ] No `market/` / `pit/` imports
- [ ] Strategy `[[data]]` covers every kind you read
- [ ] Views score in [-1, 1]; empty views ⇒ not `outcome="ok"`
- [ ] Smoke: health ok, expected reads/cell, TUI stages if you emit them
- [ ] Artifacts you care about land under `env.session.path` or cell JSON

---



## What you should *not* do

- Put the adapter inside the foreign agent repo as the primary home — **fintel owns adapters** (`agents/adapters/` for installed desks/CLIs).
- Teach the platform about “quant specialist” / “qual specialist” — that’s adapter logic.
- Force synthesizers to invent `sources_cited`.
- Fan out symbols inside `decide` when the cell is already one name.
- Reach around `env.access` for “just this one” fetch.

---



## Copy-paste path (OptimizedAgent / direct-access pattern)

1. Strategy already binds kinds — or add `[[data]]` rows.
2. `fintel/agents/adapters/<agent>.py` — evidence via `env.access`, call native pipeline, map `View`.
3. Register in `factory.py` **or** use `module:Class`.
4. Smoke 2 tickers × 1 date × `k=1`.
5. Sense-check access split, views, optional dossier / pipeline health.
6. Full package schedule + concurrency.

That’s the loop: **package binds data → adapter decides how to feed the agent → platform runs cells and scores.**