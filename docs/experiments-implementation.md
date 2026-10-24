# Experiment system: implementation notes

Companion to DESIGN.md section 6: module plan, exact API, PufferLib
pitfalls, the ordered test list and the design review changelog.

**Modules and approximate sizes**

| Module | Contents | LOC |
|---|---|---|
| `catanrl/__init__.py` | `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` before any CUDA use; re-exports | 20 |
| `catanrl/canon.py` | canonical JSON, digests, weight hash, AST digest, atomic / write-once / append IO | 90 |
| `catanrl/seeding.py` | `derive_seed` (v1), RNG state, `determinism()` | 60 |
| `catanrl/config.py` | plans, sweeps, stages, validation, coercion, migrations, hashing, diff, ref binding | 340 |
| `catanrl/provenance.py` | `code_id`, `capture`, patches, keep refs, checks, resume diff | 240 |
| `catanrl/checkpoint.py` | format, save/load, layout hash, self-test, per-part loading, errors | 220 |
| `catanrl/data/selftest_positions.json` | about 24 `apply_setup` positions (2–6p; BASE, extension and random maps; phases `apply_setup` can express) | data |
| `catanrl/runs.py` | `Run`, `RunWriter`, aliases, `load_experiment`, `load_checkpoint`, `fork`, `finish`, `check_runs` | 380 |
| `catanrl/agents/spec.py` | param dataclasses, registration, `resolve_agent`, fairness, canonical form | 200 |
| `catanrl/eval/suites.py` | `Suite`, schedule, lineup fill | 140 |
| `catanrl/eval/results.py` | key, `evaluate`, `EvalRecord`, `paired`, `verify`, `stale` | 260 |
| `catanrl/eval/stats.py` | log-space binomial fix; `paired_comparison` becomes private | +15 |
| `catanrl/_puffer.py` | the only importer of `pufferlib.pufferl` | 30 |
| `catanrl/train/puffer.py` | `CatanPuffeRL`, `train_ppo` | 220 |
| `catanrl/runner.py` | `run_plan`, `--next`, one subprocess per variant, `summarize` | 150 |
| `catanrl/report.py` | `report_data` | 170 |
| `catanrl/cli.py` | `run`, `fork`, `finish`, `pin`, `ls`, `eval`, `verify`, `results`, `check-runs`, `summarize`, `report-data`, `bot` | 200 |
| **Total** | | **≈2,550**, plus about 1,000 lines of tests |

**Exact API**

```python
# catanrl/canon.py
def canonical_json(obj: Any) -> bytes                       # sorted keys, no spaces, repr floats, tuples→lists, numpy scalars→python
def digest(obj: Any, n: int = 16) -> str
def weights_sha256(state_dict: Mapping[str, torch.Tensor]) -> str
def ast_digest(paths: Iterable[Path]) -> str                # ast.dump with docstrings stripped
def atomic_write_json(path: Path, obj: Any) -> None
def write_once_json(path: Path, obj: Any) -> None           # equal: no-op; different: ImmutableRecordError
def append_jsonl(path: Path, row: Mapping[str, Any]) -> None   # O_APPEND, one write() per row
def read_jsonl(path: Path) -> list[dict]                    # drops a torn last line
class ImmutableRecordError(RuntimeError): ...

# catanrl/seeding.py
def derive_seed(root: int, *labels: str | int) -> int       # v1 formula; "|" in a label raises
def seed_everything(seed: int) -> None
def rng_state() -> dict                                     # python, numpy, torch, cuda; weights_only-safe
def set_rng_state(state: dict) -> None
@contextmanager
def determinism(mode: Literal["strict", "fast"]) -> Iterator[None]   # sets/restores flags; strict asserts autocast off and CUBLAS env set

# catanrl/config.py
SCHEMA: int
MIGRATIONS: list[tuple[int, str, Any]]                      # (schema that added it, dotted key, legacy value)
@dataclass(frozen=True)
class Variant:
    experiment: str; name: str; stage: str | None; labels: dict[str, str]
    config: dict; config_hash: str; depends_on: tuple[str, ...]
    @property
    def ref(self) -> str: ...                               # "arch-lineage/graph-s-s42.g1"
@dataclass(frozen=True)
class Plan:
    experiment: str; source: Path; source_sha256: str; variants: list[Variant]   # dependency order
def load_plan(path: str | Path, *, overrides: Sequence[str] = (), only: str | None = None,
              variant: str | None = None, smoke: bool = False) -> Plan   # smoke: configs/smoke.yaml + epoch refs → @latest + LLM specs → stub/random (recorded)
def resolve(raw: dict, *, experiment: str, variant: str) -> dict
def config_hash(cfg: dict) -> str
def migrate(cfg: dict, *, from_schema: int) -> dict
def config_diff(a: dict, b: dict) -> list[str]              # ["train.learning_rate: 3e-4 -> 1e-4", ...]
def bind_refs(cfg: dict) -> dict                            # {"init_from": {...}, "lineups": [[canonical, ...], ...]}; reads the runs dir
def validate_plan(path: str | Path) -> list[str]            # suites exist, select purpose, heldout maps vs training pool, refs well-formed
class ConfigError(ValueError): ...
class ConfigConflict(RuntimeError): ...

# catanrl/provenance.py
def code_id(kind: Literal["eval", "train"], harness: Literal["native", "catanatron"] = "native") -> dict
def assert_code_current() -> None                           # StaleProcess if disk != import-time snapshot
def capture(*, argv: Sequence[str] | None = None) -> dict
def save_patches(prov: dict, dest: Path, *, cap_bytes: int = 20 << 20) -> None
def keep_commits(prov: dict) -> None                        # git update-ref refs/catanrl/keep/<sha> per repo
def torch_runtime() -> dict                                 # call after PuffeRL.__init__
def check(prov: dict, *, strict: bool = False) -> list[str]
def resume_diff(first: dict, now: dict) -> list[str]        # differing code_id components and uv.lock
class ProvenanceError(RuntimeError): ...
class StaleProcess(ProvenanceError): ...

# catanrl/checkpoint.py
CKPT_SCHEMA: int
@dataclass(frozen=True)
class CheckpointRef:
    run: str | None; which: str; epoch: int; agent_steps: int; path: Path; weights_sha256: str
    def __str__(self) -> str: ...                           # "arch-lineage/graph-s-s42.g1@e600#41f2a9c07d11"
class Checkpoint:
    ref: CheckpointRef; header: dict
    @classmethod
    def load(cls, path: str | Path, *, ref: CheckpointRef | None = None) -> "Checkpoint": ...
    def self_test(self, *, allow_drift: bool = False) -> None: ...
    def model(self, part: Literal["actor", "critic", "all"] = "actor", device: str = "cpu",
              *, allow_drift: bool = False) -> nn.Module: ...   # hash → layout → dims → encoder → model → .to(device).eval()
def save(path: Path, model: nn.Module, *, run: str | None, epoch: int, agent_steps: int,
         config: dict, code: dict) -> CheckpointRef        # same epoch + same weights: no-op; different weights: raise
def layout_hash(dims: Mapping[str, int], level: str) -> str # encoder.layout() + feature_names() + action layout
def make_self_test(model: nn.Module, dims: Mapping[str, int], level: str, n: int = 8) -> dict   # CPU, strict
def load_parts(model: nn.Module, init_from: dict) -> None
class CheckpointError(RuntimeError): ...
class WeightsCorrupt(CheckpointError): ...
class SchemaMismatch(CheckpointError): ...
class DimsMismatch(CheckpointError): ...
class EncoderDrift(CheckpointError): ...
class ModelDrift(CheckpointError): ...

# catanrl/runs.py
Status = Literal["running", "finished", "time_limit", "interrupted", "failed", "dead"]
def runs_root() -> Path
def load_experiment(ref: str | Path) -> "Run"               # "exp/variant", "exp" with one variant, or a directory
def list_runs(pattern: str = "*", *, status: Status | None = None) -> list["Run"]
def load_checkpoint(ref: str | Path) -> Checkpoint          # "exp/var@which[#sha]" or a .pt path
def pin(ref: str, name: str, note: str = "") -> None        # appends to aliases.jsonl; safe while training
def fork(ref: str, *, as_: str, overrides: Sequence[str] = ()) -> Variant
def finish(ref: str) -> None
def check_runs(pattern: str = "*") -> list[dict]
class Run:
    ref: str; path: Path; config: dict; config_hash: str; manifest: dict; status: Status
    def checkpoints(self) -> list[CheckpointRef]: ...       # committed only; never orphaned/
    def aliases(self) -> dict[str, dict]: ...
    def resolve(self, which: str | int = "best") -> CheckpointRef: ...   # unknown: error listing existing selectors
    def checkpoint(self, which: str | int = "best") -> Checkpoint: ...
    def build_policy(self, which: str | int = "best", device: str = "cpu", *, allow_drift: bool = False) -> nn.Module: ...
    def build_critic(self, which: str | int = "best", device: str = "cpu", *, allow_drift: bool = False) -> nn.Module: ...
    def agent(self, key: str = "POLICY", which: str | int = "best", **params: Any) -> "ResolvedAgent": ...
    def evals(self, suite: str | None = None) -> list["EvalRecord"]: ...
    def metrics(self, *, lineage: bool = False) -> list[dict]: ...
    def lineage(self) -> list[CheckpointRef]: ...
    def provenance(self) -> list[dict]: ...
class RunWriter(Run):                                       # context manager; __exit__ records failed/interrupted
    mode: Literal["new", "resume", "done"]; launch: int; run_id: str
    stop_reason: Literal["time_limit", "interrupted"] | None
    @classmethod
    def open(cls, variant: Variant, *, argv: Sequence[str], allow_code_change: bool = False) -> "RunWriter": ...
    def log(self, logs: dict, step: int) -> None: ...       # PufferLib logger interface
    def close(self, model_path: str | None = None, early_stop: bool | None = None) -> None: ...
    def save_checkpoint(self, model: nn.Module, *, epoch: int, agent_steps: int) -> CheckpointRef: ...
    def save_trainer_state(self, state: dict) -> None: ...  # the commit; then run.json
    def load_trainer_state(self) -> dict | None: ...        # also moves orphans and reconciles run.json
    def pending_selects(self) -> list[CheckpointRef]: ...
    def set_alias(self, name: str, ref: CheckpointRef, *, record: "EvalRecord | None" = None, note: str = "") -> None: ...
    def should_stop(self) -> bool: ...
    def finish(self, status: Status) -> None: ...

# catanrl/agents/spec.py
@dataclass(frozen=True)
class PolicyParams:
    ckpt: str; sample: bool = False; temperature: float = 1.0; allow_drift: bool = False
    device: str = field(default="auto", metadata={"execution": True})
@dataclass(frozen=True)
class MctsParams:
    ckpt: str; sims: int = 256; dets: int = 1; oracle: bool = False
    root: Literal["puct", "gumbel"] = "puct"; c_puct: float = 0.75; gumbel_m: int = 16
    temperature: float = 0.0; noise: bool = False; allow_drift: bool = False
    device: str = field(default="auto", metadata={"execution": True})
    threads: int = field(default=0, metadata={"execution": True})
@dataclass(frozen=True)
class BeliefParams:
    ckpt: str; worlds: int = 16; aggregate: Literal["mean_prob", "mean_logit"] = "mean_prob"; allow_drift: bool = False
    device: str = field(default="auto", metadata={"execution": True})
@dataclass(frozen=True)
class PredictParams:                                        # DESIGN 2.1: a public/full policy on the predictor's fill
    ckpt: str; predictor: str; round: bool = False; sample: bool = False; temperature: float = 1.0
    allow_drift: bool = False
    device: str = field(default="auto", metadata={"execution": True})
def register(registry: Registry = REGISTRY) -> None         # idempotent; on import of catanrl.agents
def resolve_agent(spec: "str | dict | ResolvedAgent", *, harness: Literal["native", "catanatron", "server"] = "native",
                  games: Sequence[GameConfig] = ()) -> "ResolvedAgent"   # raises catanatron.registry.SpecError
@dataclass(frozen=True)
class ResolvedAgent:
    key: str; impl: str; params: Any; checkpoint: CheckpointRef | None
    fair: bool; label: str; canonical: str; predictor: CheckpointRef | None = None   # PREDICT's second checkpoint
    checkpoints: tuple[CheckpointRef, ...]                  # property: checkpoint, then predictor
    def key_dict(self) -> dict: ...
    def native(self, *, device: str, infer_rows: int) -> "NativeAgent": ...
    def player(self, color: Color) -> Player: ...
class NativeAgent(Protocol):
    def act(self, rows: np.ndarray, masks: np.ndarray, decision_seeds: np.ndarray) -> np.ndarray: ...
def player(spec: str, color: Color) -> Player

# catanrl/eval/suites.py
@dataclass(frozen=True)
class ScheduledGame:
    game_id: str; scenario: int; board: int; rotation: int; num_players: int
    map_seed: int; game_seed: int; bot_seed: int; map_hash: str; seats: tuple[int, ...]
@dataclass(frozen=True)
class Suite:
    name: str; purpose: Literal["select", "heldout", "dev"]; seed: int; boards: int
    scenarios: tuple[dict, ...]; turn_limit: int; determinism: Literal["strict", "fast"] = "strict"
    def key(self, lineup_size: int, *, fill: bool = False) -> dict: ...
    def schedule(self, lineup_size: int, *, fill: bool = False) -> list[ScheduledGame]: ...
    def schedule_hash(self, lineup_size: int, *, fill: bool = False) -> str: ...
def load_suite(name_or_dict: str | Mapping[str, Any]) -> Suite

# catanrl/eval/results.py
def eval_key(lineup: Sequence[ResolvedAgent], suite: Suite, *, harness: str, fill: ResolvedAgent | None = None) -> dict
def find(lineup: Sequence[str | dict | ResolvedAgent], suite: str | Suite, *, fill: str | None = None,
         harness: str = "native") -> "EvalRecord | None"
def evaluate(lineup: Sequence[str | dict | ResolvedAgent], suite: str | Suite, *, fill: str | None = None,
             harness: str = "native", recompute: bool = False, allow_mixed: bool = False, device: str | None = None,
             num_envs: int | None = None, threads: int | None = None, infer_rows: int | None = None) -> "EvalRecord"
@dataclass(frozen=True)
class EvalRecord:
    eval_id: str; key: dict; summary: dict; path: Path
    def games(self) -> Iterator[dict]: ...
    @classmethod
    def load(cls, path: str | Path) -> "EvalRecord": ...
def paired(a: EvalRecord, b: EvalRecord, *, slot_a: int = 0, slot_b: int = 0) -> dict
def verify(record: EvalRecord | str, *, games: int | None = 16) -> dict   # {"identical": k, "of": n, "differing": [...]}
def stale(pattern: str = "*") -> list[EvalRecord]

# catanrl/eval/stats.py (existing)
def binomial_two_sided(successes: int, trials: int, p: float = 0.5) -> float   # pmf via math.lgamma

# catanrl/_puffer.py
def import_pufferl() -> ModuleType                          # inside warnings.catch_warnings(); then reinstalls handlers
def install_signal_handlers(on_stop: Callable[[str], None] | None = None) -> None

# catanrl/train/puffer.py
class CatanPuffeRL(pufferlib.pufferl.PuffeRL):
    def __init__(self, config: dict, vecenv: Any, policy: nn.Module, *, writer: RunWriter, select: dict | None) -> None: ...
    def save_checkpoint(self) -> str: ...                   # writer.save_checkpoint → trainer_state → run.json → monitor evals (eval.monitor: a metrics row each) → select eval → alias
    def close(self) -> None: ...                            # no data_dir copy
    def state(self) -> dict: ...
    def restore(self, state: dict) -> None: ...
def train_ppo(v: Variant, w: RunWriter) -> None

# catanrl/runner.py, catanrl/report.py
def run_plan(plan: Plan, *, next_: bool = False, allow_code_change: bool = False) -> None   # one subprocess per variant
def run_variant(v: Variant, *, argv: Sequence[str], allow_code_change: bool = False) -> None
def summarize(experiment: str) -> Path
def report_data(spec: str | Path, *, run_missing: bool = False, update: bool = False,
                final: bool = False, verify_games: int = 16) -> Path
```

**Notes for implementers** (from the critiques; checked read-only against the submodules)

* **Logging.** `PuffeRL.train` logs only when 0.25 s have passed (`pufferl.py:471`). Set `t.last_log_time = 0.0` before every `t.train()` and compute SPS yourself. Comparisons leave out `SPS`, `uptime` and `performance/*`.
* **Autocast.** `amp_context.__enter__()` runs twice per minibatch and is never exited (`pufferl.py:350`, `433`), so bf16 autocast persists. Wrap the select eval in:
  * `torch.autocast("cuda", enabled=False)`,
  * `torch.random.fork_rng()`,
  * `determinism("strict")`.
* **Import side effects.** Importing `pufferl` turns `RuntimeWarning` into errors (line 7) and installs `os._exit(0)` on SIGINT (line 47). Import it only through `_puffer.import_pufferl()`; `cli.main` installs catanrl's handlers for every command.
* **Seeding and flags.** PufferLib's seeding is commented out (lines 65-68). `__init__` sets matmul precision 'high', `cudnn.benchmark=True` and `cudnn.deterministic=config['torch_deterministic']`, so apply catanrl's flags after it.
* **Hidden config keys.** `amp` is read with `config.get('amp', True)` (line 195) and is not in `default.ini`. Muon goes through heavyball with `compile_mode="default"` (line 160), so it compiles even with `compile = False`. `ppo.yaml` uses Adam; record the compile mode if Muon is chosen.
* **Rewards** are clamped to [-1, 1] (line 275). `env.reward` validation refuses shaping that can push a step outside that range.
* **Recurrent policies.** `use_rnn` follows the model (`model.recurrent`). PuffeRL then keeps the training observations as segments x horizon and sizes an `lstm_h`/`lstm_c` dict by `policy.hidden_size`, but it zeroes that dict at every `evaluate` and passes `lstm_h=None` to the training forward, so `PufferPolicy` keeps the rollout memory itself (one state per row, carried across epochs, zeroed where `done` starts a new game) and snapshots it before each `evaluate` (`begin_rollout`). The fork's fifth patch passes the minibatch's `idx` and `terminals` in the training `state`, so each segment replays the rollout's recursion from its stored initial state. The engine sets `terminals` for truncations too, so one flag marks every game boundary. The memory is float32 in the rollout and along the replayed horizon even under `train.amp` (CUDA's autocast returns bf16 from the cell; the cast back is explicit).
* **Budgets.** `global_step += mask.sum()` (line 252). Log `decisions` too, and compare budgets across different `dims.max_players` by epochs or decisions.
* **The aux head (DESIGN 2.1).** The fork's third hook: after the value loss, `aux_loss(outputs)` on the policy (the forward's second output, `ValueBatch`) returns `None` (the loss is untouched: the non-aux path is bitwise the old one, guarded by `test_aux_coef_zero_is_the_plain_training` and the golden training, whose pins were refreshed on this branch at the values wt/integration produces since its lstm merge) or `(weighted loss, {name: tensor})`, added to the objective and logged under `losses/<name>`. `PufferPolicy.forward` runs `GraphModel.heads_with_aux` on the acting rows when `aux_coef > 0` (one trunk pass; the rollout never runs the head) and reads the targets from the row's `critic_players` block (`aux_targets`); `pop_aux` gives the epoch's per-feature `mse_` / `var_`, logged as `aux/*` by `step_epoch`. The head is built last in `GraphNetwork`, so for one seed the trunk, policy and value start from the plain shared model's weights. Config: `train.aux_coef` (ppo.yaml, schema 3, migrated at 0.0) and `model.args.aux` (a graph class default, migrated at `false` for stored graph configs: resolution fills class defaults into `model.args`, so a new class default moves every stored hash of that class and needs a `model.args.*` migration scoped by class; schema 4 added the same default on the lstm class, migrated the same way for stored lstm runs: lstm-4p/main and arch-ablation/lstm, whose schema-3 hashes `tests/test_config.py` pins). On the recurrent network (`LstmModel`, `args.aux`) the head sits on the memory-injected embeddings and `_forward_recurrent` runs `decode_with_aux` on the replayed memory, so the aux loss reaches the cell; `aux_targets` is shared with the graph (`aux_targets_from_block`). Tests: `tests/test_lstm_aux.py`. `train_ppo` refuses `aux_coef > 0` without the head or without the block. `NativeAgent` drops a critic block the model does not read (`reads_critic_block`), for searches too. Checkpoints self-test the head under `part="all"` and `part="aux"` (`self_test["aux"]`; the `actor` and `critic` self-tests never run it, and an unknown part is refused). Tests: `tests/test_aux_head.py`. A final naming another experiment's run (aux-hidden's head to head against arch-ablation/graph-s) is pinned at creation only when that run is already usable; otherwise `bind_refs` leaves it, `bound_lineup` raises `Unbound` until the run is usable, `final_evals` defers the entry and `_missing_finals` does not count it (a `--next` worker would otherwise loop on it), so the variant is created, trained and smoked without the control and `catanrl run aux-hidden` pins and plays the deferred final once arch-ablation/graph-s is finished.
* **Critic refits (DESIGN 6.4).** `train.freeze` (ppo.yaml, schema 6, migrated at `[]` for stored PPO configs) names `model.parts` that do not train: `freeze_parts` stops their gradients, `CatanPuffeRL` drops them from the optimizer before its first step (`drop_frozen`, before `muon_exclude`, which then refuses a frozen parameter), and with `actor` frozen `check_policy_frozen` runs the policy on two self-test positions and refuses logits that still require a gradient. Nothing in the fork changes: the policy and entropy terms reach no trained parameter, so the update is the value loss alone (and `losses/approx_kl` stays about 0, `clipfrac` 0). `init_from.split_shared` maps a split graph's `actor.trunk.`/`actor.policy.` and `critic.trunk.`/`critic.value.` onto a shared checkpoint's `backbone.`, `policy_head.` and `value_head.` (`checkpoint.SPLIT_SHARED`), every source key of a part used once. `init_from` loads whenever a launch starts fresh (`puffer.starts_fresh`: a new run, or a resume with no `trainer_state.pt` because every earlier launch died before its e0 commit; PPO, AlphaZero and DAgger alike), so such a relaunch no longer resumes into the seed's random network; with `actor` or `critic` frozen, `check_frozen_init` then refuses, at every launch and after the committed state is restored, a frozen part `init_from` does not load or whose tensors differ from `init_from`'s (`checkpoint.init_tensors`, the tensors `load_parts` copies). `tools/critic_heldout.py` scores a refit's checkpoints on one fixed set of the frozen policy's games on boards outside the training pool (paired log loss, Brier, AUC within turn bins, calibration-only baselines); `tools/search_probe.py` resolves an eval id stored by several runs when every copy holds the same games. Tests: `tests/test_critic_refit.py`, `tests/test_critic_heldout.py`.
* **The LLM agent and the Catanatron harness (DESIGN 3, 6.6).** `catanrl.agents.llm`: `prompts.py` (catanllm's five strategies as strings, generated once from `~/dev/catanllm/prompts` by splitting each file at its board section; `system_prompt(strategy, catan_map, vps=)` appends `board_section(map)`: tile corners and land-node neighbours from the fork's `CatanMap.graph`), `render.py` (`render_state(game, observer)`: catanllm's non-verbose layout with `player_status` redacted per seat; `render_actions` numbers the playable actions on the fork's action values: `DISCARD_RESOURCE`, `MOVE_ROBBER (coordinate, victim)`, `PLAY_KNIGHT_CARD` without a value), `model.py` (`StubModel` reads the action count back out of the prompt's `=== AVAILABLE ACTIONS ===` block; `OpenRouterModel` is the openai SDK at OpenRouter's base URL with `extra_body={"reasoning": {"max_tokens": 0.75 * max_tokens}}` when `thinking`, else `{"enabled": false}` plus catanllm's `/no_think` suffix for qwen; check both against OpenRouter's current reasoning docs before a paid run), `player.py` (`LlmPlayer`, `parse_answer`, `fallback_action`; `before(game)` rebuilds a stub from `derive_seed(seed, "llm", game.seed, color)` so stub games replay, a model given to the constructor is kept) and `cost.py` (`catanrl llm-cost`). `spec.py` routes `LLM` through `_resolve_llm` (catanatron harness only, `fair=True`, `key_dict` = all params; `check_spec_syntax` tolerates a plan's unresolved `${params.…}` values). `catanrl.eval.catanatron.play_catanatron` is the harness (`reseat` forces the schedule's seat order the way `cppanatron.catanatron.reference_game` does; `preflight` builds every player once so a missing key fails before any game; `CATANRL_LLM_TRANSCRIPTS=<dir>` saves every exchange as `<game_id>.s<slot>.jsonl`); `results._play_games` dispatches on the harness for `evaluate` and `verify`, `NOT_OUTCOME = ("segment", "llm")` is what `_strip` drops, `llm_totals` fills `summary.slots[i].llm`, and `provenance._components` hashes `eval/catanatron.py` into the catanrl component only for `harness="catanatron"`, so native records do not move when the harness changes. `eval.final` entries take `harness` (`ENTRY_KEYS`; `_check_semantics` allows native or catanatron), which `runner.final_evals`, `_missing_finals`, `summarize` and the report's cite entries pass through. `llm-eval.yaml` ties the model axis to short labels (`stub`, `gpt5`, …) because `/` cannot appear in a variant name; a smoke plan replaces the model of every `LLM:` spec in `eval.final` and `eval.select` (lineup and fill) with `stub/random` (`config._smoke_spec` → `spec.stub_llm_spec`, the canonical form), so `catanrl run llm-eval --smoke` plays all 16 cells without a key or a cost (the real-model cells share the stub cells' records); one real model is tried on one board with `catanrl eval 'LLM:model=…' F --suite heldout-2p --boards 1 --harness catanatron`. The `POLICY:ckpt=puffer-graph2/full@best:device=cpu` opponent is a foreign run: pinned at variant creation when it is usable, else deferred. Review fixes on wt/llm: `parse_answer` reads the reply's last line only (the number alone, `action N` on it, the number leading or alone on it unless next to a board word such as `node 12`; `Final answer: 3.` parses; a number in the reasoning never counts), an empty reply is retried with an `(no answer)` assistant placeholder (an empty assistant message is a 400 at some providers), `render_state` states a pending discard (`You must discard N more of your H resource cards`), the rulebook and strategy guide say the game's VP target (`{vps}`; the rulebook's worked example keeps "ten"), `preflight` calls `player.preflight()` (the LLM builds its client, `CatanrlPlayer` loads its checkpoint) so a missing key fails before any game, a game that raises sets a stop event (running games abandon at their next tick with `GameStopped`, pending ones are cancelled, workers are joined, rows of games that finished are still delivered; KeyboardInterrupt behaves the same), `provenance.CATANATRON_ONLY` keeps `agents/llm` out of the native code id (a prompt edit no longer moves native records), `results.verify` refuses a record with a real model (`Unreplayable`, `real_llm_slots`) unless `force` / `catanrl verify --force`, `report._eval_flags` flags it `nondeterministic` and `--final` skips its replay, and `catanrl llm-cost` pads the stub's notes (`--notes-chars`, default 200; `StubModel(notes_chars=)` through `LlmPlayer.stub_options`) and projects the prompt tokens at `--decisions` prompted decisions (default twice the measured) from the measured per-decision base and per-note cost (`cost.projected`).
* **The PREDICT agent (DESIGN 2.1, 6.5).** `spec.PredictParams` (`ckpt`, `predictor`, `round`, `sample`, `temperature`, `allow_drift`; `device` execution-only); `PREDICT` is in `MODEL_KEYS`, so everything that enumerates model keys (`check_spec_syntax`, `key_dict`, `runs.bind_spec` / `spec_runs` / `foreign_runs`, the registry, and the bridge's `is_model_spec`, which imports the spec module's tuple rather than keeping its own) covers it. `_resolve_model` pins the policy as POLICY does (part `actor`) and the predictor with part `aux` (its self-test compares the aux head's predictions: the trunk and head the agent runs; `Checkpoint.model("aux")` loads it), then `_check_predict` refuses a `private` policy, a predictor without the head, at another level, or recurrent; the canonical spec carries both pinned refs, `key_dict` adds `predictor` (weights sha, model, levels, dims), `ResolvedAgent.predictor` / `.checkpoints` expose them, the record adds `predictors` beside `refs`, and `report._training` walks every checkpoint's lineage. `spec_ckpts` lists both refs as written; `runs.spec_runs` / `foreign_runs` list both runs, so `bind_refs` pins the spec only when every foreign run is usable and `bound_lineup` raises `Unbound` naming the first that is not. `native.PredictAgent` (`needs_games`, no trackers): `NativeAgent` gained `reads_level` (the rows the agent reads, `private` here, while `_levels` stay the network's; `_rebuild` and the critic-block drop use `_levels`), `_set_model` rebuilds the predictor at the policy's dims on private rows and evaluates it through `BatchEvaluator(heads="aux")` (the aux predictions flattened in the logits' place, so strict padding, chunking and CUDA graphs are the policy's), `policy_rows` writes the predictions into the hidden-feature columns (`hidden_feature_indices` of the row's players block, `shown_hidden_features(policy level)`) of the present other slots, `round_predictions` snaps to `float32(count / total)` with the totals from `game.rules` (bank, deck) and `game.vps_to_win`, flags at 0.5; `act` runs the policy on those rows and `_choose`s like POLICY. `make_native_agent` takes `predictor=` for in-memory networks (`tests/eval_helpers.agent`). `spec.PredictPlayer` is the Catanatron player (the `CatanrlPlayer` mirror passes the game); the server bridge refuses the key at resolution, and `catanrl play` (the TUI) routes a PREDICT seat through `ViewAgent` like the other model keys, so it is refused there too, before the TUI opens. `configs/experiments/predict-vs-belief.yaml` is the eval; its code is outcome code (DESIGN 6.10), so the native eval code id moves with the merge and the existing records (puffer-graph2's BELIEF-vs-F final among them) get new eval ids: the merge budgets replaying that final (5,000 games, ~390 s on the GPU) before row 2 can be paired with it. Tests: `tests/test_predict_agent.py`.
* **The TRADE agent (DESIGN 6.5; domestic trades stay outside the action space).** `spec.TradeParams` (`ckpt`, `critic`, `margin`, `accept_margin`, `dets`, `min_feasible`, `tracker`, `offers`, `when`, `shapes`, `propose`, `oracle`, `sample`, `temperature`, `allow_drift`; `audit` and `device` execution-only; the owner's defaults: margins `0.0` with strict `>` comparisons, `dets=8`, `min_feasible=0.25`, `offers=3`, `when=first`, `shapes=11-21-12`); `TRADE` is in `MODEL_KEYS`, so `check_spec_syntax`, `key_dict`, `runs.bind_spec` / `spec_runs` / `foreign_runs`, the registry and the bridge's `is_model_spec` cover it. `_build_params` refuses negative margins, `offers < 1`, `dets < 1`, `oracle=true` with a written `dets > 1` or with `tracker=exact` (as MCTS; `oracle=true` alone pins `dets=1`, the true game being the one world, so the design's `T(oracle=true)` resolves and every oracle lineup has one canonical key), `min_feasible` outside `(0, 1]`, and `shapes` that is not a `-`-separated non-empty subset of `11`, `21`, `12` (`spec.trade_shapes` parses it to `(give, get)` sizes). `_resolve_model` pins the policy with part `actor` and the critic with part `critic`, or the one checkpoint with part `all` when `critic` is empty or names `ckpt`, then rewrites `critic` to the pinned ref so the canonical spec always names it (`TRADE:ckpt=X` and `TRADE:ckpt=X:critic=X` key identically); `_check_trade` refuses every harness but the native one (the Catanatron mirror returns one action; the server bridge cannot negotiate), a recurrent critic, a critic with a separate critic block (`critic_level_of` is not None: the agent encodes one row per candidate at the actor's level) and a critic above `private` (v1: one row per candidate; a public or full critic would need world-averaged rows); a recurrent *policy* is allowed natively, as for POLICY. `fair` = the policy's POLICY rule and not `oracle` (the critic reads private rows of worlds dealt from the seat's tracker; `oracle=true` reads the true hands, so `report._row` cites it only with `oracle_ok`). `ResolvedAgent.critic` / `.checkpoints` (each ref once) expose the critic, `key_dict` adds `critic` (weights sha, model, levels, dims, allow_drift) and skips `ckpt`/`critic`/execution params, the label is `TRADE:<run>@eN` plus `+critic <run>@eM` when it differs, the record adds `critics` beside `predictors`, and `report._training` walks both lineages. `spec.TradePlayer` holds the registry entry and the params (its `decide` is unreachable). Records: a row of a game with a domestic trade carries `trades` = `{offers, accepts, confirms, cancels}` per seat (`results.TRADE_KEYS`; written by the harness only when a count is non-zero, so every existing record's rows and `verify` / `recompute` stay byte-identical); `results.summarize` reads it with `.get` and adds per-slot totals with per-game means and the cancel rate (`trade_totals`, cancels over the offers somebody accepted) to the slot's summary, which `catanrl eval` prints; the readings a record cannot hold (the bundles offered and confirmed, the gains the critic claimed for them, the fronts read, the critic rows per read) come from `tools/trade_pilot.py`, which replays a lineup (`--lineup`, or an experiment's `eval.final` row with `--config/--row`) through the native harness with the agents held in process and reads `TradeAgent.decisions` / `.stats`, writing no record; `results.paired` already joins `[TRADE, N x3]` with `[POLICY, N x3]` (slots 1-3 identical) and refuses tables whose other slots differ. The agent itself (`agents/trade.py`, `native.TradeAgent`: the mover-side menu, stand-in partner hands, ego-view gains read from the proposer's play-turn row, the Pareto front offered in order, the responder's mean over `dets` worlds, the confirm by own gain) and the harness side (`harness._play_vecenv` routes trade-prompt rows by their all-zero mask to `respond_trade`, injects non-indexed actions through `VecEnv.set_action`, counts the per-seat `trades`) are the other halves of the design. `tests/eval_helpers.agent("TRADE", model, critic=...)` builds one over in-memory networks. `configs/experiments/trade-harness-pilot.yaml` (200-game pilots: the defaults, the 1 pp arm, `when=every`, the oracle ceiling) and `trade-harness.yaml` (4,000-game rows M1/M2 with their `POLICY` controls `M1c`/`M2c`) are the evals; the stated prior is a null at 2 pp (the mutual pool sits at the critic's noise floor) and the 0 pp row is the owner's rule. Tests: `tests/test_agent_spec.py` (specs), `tests/test_results.py` (records).
* **Checkpoint hook.** PufferLib's `save_checkpoint` returns early if the file exists (lines 544-545). It is fully overridden; nothing of its directory layout is used.
* **Batch sizes.** `batch_size >= minibatch_size` is enforced (line 127). `configs/smoke.yaml` sets Adam, `bptt_horizon: 8`, `minibatch_size` equal to the batch, and compile off.
* **Spec parsing.** `registry.parse_spec` splits on every `:` and `build_all` caps games at 2–4 players. Cite these by symbol, since submodule line numbers drift.
* **MCTS mapping.** The engine's `SearchConfig.determinizations = 0` means the game as given. Map `oracle=true` to 0 and `dets=K` to K. `c_puct` defaults to 0.75 in engine units.
* **MCTS tree reuse, leaf dedup and transpositions.** `MctsParams.reuse` (default false): `MctsAgent` keeps one live search per stream (`needs_memory`, so the harness hands `streams` and `first`) and at the stream's next decision re-roots it at the action played (`SearchPool.advance`: the subtree is kept, the budget restarts) when the game handed is exactly the previous game plus that action (`Game.search_equivalent` on `prev.copy().execute(action)`: a chance action, a forced move or another seat's move in between diverge) and the engine keeps the worlds (its own refusals: chance children, another player's decision, a reveal, worlds disagreeing on the new root); otherwise it removes the live search and adds a fresh one from the game and tracker handed now. A seat's first decision of a game drops every live search of its env (streams are `env * max_players + seat`): the env's previous game is over, and a seat of the new game that decided already loses nothing, since that first decision is a move between its own and its next decision would be refused as diverged anyway; nothing is added to game rows or to harness.py; `agent.counters` (`decisions`, `fresh`, `reused`, `dropped_first`, `dropped_over`, `refused_diverged`, `refused_engine`) and `agent.search_stats` (`SearchStats` sums) report it, `tools/search_probe.py` prints them over a schedule. Rows depend on the stream's history (a reused search keeps its RNG and retained visits), so `reuse` is keyed. The server bridge streams it too: `ViewAgent` hands one stream per game slot and seat (`env * max_players + seat`, the slot a per-game integer; a game's last message, `after` or `aborted` with its id, drops its searches through `MctsAgent.forget` and frees the slot at the next decision, and a game whose end was never seen keeps them until it is evicted, when its slot's next game drops them at its first decision) and `first` at a seat's first decision of a game, and sets `MctsAgent.reuse_check = "view"`: the replayed game holds placeholders and an RNG of its own, so it is matched to the previous game plus the action by the seat's private row, the legal actions and the mover, prompt, turn and phase flags; with `dets >= 1` the engine's own refusals (a drawn outcome, another player's decision, a monopoly, worlds disagreeing on the new root; the friendly robber's options read public points since 2026-10-24, so its prompt is no longer refused) keep every retained world a fair world plus the seat's own move (DESIGN 6.5). A forced move answered without the agent diverges its search (refused, fresh). The catanatron harness refuses `reuse`. Memory: at most one retained subtree per seat the agent holds in an env's current game, about 12 MB at 1024 sims in 4p, i.e. `num_envs x MCTS seats x 12 MB`; no eviction, since evicting by pool population would depend on `num_envs`. `MctsParams.dedup` (execution-only, default true) wraps the agent's evaluator in `catanrl.search.dedup.DedupEvaluator`: byte-identical rows of a batch (blake2b keys) are forwarded once and a 65,536-row LRU cache answers repeats of earlier batches; under the strict contract (a row's outputs never depend on its batch) that is bitwise the plain evaluation, which the golden and invariance tests pin with it on. It hashes only where a merge can save a forward (CPU, `fast` mode, batches above `infer_rows`; `CATANRL_DEDUP_FORCE=1` always), so a strict GPU run costs nothing. `MctsParams.transpositions` (default false) passes `SearchConfig.transpositions` through (the engine's per-world transposition table: a position another order of actions already reached links to the existing node, searched once; `advance` re-parents the retained graph, so `reuse` and `transpositions` combine). Key gating: `reuse` and `transpositions` carry `metadata={"key_if_set": True}` and `spec.keyed_fields` leaves them out of the canonical spec and of `key_dict()["params"]` at their default; `spec.search_details` leaves `SEARCH_KEY_IF_SET` attributes (`transpositions`, `canonical_discards`) out of `key_dict()["search"]` at the engine default; so every stored MCTS record's key and canonical spec rebuild unchanged (`tests/data/stored_mcts_keys.json` pins the stored ones, `-m runs` checks the real directory) and old records stay verifiable (`verify` ignores `code`). `results.paired` still refuses records whose code differs, since a code change can move every seat's rows: a new row is paired with a stored baseline by replaying the baseline's row at the current code. `tools/search_probe.py --pair A B` scores a slot on the games two records share across code and schedule (a shorter schedule of the same suite is a prefix of a longer one) and reports `code_differs` and `schedule_differs`: a reading aid, not a record-level comparison. Tests: `tests/test_mcts_reuse.py` (reuse alone and with the table: the re-rooted graph keeps its links), `tests/test_dedup.py`, the `mcts-reuse*` cases of `tests/test_eval_invariance.py` (`mcts-reuse-tt*` play both options together), the `transpositions` params of `tests/test_search_invariance.py`.
* **Seat order.** Catanatron shuffles seats with the game RNG (`state.py:97`). Read seat order back from `game.state.colors`.
* **Determinism.** Graph message passing must use ops that `use_deterministic_algorithms(True)` supports on CUDA; `-m gpu` catches any that are not.

**Tests, in the order to write them**

*P0, before the first real run:*

1. `tests/test_canon.py`:
   * canonical JSON is stable and `digest` is pinned;
   * `weights_sha256` is unchanged by save→load→save and by key order;
   * write-once raises on different content;
   * `read_jsonl` drops a torn last line.
2. `tests/test_seeding.py`:
   * `derive_seed` equals pinned v1 values;
   * `|` in a label is rejected;
   * `determinism()` restores the flags.
3. `tests/test_stats.py` (extend): the binomial at n = 1030 and 5000 matches `scipy.stats.binomtest`.
4. `tests/test_config.py`:
   * unknown key with a suggestion; `3e-4` is coerced;
   * list, tied and exclude axes; name collisions; `train!`; a stage and an axis setting the same key;
   * interpolation; `${self}` is allowed only in select; an unreferenced `params` key raises;
   * the hash ignores `meta`, `runtime` and `eval.final` (train);
   * golden hashes of `tests/fixtures/experiments/*.yaml`;
   * migration: an old stored config plus a new key with its legacy value gives an equal hash; a new schema key without a migration fails against a schema snapshot;
   * execution keys under `train` raise.
5. `tests/test_ppo_keys.py`: the keys of `ppo.yaml` plus the runner-set keys equal the `config[...]` and `config.get(...)` keys read in `PuffeRL`, found by an AST scan.
6. `tests/test_checkpoint.py`:
   * a tiny MLP and a tiny graph net give identical logits through `build_policy`, `load_checkpoint().model()` and `POLICY:`;
   * loading works with `weights_only`;
   * a tampered byte raises `WeightsCorrupt`, a changed layout `SchemaMismatch`, a patched encoder `EncoderDrift`, a perturbed forward `ModelDrift`;
   * `allow_drift` is recorded;
   * a graph net loads at larger dims while an MLP raises;
   * a mismatched `init_from` part raises.
7. `tests/test_runs.py` (temporary runs dir, `chdir` elsewhere):
   * every row of the relaunch table; the lock is exclusive;
   * an `os._exit` child shows as `dead`;
   * a crash injected at each of the 5 checkpoint write points: orphans moved, `run.json` reconciled, missing selects re-run;
   * all selectors and `#sha` resolve, and an unknown selector lists the existing ones;
   * `best` requires a select-purpose record; `pin` works under the lock; bad names are refused;
   * a `.pt` loads after its run dir is deleted;
   * `fork`, `finish`, and a `time_limit` parent feeding a child.
8. `tests/test_puffer_import.py`:
   * after `import_pufferl()`, the SIGINT and SIGTERM handlers and the warning filters are catanrl's;
   * Ctrl-C on a `catanrl eval` subprocess exits nonzero.
9. `tests/test_agent_spec.py`:
   * canonical form is a fixed point, and `@best` becomes `@eN#sha`;
   * `det=8`, `oracle=true` with `dets>1`, and `,` or `:` in a ref all raise;
   * execution params are out of `key_dict`; MCTS `key_dict` holds the full `SearchConfig` read back from the binding, less `transpositions` at its default, and `reuse` and `transpositions` appear in `params` only when set, so stored MCTS keys rebuild unchanged;
   * the `fair` table holds; a public/full POLICY is refused on the server harness; Catanatron-only keys are refused natively;
   * POLICY, MCTS and BELIEF are in `REGISTRY`; `c_puct` equals the engine default.
10. `tests/test_suites.py`:
    * the first 8 `ScheduledGame`s are golden;
    * select and heldout seeds and map hashes are disjoint;
    * every rotation appears once per seat; the quota is exact;
    * a larger suite extends a smaller one; renaming a game leaves seeds unchanged;
    * `fill` expands correctly for 4p and 6p.
11. `tests/test_results.py` (fake harness with a call counter):
    * `eval_id` is pinned; records are write-once;
    * a partial file plays only the missing ids;
    * an execution mismatch refuses without `--allow-mixed`;
    * a forfeit makes the record invalid; truncation re-scores without replaying;
    * port v1's `test_paired_comparison.py`: join, mismatch and missing-id errors.
12. `tests/test_provenance.py` (temporary superproject with a submodule):
    * dirty, untracked and `+` states are detected;
    * the patch passes `git apply --check`; keep refs exist;
    * strict refuses a submodule mismatch;
    * `code_id` changes on a code edit but not a comment edit;
    * `assert_code_current` catches a file edited after import;
    * `resume_diff` reports the differences.
13. `tests/test_report.py`:
    * every refusal gate;
    * `supersedes` checked over the training lineage in a temporary git repo;
    * † rows, and `--final` refusing them;
    * the lockfile diff when a ref moves; wildcards expand to finished runs only.
14. `tests/test_e2e_tiny.py`: the end-to-end row of 6.9.
15. `tests/test_seeded_training.py` (`-m train`):
    * two strict CPU runs give equal weight hashes per checkpoint and equal `losses/*` and `environment/*` per epoch;
    * a different seed gives different results.
16. `tests/test_resume_equivalence.py` (`-m train`): K epochs straight vs K/2, kill, resume. Weights, LR sequence, scheduler `last_epoch`, optimizer step and RNG are equal, with 2 launch rows.

*P1:*

17. `tests/test_experiment_configs.py`: `validate_plan` passes on every `configs/experiments/*.yaml`.
18. `tests/test_golden_maps.py`: every generator with seeds 0–4, in two `PYTHONHASHSEED` subprocesses.
19. `tests/test_selftest_positions.py`: every packaged position applies through `apply_setup`, and its encodings at each level are golden.
20. `tests/test_vecenv_invariance.py` (needs E1, E2):
    * buffers are identical for threads {1, 4};
    * env i is identical for `num_envs` 8 and 16;
    * schedule mode reproduces a `ScheduledGame`.
21. `tests/test_eval_invariance.py`: a tiny graph net plus F and R on a 16-board suite. Rows are identical across `num_envs` {1, 5, 16}, threads {1, 4}, workers {1, 2}, interrupt/resume and rerun. Played ids, seats and map hashes equal the schedule.
22. `tests/test_search_invariance.py`: SearchPool visit counts are identical across pool size, batch size and threads.
23. `tests/test_golden_eval.py`: the committed `tiny-graph.pt` on 32 games gives pinned per-game outcomes for POLICY, `MCTS:sims=16` and `BELIEF:worlds=4` against F.
24. `tests/test_backend_parity.py`: per position, Catanatron state through `apply_setup` gives the same greedy POLICY action natively and as a `Player`.
25. `tests/test_catanatron_order.py`: port of v1's 08852ec. A seeded F-vs-R game in two `PYTHONHASHSEED` subprocesses gives identical logs, with seats read from `game.state.colors`.
26. `tests/test_fairness.py`: on 200 positions, the observer's private encoding is identical for tracker-sampled worlds and the true game.
27. `tests/test_smoke_experiments.py` (`-m train`): `--smoke` on the first variant of every experiment, every stage.
28. `tests/test_gpu_strict.py` (`-m gpu`): the CUDA graph matches eager argmax, and strict GPU eval is invariant as in test 21.
29. `tests/test_check_runs.py` (`-m runs`): `check_runs()` over the real runs dir.
30. `tests/test_llm_agent.py`: LLM specs (canonical fixed point, key dict, refusals, the native harness refused), the registry build, a stub game through `play_catanatron` vs F (rows, `llm` stats, replay from the seeds) and through `catanrl eval --harness catanatron` (record key, summary totals, cache), the opponent's cards never in the state text, answer parsing (index, markdown, text answers; garbage and out-of-range raise), retry then fallback with a scripted stub, characters/4 token accounting, the cost table arithmetic, and `llm-eval`'s stub cells (`run_variant` plays the F cell; the POLICY cell is deferred without the run).
31. `tests/test_predict_agent.py`: the claim (a `public`/`full` row is the private row plus the hidden features that level shows, of the other present seats, bitwise elsewhere; 2p and 4p, every seat, initial placement to robber and discard prompts, declared hands); the grid (features are `float32(count / total)` with the totals from the game's rules, incl. a rules variant; a prediction within half a step rounds back bitwise; clamps); oracle equivalence (true features in the head's place give exactly the level row and the POLICY agent's actions: argmax, sampled, rounded; `full` and `public` policies); present-only fill; VecEnv rows equal the reference stepper's and are invariant to `num_envs` and threads with PREDICT first or second; specs (canonical fixed point with both refs pinned, `key_dict` with the predictor, labels, `spec_ckpts`, deferred `${params}` syntax, refusals per pair, the server harness, a drifted aux head caught by the predictor's self-test); foreign runs of both checkpoints deferring a final; a whole Catanatron game; `catanrl eval` records (fair, `refs` + `predictors`, under the policy's run, cached, the catanatron harness).
32. `tests/test_mcts_reuse.py`: with `reuse=true` a seat's search is re-rooted at the action it played when the game followed exactly (the counters add up: `reused + fresh == decisions`, every later decision of a stream pops its live search), a seat's first decision drops the env's live searches (at most the seats the agent holds per env; a fill agent at two seats loses no reuse to it), a diverged game is refused before the engine is asked, a chance action is refused by the engine and the fresh search equals a plain one with the decision seed, streams are required and checked, an error empties the pool; the rows differ from `reuse=false` (why it is keyed) and are invariant to `num_envs`, threads and worker splits (`tests/test_eval_invariance.py`, `mcts-reuse*`).
33. `tests/test_dedup.py`: `DedupEvaluator` is bitwise the wrapped evaluator on repeated rows within and across batches (counters: rows, forwarded, batch and cache hits), hashes only where a merge can save a forward (`CATANRL_DEDUP_FORCE=1` always), bounds its cache, and MCTS chooses and visits the same with `dedup` on and off.
34. `tests/test_search_probe.py`: a probe report names every counter, zero or not; `--pair` scores a slot on the games two records share, across code and schedule (said in the report), and refuses another suite, other opponents or an empty intersection.

**Build order**

1. **P0:**
   * `canon`, `seeding`, the stats fix;
   * `config` without stages or fork;
   * `checkpoint` with the self-test;
   * `runs`, `provenance`, `_puffer` and `train/puffer`;
   * `agents/spec` (POLICY), `eval/suites`, `eval/results` (native);
   * tests 1–16.
2. **P1:**
   * stages, `fork`, `finish`;
   * MCTS and BELIEF specs;
   * E1–E3;
   * `report`, `verify`, `summarize`, `check-runs`;
   * tests 17–29.
3. **P2:** E4, `rerun --at-recorded-commit`, `import-v1`, the v1 stdio bridge, Protein.

**Changelog: reproducibility critique (R)**

| Point | Decision |
|---|---|
| H1: eval key misses code, maps and SearchConfig | **Accepted.** `code_id` is in the key from P0. `schedule_hash` covers map hashes. MCTS `key_dict` holds the full `SearchConfig`. `--final` re-plays 16 games per cited record, not 32, because MCTS records are costly. The version constants are dropped. |
| H2: a resumed eval mixes setups | **Accepted, for execution only**, since code is now in the key. Segment headers; `--allow-mixed` marks the record. |
| H3: orphaned checkpoints | **Accepted.** `trainer_state` is the commit point, orphans are moved, and epoch refs only reach committed checkpoints. |
| H4: resume across code versions | **Accepted, scoped** to the training `code_id` plus `uv.lock`. A whole-repo SHA would refuse after any unrelated commit. |
| H5: `supersedes` vs training lineage; unreachable SHAs | **Accepted.** The check walks every launch of the run and its ancestors, and keep refs make commits reachable. The eval-side scope is dropped as redundant with `code_id`. |
| H6: eval refs hashed unresolved | **Intent accepted, mechanism rejected.** Refs are bound at creation, never re-resolved, and moves warn. The hash keeps refs as written, so plan hashes stay pure for golden tests and fresh checkouts (T5). |
| M1: throttled logs | **Accepted.** |
| M2: select-eval contamination | **Accepted**, all four parts. |
| M3: dead or hidden PufferLib keys, reward clamp | **Accepted.** |
| M4: `dims: auto` depends on the seed | **Accepted:** generator bounds. |
| M5: long-lived processes | **Accepted:** one process per variant, and import-time hashes checked against the disk. |
| M6: the fixture records only dice; self-test should be in P0 | **Accepted, differently.** Positions are `apply_setup` payloads embedded in each checkpoint, so nothing is replayed. The self-test is in P0. |
| M7: wildcards are not pinned | **Accepted:** `results.json` pins expansions, finished runs only. |
| M8: Catanatron seats, parity test, bot seeds | **Accepted**, all three. |
| M9: `c_puct` units | **Accepted.** |
| L1, L2, L5, L7, L8, L9, L10 | **Accepted.** `require_fair` and `disjoint_from_training` leave the suite; `validate_plan` checks disjointness instead. |
| L3: partial package list | **List and digest accepted.** Diffing against `uv.lock` rejected: `uv run` syncs, and the lock sha is in the resume check. |
| L4: `ADVANTAGE_CUDA` and a stale `_C` | **Recording accepted.** A separate stale check rejected: the `_C` sha and the sources are in the training `code_id`. |
| L6: weak tests 16 and 22 | **Accepted.** Test 16 is dropped; test 22 is a same-machine guarantee. |
| Facts: PufferLib is patched, Muon compiles, cudnn.deterministic, the clamp, `global_step` rows, F has no RNG, the E4 list | **Accepted.** |
| v1 facts: selection on seed 67, the `dets=1` conflict, reshape bugs, fallback direction | **Accepted.** |

**Changelog: testability/ergonomics critique (T)**

| Point | Decision |
|---|---|
| 1: stale cache after a model or agent change | **Accepted**, merged with R-H1. The self-test runs in `resolve_agent`. Code is hashed from an AST snapshot rather than `git rev-parse HEAD:path`, so dirty trees are covered. |
| 2: new keys block resume | **Accepted:** `MIGRATIONS` plus a schema snapshot test. |
| 3: the time-limit habit stalls stages | **Accepted:** `time_limit` feeds dependents, `finish` ends a run, and LR is constant by default. `restore: full` with a recomputed T_max rejected as unneeded under constant LR. |
| 4: logging breaks test 22 | **Accepted.** |
| 5: smoke cannot run; no e2e test | **Accepted:** smoke overrides, epoch refs mapped to `@latest` in smoke, a default-tier e2e test, refs hashed as written. |
| 6: PufferLib import side effects | **Accepted.** |
| 7: `ppo.yaml` copies PufferLib's defaults | **Accepted.** |
| 8: a crash breaks write-once | **Accepted** (same as R-H3). |
| 9: engine identity too narrow | **Accepted:** the `.so` sha. The rollout fingerprint is dropped entirely rather than kept as a diagnostic. |
| 10: SHAs exist only locally | **Keep refs accepted.** A `--final` gate on remote reachability rejected: the superproject has no remote, so it would refuse everything. |
| 11: binomial overflow | **Accepted.** |
| 12: non-outcome key fields; the game name in seeds | **Accepted.** |
| 13: ambiguous fairness gate | **Accepted:** the gate applies to slots under test. `oracle=` for Catanatron bots on views is deferred to arena work. |
| 14: goldens will be bypassed; flaky canary | **Accepted:** frozen fixture YAMLs, and the canary is cut. Keying goldens by engine identity rejected: with no version constants, a re-pin is just a reviewed diff. |
| 15: small items | **Accepted:** fill (as a `fill:` key, not `*` grammar), `params` typos, stale build warns until E3, `fast` dev evals in the key. A new wildcard character rejected: quote patterns in zsh; YAML is unaffected. |
| Cuts | **Partly accepted.** `pins.lock` is folded into `results.json`. Dropped: `citations.json`, the encoder fingerprint, the version constants, the canary, `require_fair`, `_reruns/`, `_v1/`. **Kept:** `supersedes`, because training-side bugs are exactly bcea78e and `code_id` cannot see them; and †/`--final`, about 15 lines that keep dirty numbers out of the paper. |
| Facts: PufferLib patches, EncoderDrift vs reshape bugs, anchor uses official spiral, anchor is a new recipe, Muon default, line drift | **Accepted.** Cite by symbol. |

**Changes to the synthesized design's conflict resolutions**

| # | Change |
|---|---|
| 4 | `fast` is allowed on dev suites, and determinism is part of the key. |
| 7 | The fixture is replaced by `apply_setup` positions embedded in each checkpoint. |
| 9 | The fairness gate applies only to slots under test. |
| 13 | A stale build warns until E3 lands. |
| — | Dropped: `obs_spec.py`, `ENCODER_REVISION`, `AGENT_VERSIONS`, `HARNESS_VERSION`, `SCHEDULE_VERSION`, the engine rollout fingerprint, `pins.lock`, `citations.json`, test 16 and the canary. |