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
              variant: str | None = None, smoke: bool = False) -> Plan   # smoke: configs/smoke.yaml + epoch refs → @latest (recorded)
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
def register(registry: Registry = REGISTRY) -> None         # idempotent; on import of catanrl.agents
def resolve_agent(spec: "str | dict | ResolvedAgent", *, harness: Literal["native", "catanatron", "server"] = "native",
                  games: Sequence[GameConfig] = ()) -> "ResolvedAgent"   # raises catanatron.registry.SpecError
@dataclass(frozen=True)
class ResolvedAgent:
    key: str; impl: str; params: Any; checkpoint: CheckpointRef | None
    fair: bool; label: str; canonical: str
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
    def save_checkpoint(self) -> str: ...                   # writer.save_checkpoint → trainer_state → run.json → select eval → alias
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
* **Checkpoint hook.** PufferLib's `save_checkpoint` returns early if the file exists (lines 544-545). It is fully overridden; nothing of its directory layout is used.
* **Batch sizes.** `batch_size >= minibatch_size` is enforced (line 127). `configs/smoke.yaml` sets Adam, `bptt_horizon: 8`, `minibatch_size` equal to the batch, and compile off.
* **Spec parsing.** `registry.parse_spec` splits on every `:` and `build_all` caps games at 2–4 players. Cite these by symbol, since submodule line numbers drift.
* **MCTS mapping.** The engine's `SearchConfig.determinizations = 0` means the game as given. Map `oracle=true` to 0 and `dets=K` to K. `c_puct` defaults to 0.75 in engine units.
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
   * execution params are out of `key_dict`; MCTS `key_dict` holds the full `SearchConfig`;
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