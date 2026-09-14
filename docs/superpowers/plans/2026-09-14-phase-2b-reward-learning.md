# Phase 2B Online Reward Learning Implementation Plan

> For agentic workers: execute this plan task by task with RED-GREEN-REFACTOR.
> Do not combine tasks, weaken a scientific gate, or modify a frozen module.

**Goal:** Prove or falsify that a local online softmax eligibility rule, driven
only by decision-time recurrent activity, sampled action probabilities, and a
later scalar reward, can learn the delayed-cue action mapping that Phase 2A
showed is linearly available.

**Architecture:** Keep RecurrentPolicy as a frozen recurrent state source and
keep its existing output/readout path as the immutable V0 baseline. Add an
independent RewardModulatedReadout with zero-initialized weights and biases,
legal-action softmax selection, one pending eligibility, and scalar-reward
updates. Compose it with the existing DelayedCueTask in a separate Phase 2B
experiment module containing matched normal, reset, and shuffled-reward
controls.

**Tech stack:** Python 3.10+, NumPy, pytest, Ruff, setuptools, GitHub Actions.

**Approved spec:**
docs/superpowers/specs/2026-09-14-phase-2b-reward-learning-design.md

## Global constraints

- Work only on the standalone branch experiment/neural-state-machine.
- Do not merge, rebase onto, copy from, or modify master.
- NumPy remains the only runtime dependency.
- Do not modify policy.py, controller.py, memory_task.py,
  memory_benchmark.py, or memory_probe.py.
- Do not modify existing Phase 1, V0 reward-baseline, or Phase 2A tests.
- The recurrent input, recurrent, and legacy output matrices remain frozen.
- The new learner may receive only a numeric hidden vector, legal action
  indices, a seeded action RNG at training selection, and a later scalar
  reward.
- Cue, correct action, delay length, episode phase, observation history, and
  probe predictions never enter the policy or readout.
- Do not add BPTT, supervised labels, an RL framework, an explicit state
  variable, or semantic behavior transitions.
- Acceptance defaults are fixed before execution: hidden size 64, recurrent
  radius 0.9, learning rate 0.05, temperature 1.0, 2,000 training episodes,
  20 evaluation blocks, and seeds 7, 17, and 29.
- Preserve the 0.90 overall, 0.85 per-delay, exact 0.50 reset, and
  shuffled-control gates. A miss is an experiment failure and design-review
  stop, not permission to tune the committed protocol.
- Every production behavior begins with a focused failing test.
- Commit after every task with the exact subject specified by that task.
- Publish every completed commit to the same experiment branch. Never leave a
  completed task only in an unpushed local history.

## File and responsibility map

| File | Responsibility | Change |
|---|---|---|
| src/neural_state_machine/reward_readout.py | Softmax selection, eligibility, scalar-reward update, digest | Create |
| tests/test_reward_readout.py | Readout validation, numerics, lifecycle, immutability | Create |
| src/neural_state_machine/reward_learning.py | Phase 2B fixtures, training, controls, evaluation, result types | Create |
| tests/test_reward_learning.py | Boundary, frozen-system, determinism, scientific gates | Create |
| scripts/benchmark_reward_learning.py | Stable JSON benchmark and exit status | Create |
| src/neural_state_machine/__init__.py | Export approved public Phase 2B API | Modify |
| README.md | Record Phase 2B protocol, evidence, and interpretation | Modify |
| .github/workflows/ci.yml | Run the new benchmark in the existing Python matrix | Modify |

---

## Task 1: Add the zero-initialized reward-readout foundation

**Files:**

- Create: src/neural_state_machine/reward_readout.py
- Create: tests/test_reward_readout.py
- Test: tests/test_reward_readout.py

**Interfaces introduced:**

~~~python
@dataclass(frozen=True)
class RewardReadoutDecision:
    action_index: int
    logits: np.ndarray
    probabilities: np.ndarray

class RewardModulatedReadout:
    def __init__(
        self,
        hidden_size: int,
        action_count: int,
        learning_rate: float = 0.05,
        temperature: float = 1.0,
    ) -> None: ...
~~~

**Steps:**

1. Write an import/construction test before the module exists. Require
   hidden_size and action_count public attributes, learning_rate 0.05,
   temperature 1.0, weight shape (action_count, hidden_size), bias shape
   (action_count,), and exact zero initialization.
2. Add parametrized RED tests rejecting hidden sizes 0, -1, True, 2.5, and
   None; action counts 0, 1, True, 2.5, and None.
3. Add RED tests rejecting learning rates 0, negative, NaN, infinities, True,
   strings, and None.
4. Add RED tests rejecting temperatures 0, negative, NaN, infinities, True,
   strings, and None.
5. Run:

~~~bash
.venv/bin/pytest tests/test_reward_readout.py -q
~~~

   Expected RED: ModuleNotFoundError for reward_readout.
6. Implement only the frozen decision dataclass, constructor validation,
   exact-zero float64 weight/bias arrays, public scalar configuration
   attributes, and empty pending-eligibility fields. Do not implement
   selection or learning yet.
7. Test that constructor input objects cannot alias the parameter arrays and
   that each instance owns independent matrices.
8. Run the focused test and Ruff:

~~~bash
.venv/bin/pytest tests/test_reward_readout.py -q
.venv/bin/ruff check src/neural_state_machine/reward_readout.py tests/test_reward_readout.py
~~~

9. Commit only these two files:

~~~bash
git add src/neural_state_machine/reward_readout.py tests/test_reward_readout.py
git commit -m "feat: add reward readout foundation"
~~~

---

## Task 2: Implement legal-action softmax and greedy evaluation

**Files:**

- Modify: src/neural_state_machine/reward_readout.py
- Modify: tests/test_reward_readout.py
- Test: tests/test_reward_readout.py

**Interface completed:**

~~~python
select_greedy(
    hidden_state: np.ndarray,
    legal_action_indices: object,
) -> RewardReadoutDecision
~~~

**Steps:**

1. Add RED validation tests for scalar, rank-two, wrong-length,
   non-convertible, NaN, and infinite hidden inputs. Invalid input must not
   change parameters or pending-feedback state.
2. Add RED legal-mask tests for empty, duplicate, boolean, float, negative,
   and out-of-range indices.
3. Add a zero-parameter test with action_count 3 and legal indices (2, 0).
   Require logits [0.0, -inf, 0.0], probabilities [0.5, 0.0, 0.5], and action
   index 0 despite the reversed legal-index order.
4. Add an exact nonzero test by setting literal private weight/bias fixtures,
   calculating the stabilized legal-only softmax independently in the test,
   and comparing logits/probabilities with zero relative tolerance and
   1e-12 absolute tolerance.
5. Add a large-logit stability test proving all legal probabilities are
   finite, sum to exactly 1 within 1e-15, illegal probabilities are zero, and
   no overflow warning occurs.
6. Add snapshot tests proving returned logits/probabilities are independent
   read-only float64 arrays and remain unchanged after later selections.
7. Run the focused test and confirm RED because select_greedy is absent.
8. Implement shared hidden/mask validation, stabilized temperature-scaled
   softmax, immutable snapshots, and lowest-index greedy tie behavior.
   select_greedy must never create or consume eligibility.
9. Run focused tests and Ruff.
10. Commit:

~~~bash
git add src/neural_state_machine/reward_readout.py tests/test_reward_readout.py
git commit -m "feat: add legal-action softmax selection"
~~~

---

## Task 3: Add seeded training selection and one-shot eligibility

**Files:**

- Modify: src/neural_state_machine/reward_readout.py
- Modify: tests/test_reward_readout.py
- Test: tests/test_reward_readout.py

**Interfaces completed:**

~~~python
@property
def has_pending_feedback(self) -> bool: ...

def select_for_training(
    self,
    hidden_state: np.ndarray,
    legal_action_indices: object,
    rng: np.random.Generator,
) -> RewardReadoutDecision: ...
~~~

**Steps:**

1. Add RED tests requiring a NumPy Generator and rejecting RandomState, int,
   None, and arbitrary objects before any pending state is created.
2. Add a same-seed test using two separately constructed readouts and
   generators. Require the single sampled action to match and be legal.
   Multi-step sampled-sequence repeatability is added after learn can consume
   each pending eligibility in Task 4.
3. Add a three-action masked test proving an illegal action is never sampled
   and receives zero probability.
4. Add a RED lifecycle test:

   - has_pending_feedback starts false;
   - one valid training selection changes it to true;
   - a second training selection raises RuntimeError;
   - the failure leaves the original pending eligibility intact;
   - greedy selection remains evaluation-only and does not create pending
     feedback on a fresh readout;
   - greedy selection while training feedback is pending does not consume or
     replace the pending record.

5. Add an aliasing test: mutate the original hidden input after selection and
   prove the stored eligibility inputs are unaffected.
6. Run the focused test and confirm RED because the training API is absent.
7. Implement categorical sampling over the legal probability vector.
   Store independent copies of:

~~~text
E_weights = (one_hot(selected) - probabilities) outer hidden
E_bias = one_hot(selected) - probabilities
~~~

8. Do not implement parameter updates in this task. Pending eligibility is an
   opaque one-decision record.
9. Run focused tests and Ruff.
10. Commit:

~~~bash
git add src/neural_state_machine/reward_readout.py tests/test_reward_readout.py
git commit -m "feat: add one-shot reward eligibility"
~~~

---

## Task 4: Apply scalar reward to the pending eligibility

**Files:**

- Modify: src/neural_state_machine/reward_readout.py
- Modify: tests/test_reward_readout.py
- Modify: src/neural_state_machine/__init__.py
- Test: tests/test_reward_readout.py

**Interfaces completed:**

~~~python
def learn(self, reward: float) -> None: ...
def parameter_digest(self) -> str: ...
~~~

**Steps:**

1. Add RED tests that learn before selection and a second learn after
   consumption raise RuntimeError.
2. Add a literal numerical test. Use a known hidden vector and zero
   parameters, call select_for_training, build the expected one-hot-minus-p
   term from the returned selected action and probabilities, then require:

~~~text
expected_weights = learning_rate * reward
                   * outer(one_hot - probabilities, hidden)
expected_biases = learning_rate * reward
                  * (one_hot - probabilities)
~~~

   Compare every element with zero relative tolerance and 1e-12 absolute
   tolerance.
3. Add positive and negative reward tests proving the selected action's margin
   respectively rises and falls when replayed on the same hidden vector.
4. Add clipping tests proving +100 is byte-identical to +1 and -100 is
   byte-identical to -1 from independently reconstructed selections.
5. Add a zero-reward test proving eligibility is consumed while the parameter
   digest remains unchanged.
6. Add non-numeric, NaN, and infinity tests. Invalid reward must leave a valid
   pending eligibility available for a later valid learn.
7. Add a deterministic multi-step action test using two independent readouts
   and same-seed generators. Consume each pending eligibility with learn(0.0);
   require identical complete action sequences containing only legal actions.
8. Define parameter_digest as SHA-256 over explicit weight shape, contiguous
   float64 weight bytes, bias shape, and contiguous float64 bias bytes. Add an
   independent digest reconstruction test.
9. Add a multi-action test proving a negative reward changes every legal row
   according to its probability but leaves illegal rows unchanged.
10. Run focused tests and confirm RED for learn and digest.
11. Implement finite validation, clipping to [-1, 1], the exact two parameter
    updates, and one-shot consumption. No reward history or semantic state may
    be stored.
12. Export RewardReadoutDecision and RewardModulatedReadout from package
    __init__.py. Add an export identity test.
13. Run:

~~~bash
.venv/bin/pytest tests/test_reward_readout.py tests/test_policy.py tests/test_phase1_compatibility.py -q
.venv/bin/ruff check src/neural_state_machine/reward_readout.py src/neural_state_machine/__init__.py tests/test_reward_readout.py
~~~

14. Commit:

~~~bash
git add src/neural_state_machine/reward_readout.py src/neural_state_machine/__init__.py tests/test_reward_readout.py
git commit -m "feat: add reward-modulated readout learning"
~~~

---

## Task 5: Build the frozen-state Phase 2B evaluation protocol

**Files:**

- Create: src/neural_state_machine/reward_learning.py
- Create: tests/test_reward_learning.py
- Test: tests/test_reward_learning.py

**Interfaces introduced:**

~~~python
@dataclass(frozen=True)
class RewardLearningConfig:
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    learning_rate: float = 0.05
    temperature: float = 1.0
    training_episodes: int = 2_000
    evaluation_blocks: int = 20
~~~

Internal helpers include _balanced_cases, _build_fixtures, _decision_hidden,
and _evaluate.

**Steps:**

1. Add RED config tests requiring positive integer hidden_size and
   evaluation_blocks; a positive multiple-of-ten training_episodes; radius in
   [0, 1); positive finite learning_rate and temperature; and strict boolean
   rejection.
2. Add RED balanced-case tests. Every shuffled ten-case block must contain
   each Cue x delay pair exactly once; same-seed generators produce the same
   order.
3. Add a fixture test requiring fresh distractors, immutable arrays, 200
   default evaluation episodes, exactly 100 labels per action, and exactly 40
   cases per delay.
4. Add a recording-policy test for _decision_hidden. Require this exact call
   sequence:

~~~text
reset_state
advance(cue stimulus)
advance(each delay stimulus in order)
optional reset_state
advance(shared decision stimulus)
~~~

   Only the returned numeric hidden vector proceeds to the readout.
5. Add a recording-readout evaluation test proving _evaluate calls only
   select_greedy(hidden, (0, 1)); it never calls select_for_training or learn.
6. Add a real-policy reset-ablation test. All reset hidden vectors must be
   byte-identical, every choice must be one constant action, and the balanced
   result must be exactly 100/200 overall and 20/40 at every delay.
7. Snapshot policy input, recurrent, and legacy output matrices plus the new
   readout digest around both evaluation modes. Require byte equality and no
   pending feedback.
8. Run focused tests and confirm RED because reward_learning is absent.
9. Implement only config, fixtures, hidden collection, immutable integer-count
   evaluation results, and choice digests. Do not implement training.
10. Reuse AccuracyCount from memory_benchmark without modifying that frozen
    module.
11. Run focused tests and Ruff.
12. Commit:

~~~bash
git add src/neural_state_machine/reward_learning.py tests/test_reward_learning.py
git commit -m "feat: add phase two-b evaluation protocol"
~~~

---

## Task 6: Add normal reward training and the shuffled-reward control

**Files:**

- Modify: src/neural_state_machine/reward_learning.py
- Modify: tests/test_reward_learning.py
- Test: tests/test_reward_learning.py

**Interfaces introduced internally:**

~~~text
_train_normal(...)
_train_shuffled(...)
_run_once(seed, config)
~~~

**Steps:**

1. Build the complete immutable 2,000-episode training fixture tuple once from
   SeedSequence([seed, 0x54524149]). Reuse the exact episode objects for normal
   and shuffled-control runs.
2. Add a RED test proving the training tuple contains 200 balanced blocks,
   fresh distractors, and no dependence on evaluation RNG draws.
3. Add a boundary-spy RED test for normal training. For each episode, require
   only:

~~~text
policy reset/advance calls
readout.select_for_training(hidden, (0, 1), action_rng)
task.reward(episode, sampled action)
readout.learn(returned scalar)
~~~

   Assert that readout methods never receive cue, correct_action_index,
   delay_steps, phase, or the episode object.
4. Add an independent reference loop for a 20-episode configuration. Compare
   the implementation's ordered sampled actions, rewards, total reward, final
   ten-case count, and final digest exactly.
5. Add shuffled-control RED tests:

   - every ten-case block supplies exactly five +1 and five -1 values;
   - order comes only from the dedicated shuffled_reward_rng;
   - a task double whose reward method raises is accepted, proving shuffled
     training never calls task.reward;
   - rewards are not derived from cue, correct action, delay, or sampled
     action;
   - normal and shuffled runs own separate policy, readout, and action RNG
     objects.

6. Use these fixed lineages:

~~~python
training_fixture_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x54524149])
)
training_action_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x4143544E])
)
evaluation_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x4556414C])
)
shuffled_reward_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x53485546])
)
shuffled_action_rng = np.random.default_rng(
    np.random.SeedSequence([seed, 0x53414354])
)
~~~

7. In _run_once, construct independent but identically configured normal and
   shuffled state sources and readouts. Evaluate pre-training, train both,
   evaluate normal recurrent/reset and shuffled recurrent on the same
   evaluation fixtures, then compute all counts and digests.
8. Snapshot all three matrices of each RecurrentPolicy before training and
   compare with np.array_equal afterward. Require only the new readout digest
   to change.
9. Require no readout to retain pending feedback after training or evaluation.
10. Run focused tests and confirm RED before implementing each helper.
11. Implement the two training paths and one-run assembly without public
    acceptance wrappers.
12. Run focused tests and Ruff.
13. Commit:

~~~bash
git add src/neural_state_machine/reward_learning.py tests/test_reward_learning.py
git commit -m "feat: add reward and shuffled-control training"
~~~

---

## Task 7: Add the repeatable three-seed acceptance benchmark

**Files:**

- Modify: src/neural_state_machine/reward_learning.py
- Modify: tests/test_reward_learning.py
- Create: scripts/benchmark_reward_learning.py
- Modify: src/neural_state_machine/__init__.py
- Test: tests/test_reward_learning.py
- Test: scripts/benchmark_reward_learning.py

**Public interfaces introduced:**

~~~python
@dataclass(frozen=True)
class RewardLearningResult: ...

def run_reward_learning_experiment(
    seed: int = 7,
    config: RewardLearningConfig | None = None,
) -> RewardLearningResult: ...

def run_reward_learning_benchmark(
    seeds: Sequence[int] = (7, 17, 29),
    config: RewardLearningConfig | None = None,
) -> dict[str, object]: ...
~~~

**Steps:**

1. Define frozen public result types containing literal integer counts for
   pre-training, post-training, state reset, shuffled control, per-delay
   normal evaluation, total training rewards, final blocks, parameter digests,
   reset per-delay counts, matrix-control digests, ordered-choice digests,
   and repeatability.
2. Validate seed inputs before constructing any policy or RNG. Reject empty or
   duplicate benchmark seeds and reject booleans.
3. Implement run_reward_learning_experiment by calling _run_once twice with
   entirely reconstructed objects. Set repeatable from full immutable result
   equality; never compare an object with itself.
4. Add same-seed tests requiring independent public calls to return equal
   dataclasses and identical digests.
5. Add the pre-registered acceptance test:

~~~python
@pytest.mark.parametrize("seed", [7, 17, 29])
def test_phase_two_b_acceptance(seed):
    result = run_reward_learning_experiment(seed)
    assert result.repeatable is True
    assert result.post_training.total == 200
    assert result.post_training.correct >= 180
    assert all(score.correct >= 34 for _, score in result.per_delay)
    assert result.state_reset == AccuracyCount(100, 200)
    assert all(
        score == AccuracyCount(20, 40)
        for _, score in result.reset_per_delay
    )
    assert result.all_reset_hidden_equal is True
    assert result.shuffled_control.correct < 150
~~~

6. Run only this acceptance test before writing the JSON CLI. If any seed
   misses any gate, stop. Preserve and report its exact counts/digests; do not
   change learning rate, temperature, seed, task, reservoir, threshold, or
   sample count.
7. Implement top-level pooled shuffled-control gating across 600 evaluations.
   Require pooled accuracy in [0.40, 0.60] and each seed below 150/200.
8. Convert results into one stable JSON object. Use sorted keys and compact
   separators. Include phase "2B", configuration, all literal counts and
   accuracies, every digest, per-seed pass flags, shuffled pooled counts, and
   all_passed.
9. Add a subprocess test that runs the CLI twice, requires byte-identical
   stdout, parses one JSON line, verifies seeds [7, 17, 29], and checks exit
   status equals the all_passed field.
10. Export RewardLearningConfig, RewardLearningResult,
    RewardModulatedReadout, RewardReadoutDecision,
    run_reward_learning_experiment, and run_reward_learning_benchmark.
11. Run:

~~~bash
.venv/bin/pytest tests/test_reward_readout.py tests/test_reward_learning.py -q
.venv/bin/python scripts/benchmark_reward_learning.py
.venv/bin/ruff check src/neural_state_machine/reward_readout.py src/neural_state_machine/reward_learning.py src/neural_state_machine/__init__.py tests/test_reward_readout.py tests/test_reward_learning.py scripts/benchmark_reward_learning.py
~~~

12. Commit only after the scientific gate passes. If it fails, commit the
    truthful failing experiment evidence only after returning to design
    review.

~~~bash
git add src/neural_state_machine/reward_learning.py src/neural_state_machine/__init__.py tests/test_reward_learning.py scripts/benchmark_reward_learning.py
git commit -m "feat: add phase two-b acceptance benchmark"
~~~

---

## Task 8: Document evidence, wire CI, and perform the final audit

**Files:**

- Modify: README.md
- Modify: .github/workflows/ci.yml
- Modify only if an already demonstrated implementation defect is found:
  new Phase 2B files and tests
- Test: complete repository

**Steps:**

1. Add a README Phase 2B section with:

   - the distinction among Phase 2A memory, V0 reward failure, and the new
     Phase 2B learner;
   - the exact softmax eligibility equation in prose or math;
   - fixed configuration and three acceptance seeds;
   - normal, reset, and shuffled-reward result tables;
   - the command python scripts/benchmark_reward_learning.py;
   - an explicit statement that only the new readout weights/biases learn;
   - the narrow interpretation boundary and Phase 3 stop.

2. Append a Phase 2B reward-learning benchmark step to the existing CI job.
   Retain unit tests, Ruff, Phase 1 benchmark, and Phase 2A probe unchanged.
3. Run the no-hidden-label audit:

~~~bash
rg -n "correct_action_index|[.]cue|delay_steps|episode_phase|current_phase|remembered_cue|last_cue|behavior_state|transition_table|probe" src/neural_state_machine/reward_readout.py
~~~

   Expected: no matches. In reward_learning.py, label/task metadata may appear
   only in fixture construction and post-prediction scoring; inspect every
   match manually.
4. Inspect the training boundary directly. Confirm that readout training calls
   contain only hidden vectors, legal numeric indices, RNG, and scalar reward.
5. Prove frozen artifacts stayed unchanged relative to this plan commit's
   parent:

~~~bash
plan_commit=$(git log -1 --format=%H -- docs/superpowers/plans/2026-09-14-phase-2b-reward-learning.md)
plan_parent=$(git rev-parse "$plan_commit^")
git diff --exit-code "$plan_parent"..HEAD -- src/neural_state_machine/policy.py src/neural_state_machine/controller.py src/neural_state_machine/memory_task.py src/neural_state_machine/memory_benchmark.py src/neural_state_machine/memory_probe.py tests/test_policy.py tests/test_phase1_compatibility.py tests/test_memory_task.py tests/test_memory_benchmark.py tests/test_memory_probe.py
~~~

6. Run complete verification in this order:

~~~bash
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/benchmark_memory_probe.py
.venv/bin/python scripts/benchmark_reward_learning.py
git diff --check
git status --short --branch
~~~

7. Confirm Phase 1 JSON remains byte-for-byte identical to its frozen output.
8. Confirm Phase 2A still reports 200/200 and every delay 40/40 for all three
   seeds, reset 100/200, unchanged legacy output digests, and repeatability.
9. Confirm Phase 2B reports all normal/reset/shuffled gates from the approved
   spec without rounding away literal counts.
10. Commit documentation and CI:

~~~bash
git add README.md .github/workflows/ci.yml
git commit -m "docs: report phase two-b reward learning"
~~~

11. Publish the exact commit chain to
    origin/experiment/neural-state-machine without force and without touching
    master.
12. Verify the remote branch tree equals the local tree and no merge base with
    origin/master exists.
13. Inspect GitHub Actions for the exact pushed SHA. Require all Python 3.10,
    3.11, and 3.12 jobs to pass the complete suite and all three benchmarks.
14. If CI differs from local results, use systematic debugging and add one
    focused portability-fix commit. Do not weaken any scientific criterion.

## Completion handoff

The final report must include:

- exact final remote SHA and experiment branch URL;
- every implementation commit and subject;
- complete pytest count and Ruff result;
- unchanged Phase 1 JSON;
- unchanged Phase 2A counts and digests;
- Phase 2B normal counts for each seed and each delay;
- exact reset counts and reset-state equality;
- individual and pooled shuffled-reward counts;
- normal/shuffled training totals and parameter digests;
- frozen reservoir and legacy-readout evidence;
- exact independent-run repeatability;
- exact-head GitHub Actions URL;
- a truthful failure conclusion if any gate misses.

A passing Phase 2B result permits only a new Phase 3 design session. It does not
authorize implementation of multiple concurrent memories, conflicting game
objectives, fly-brain integration, or stronger biological claims.
