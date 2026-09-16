# Phase C4-A Failure Attribution

Outcome A

This report records diagnostic associations and matched counterfactual comparisons; it does not automatically establish a causal mechanism.

## Observed association

- `design_geometry`: seed 7: rank 130, seed 17: rank 130, seed 29: rank 130
- `task_relevant_target_projection`: seed 7: shuffled parallel ratio 0.259900, seed 17: shuffled parallel ratio 0.293983, seed 29: shuffled parallel ratio 0.280757
- `bias_action_shortcut`: Registered normal and shuffled readouts were decomposed into hidden, bias, and action-block contributions.
- `local_block_structure`: block10 summaries: seed 7: mean 156.406, seed 17: mean 162.938, seed 29: mean 161.219
- `evaluation_fixture_sensitivity`: seed 7: registered shuffled secondary [148, 142, 145, 142, 152, 135, 138, 151], seed 17: registered shuffled secondary [173, 174, 174, 173, 167, 173, 166, 169], seed 29: registered shuffled secondary [146, 146, 157, 159, 167, 159, 156, 166]
- `unresolved_multiple_mechanisms`: D1-D4 evidence is retained jointly for all three registered seeds.

## Matched counterfactual evidence

- `design_geometry`: Normal and shuffled registered conditions use the same D0 design bytes.
- `task_relevant_target_projection`: The matched permutation grid records target projection and supervised-direction alignment for every replicate.
- `bias_action_shortcut`: The decompositions reuse the fitted readouts and do not retrain ablated models.
- `local_block_structure`: global summaries: seed 7: mean 100.531, seed 17: mean 99.656, seed 29: mean 95.969
- `evaluation_fixture_sensitivity`: Every permutation and registered readout is evaluated on the original plus eight fixed secondary sets.
- `unresolved_multiple_mechanisms`: No diagnostic row is discarded because its score or association is inconvenient.

## Interpretation

- `design_geometry`: Design geometry is explanatory context and is not sufficient by itself.
- `task_relevant_target_projection`: Observed alignment indicates task-relevant structure under the frozen design without establishing a causal source.
- `bias_action_shortcut`: Component differences are post-hoc associations rather than independent interventions.
- `local_block_structure`: Block10/global differences are matched diagnostic associations, not a pure causal intervention.
- `evaluation_fixture_sensitivity`: Cross-fixture variation describes evaluation-realization sensitivity without changing the frozen C4-A verdict.
- `unresolved_multiple_mechanisms`: Multiple or unresolved mechanisms remain valid interpretations for later no-refit review.

### Review rationale

- D0 integrity is valid and the registered normal/shuffled replays reproduce the frozen C4-A scores.
- Across the fixed grid, block10 permutations score at least 150 in 84 of 96 rows, while global permutations do so in 0 of 96 rows.
- The block10-to-global score separation is accompanied by lower target/design parallel ratios under global permutations; this is matched diagnostic association rather than a single-cause claim.
- Secondary evaluation sets do not show a general single-fixture collapse, especially for seed 17; design geometry and readout decomposition remain contextual evidence.
- The reviewed interpretation is therefore Outcome A: residual negative-control predictivity is strongly associated with local block-preserving reward structure under the tested frozen operator.
