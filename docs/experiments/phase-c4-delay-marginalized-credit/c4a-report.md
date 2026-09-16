# Phase C4-A Registered Result

- Scientific head: `8ae3154950ed53c4d0a0f555463042ff72674d31`
- Formal head: `8b2180ed24b6ff03db4b927ec29cdd9903b0ccac`
- Measurement workflow run: `35105578412`
- Manifest SHA-256: `a990745cf6f0e8c5fd7ae5d9a7189737b30a40168a4db3df1a6d3043d6d5876a`
- Result SHA-256: `7f398f474dba928d2a6aa993d0801563aa05af20faa8cd9a61ec94a6ca4697a4`
- Provenance SHA-256: `8fb47b5e670754bc44c9ea70dd4cedce40e6f52165923142114577b4aec00351`

## Registered verdict

- `formal_valid = true`
- `protocol_valid = true`
- `operator_passed = false`

| Seed | Normal | Reset | Shuffled | Seed gate |
| ---: | ---: | ---: | ---: | :--- |
| 7 | 200/200 | 100/200 | 147/200 | PASS |
| 17 | 200/200 | 100/200 | 170/200 | FAIL |
| 29 | 200/200 | 100/200 | 160/200 | FAIL |

All three normal evaluations are `200/200`, every normal per-delay cell is `40/40`, and every reset condition is exactly `100/200` with `20/40` in each delay cell. The registered failure is the shuffled negative control: seed 7 is `147/200` and passes the `<150` threshold, while seed 17 is `170/200` and seed 29 is `160/200`.

Under the registered stop rule, C4-A therefore does **not** establish operator sufficiency. The failure is specifically lack of negative-control specificity rather than failure to recover the normal task. C4-B is not authorized and must not be run from this result.

The eight secondary evaluation sets remain diagnostic only and were not used for fitting, selection, or gate decisions.
