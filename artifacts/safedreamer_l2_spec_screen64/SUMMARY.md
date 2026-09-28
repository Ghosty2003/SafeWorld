# 64-step exploratory specification screening

Development data only. No new V, confidence bound, or warrant. All distances are decoded.

| Specification / initial scope | Training | Development | Initially in target (train/dev) |
|---|---:|---:|---:|
| `F[0,64](d<0.3); all` | 166/300 | 55/100 | 0/0 |
| `F[0,64](d<0.3); initial_outside` | 166/300 | 55/100 | 0/0 |
| `F[0,64](d<0.5); all` | 231/300 | 69/100 | 0/0 |
| `F[0,64](d<0.5); initial_outside` | 231/300 | 69/100 | 0/0 |
| `F[0,64](d<1); all` | 254/300 | 78/100 | 0/0 |
| `F[0,64](d<1); initial_outside` | 254/300 | 78/100 | 0/0 |
| `F[0,64](d<1.25); all` | 261/300 | 84/100 | 85/27 |
| `F[0,64](d<1.25); initial_outside` | 176/215 | 57/73 | 0/0 |
| `F[0,64](d<1.5); all` | 278/300 | 92/100 | 171/51 |
| `F[0,64](d<1.5); initial_outside` | 107/129 | 41/49 | 0/0 |
| `F[0,64](d<1.75); all` | 285/300 | 97/100 | 236/83 |
| `F[0,64](d<1.75); initial_outside` | 49/64 | 14/17 | 0/0 |
| `F[0,64](d<2); all` | 295/300 | 98/100 | 272/94 |
| `F[0,64](d<2); initial_outside` | 23/28 | 4/6 | 0/0 |
| `F[0,64](d<2.25); all` | 300/300 | 100/100 | 298/99 |
| `F[0,64](d<2.25); initial_outside` | 2/2 | 1/1 | 0/0 |
| `F[0,64](d<2.5); all` | 300/300 | 100/100 | 300/100 |
| `F[0,64](d<2.5); initial_outside` | 0/0 | 0/0 | 0/0 |
| `F[1,64](d<0.3); 1<=d0<1.25` | 61/85 | 14/27 | 0/0 |
| `F[1,64](d<0.5); 1<=d0<1.25` | 71/85 | 21/27 | 0/0 |
| `F[1,64](d<1); 1<=d0<1.25` | 81/85 | 25/27 | 0/0 |
| `F[1,64](d<0.3); 1<=d0<1.5` | 113/171 | 27/51 | 0/0 |
| `F[1,64](d<0.5); 1<=d0<1.5` | 140/171 | 36/51 | 0/0 |
| `F[1,64](d<1); 1<=d0<1.5` | 154/171 | 42/51 | 0/0 |
| `F[1,64](d<0.3); 1<=d0<1.75` | 140/236 | 48/83 | 0/0 |
| `F[1,64](d<0.5); 1<=d0<1.75` | 190/236 | 61/83 | 0/0 |
| `F[1,64](d<1); 1<=d0<1.75` | 211/236 | 69/83 | 0/0 |
| `F[1,64](d<0.3); 1.25<=d0<1.75` | 79/151 | 34/56 | 0/0 |
| `F[1,64](d<0.5); 1.25<=d0<1.75` | 119/151 | 40/56 | 0/0 |
| `F[1,64](d<1); 1.25<=d0<1.75` | 130/151 | 44/56 | 0/0 |
| `F[1,64](d<0.3); 1.5<=d0<2` | 42/101 | 26/43 | 0/0 |
| `F[1,64](d<0.5); 1.5<=d0<2` | 73/101 | 31/43 | 0/0 |
| `F[1,64](d<1); 1.5<=d0<2` | 82/101 | 34/43 | 0/0 |
| `F[1,64](d<0.3); 1.75<=d0<2.5` | 26/64 | 7/17 | 0/0 |
| `F[1,64](d<0.5); 1.75<=d0<2.5` | 41/64 | 8/17 | 0/0 |
| `F[1,64](d<1); 1.75<=d0<2.5` | 43/64 | 9/17 | 0/0 |
| `F[1,64](d<d0-0.01)` | 276/300 | 94/100 | 0/0 |
| `F[1,64](d<d0-0.05)` | 276/300 | 92/100 | 0/0 |
| `F[1,64](d<d0-0.1)` | 274/300 | 91/100 | 0/0 |
| `F[1,64](d<d0-0.2)` | 265/300 | 88/100 | 0/0 |
| `F[1,64](d<d0-0.3)` | 258/300 | 85/100 | 0/0 |
| `F[1,64](d<d0-0.5)` | 248/300 | 79/100 | 0/0 |
| `G[0,64](hazard_margin>=0)` | 236/300 | 81/100 | None/None |
| `F[1,60](G[0,4](hazard_margin>=0))` | 300/300 | 100/100 | None/None |
| `F[1,60](G[0,4](hazard_margin>=0)); initial_hazard` | 26/26 | 6/6 | None/None |
| `F[1,55](G[0,9](hazard_margin>=0))` | 300/300 | 100/100 | None/None |
| `F[1,55](G[0,9](hazard_margin>=0)); initial_hazard` | 26/26 | 6/6 | None/None |
| `F[1,49](G[0,15](hazard_margin>=0))` | 300/300 | 100/100 | None/None |
| `F[1,49](G[0,15](hazard_margin>=0)); initial_hazard` | 26/26 | 6/6 | None/None |
| `F[1,41](G[0,23](hazard_margin>=0))` | 300/300 | 100/100 | None/None |
| `F[1,41](G[0,23](hazard_margin>=0)); initial_hazard` | 26/26 | 6/6 | None/None |
| `F[1,33](G[0,31](hazard_margin>=0))` | 300/300 | 100/100 | None/None |
| `F[1,33](G[0,31](hazard_margin>=0)); initial_hazard` | 26/26 | 6/6 | None/None |
| `F[1,17](G[0,47](hazard_margin>=0))` | 300/300 | 100/100 | None/None |
| `F[1,17](G[0,47](hazard_margin>=0)); initial_hazard` | 26/26 | 6/6 | None/None |
| `F[1,1](G[0,63](hazard_margin>=0))` | 253/300 | 86/100 | None/None |
| `F[1,1](G[0,63](hazard_margin>=0)); initial_hazard` | 17/26 | 5/6 | None/None |
| `G[4,64](hazard_margin>=0)` | 283/300 | 94/100 | None/None |
| `G[8,64](hazard_margin>=0)` | 300/300 | 100/100 | None/None |
| `G[12,64](hazard_margin>=0)` | 300/300 | 100/100 | None/None |
| `G[16,64](hazard_margin>=0)` | 300/300 | 100/100 | None/None |
| `G[17,64](hazard_margin>=0)` | 300/300 | 100/100 | None/None |
| `G[0,64](d<2)` | 198/300 | 63/100 | None/None |
| `G[0,64](d<2.5)` | 263/300 | 88/100 | None/None |
| `G[0,64](d<3)` | 289/300 | 98/100 | None/None |
| `G[0,64](d<3.5)` | 300/300 | 100/100 | None/None |
| `G[0,64](d<4)` | 300/300 | 100/100 | None/None |

## Limitations

- Training and repeatedly used development data only; no fresh test.
- Adaptive specification selection: no confidence bounds or warrant issued.
- All quantities are decoded imagination APs, not real-environment cost.
- Window indices are inclusive state indices; 64 transitions give 65 states.
- Post-hoc narrowed initial distributions require fresh conditional sampling.
- A long clear window permits earlier or later hazards; it is not full-horizon safety.
- No V was trained for the newly screened formulas.
