# Recursive streams and placements forecast

Model: first order vector autoregression on log yearly totals, fit on the pooled year over year transitions and applied recursively for 10 years.

streams and placements predict each other forward, so the two paths co evolve. Every pool artist is forecast forward, and each predicted year is normalized against the field predicted for that year, so a predicted position reflects standing relative to peers and can diverge from the artist's own raw trend when the field moves too.

## One step transition, log space coefficients

| Next year | log streams | log placements | intercept |
|-----------|------------:|---------------:|----------:|
| log streams | 0.6627 | 0.3273 | 1.9438 |
| log placements | -0.0361 | 1.0193 | 0.4115 |

Each next year total loads on both current totals, which is how the two variables carry each other forward.

## One step held out error

Fit on 914 pooled year over year transitions, evaluated on 453 transitions from held out artists. Error is the median absolute percentage error in raw units, against a persistence baseline that predicts next year equals this year.

| Axis | Model | Persistence |
|------|------:|------------:|
| Streams | 16.4% | 17.1% |
| Placements | 14.0% | 13.5% |

## Example ten year paths

Predicted yearly totals stepping forward from each artist's most recent observed year.

- Nova Reyes, last observed 2025 at 54,191,193 streams and 235,248 placement adds: 2026 (53,431,268 / 236,766), 2027 (53,045,133 / 238,446), 2028 (52,913,050 / 240,232), 2029 (52,954,954 / 242,089), 2030 (53,116,448 / 243,990), 2031 (53,360,146 / 245,915), 2032 (53,660,146 / 247,852), 2033 (53,998,393 / 249,792), 2034 (54,362,252 / 251,727), 2035 (54,742,857 / 253,653)
- The Meridian, last observed 2025 at 84,097,085 streams and 329,327 placement adds: 2026 (79,817,504 / 328,353), 2027 (77,027,355 / 327,981), 2028 (75,204,371 / 328,024), 2029 (74,023,279 / 328,352), 2030 (73,274,757 / 328,874), 2031 (72,820,774 / 329,529), 2032 (72,568,673 / 330,271), 2033 (72,455,446 / 331,071), 2034 (72,437,851 / 331,907), 2035 (72,486,018 / 332,765)
- Kaito Sol, last observed 2025 at 17,005,305 streams and 73,477 placement adds: 2026 (16,935,588 / 75,404), 2027 (17,033,279 / 77,432), 2028 (17,247,494 / 79,538), 2029 (17,544,397 / 81,707), 2030 (17,900,926 / 83,927), 2031 (18,301,065 / 86,189), 2032 (18,733,538 / 88,487), 2033 (19,190,314 / 90,815), 2034 (19,665,625 / 93,170), 2035 (20,155,293 / 95,549)

Each future year is written as year (streams / placement adds).
