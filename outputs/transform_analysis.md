# Quadrant axis transform analysis

Criterion: minimize mean absolute skewness of the pooled artist year totals across the two axes, restricted to smooth magnitude preserving transforms.

Each axis carries per artist per year totals across a pool of a few hundred artists. Streams and placements are heavily right skewed, so raw min max scaling would crush most artists near the lower bound and stretch a few outliers to the upper bound. The table below reports Fisher Pearson skewness after each candidate transform, and the mean of the absolute values across the two axes.

| Transform | Streams skew | Placements skew | Mean abs skew |
|-----------|-------------:|----------------:|--------------:|
| identity | 2.018 | 1.305 | 1.661 |
| sqrt | 1.076 | 0.635 | 0.856 |
| log1p chosen | -0.218 | -0.117 | 0.168 |
| rank (reference) | 0.000 | 0.000 | 0.000 |

Chosen transform: **log1p**.

The rank transform reaches near zero skewness by construction but is rejected because it centers the median and discards magnitude.
