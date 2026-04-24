# Final Results

This report contains the final OSL-only results. Random baselines are intentionally omitted from the final report package.

## Shared OSL Configuration

- `output_dim` = `320`
- `hash_length` = `8`
- `row_active` = `8`
- `y_iters` = `20`
- `w_iters` = `40`
- `covariance_reg` = `0.01`
- `seed` = `0`

`osl_mahalanobis_full` was swept on `cshsi_paviaU`, `cshsi_Houston18`, and `mars_15d_band4`; the selected configuration above is shared by all three OSL methods in the final comparison.

## Retrieval Results

### cshsi_Dioni

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.442630 | 0.016250 | 2.2966 | 0.1957 | 0.2065 |
| osl_mahalanobis_diag | 100 | 0.461530 | 0.026429 | 3.3740 | 0.1008 | 0.2083 |
| osl_mahalanobis_full | 100 | 0.284570 | 0.014821 | 3.7731 | 0.1010 | 0.2021 |

### cshsi_Houston13

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.347280 | 0.011579 | 1.2980 | 0.0989 | 0.2112 |
| osl_mahalanobis_diag | 100 | 0.491190 | 0.012105 | 2.7010 | 0.0997 | 0.2037 |
| osl_mahalanobis_full | 100 | 0.720090 | 0.007895 | 2.8733 | 0.1998 | 0.2023 |

### cshsi_Houston18

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.287820 | 0.176421 | 1.5981 | 0.0983 | 0.2023 |
| osl_mahalanobis_diag | 100 | 0.494280 | 0.181018 | 2.2761 | 0.0999 | 0.2013 |
| osl_mahalanobis_full | 100 | 0.648480 | 0.199474 | 2.9737 | 0.1020 | 0.2090 |

### cshsi_Loukia

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.408350 | 0.019318 | 1.7958 | 0.0055 | 0.2071 |
| osl_mahalanobis_diag | 100 | 0.515020 | 0.016818 | 2.9857 | 0.1007 | 0.2042 |
| osl_mahalanobis_full | 100 | 0.374650 | 0.014545 | 3.6738 | 0.1016 | 0.2023 |

### cshsi_paviaC

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.378810 | 0.025745 | 1.1014 | 0.0050 | 0.2972 |
| osl_mahalanobis_diag | 100 | 0.413470 | 0.027872 | 3.5724 | 0.1012 | 0.2031 |
| osl_mahalanobis_full | 100 | 0.520680 | 0.037021 | 3.9766 | 0.0994 | 0.2070 |

### cshsi_paviaU

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.317300 | 0.125211 | 1.6998 | 0.0063 | 0.3042 |
| osl_mahalanobis_diag | 100 | 0.542030 | 0.124737 | 2.8642 | 0.1002 | 0.2068 |
| osl_mahalanobis_full | 100 | 0.667390 | 0.129789 | 3.4697 | 0.1015 | 0.2188 |

### glove_6b_300d

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.290870 | - | 2.8918 | 0.1974 | 0.1099 |
| osl_mahalanobis_diag | 100 | 0.402850 | - | 3.7863 | 0.1958 | 0.2017 |
| osl_mahalanobis_full | 100 | 0.317160 | - | 4.3832 | 0.1960 | 0.2015 |

Note: `streamed_first_10000_vectors_from_raw_glove`

### mars_15d_band4

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 50 | 0.271133 | - | 13.6013 | 0.1011 | 0.0964 |
| osl_mahalanobis_diag | 50 | 0.456333 | - | 15.3958 | 0.1020 | 0.0025 |
| osl_mahalanobis_full | 50 | 0.330400 | - | 15.0134 | 0.0861 | 0.0961 |

### mnist

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.484500 | 0.359750 | 3.3942 | 0.0999 | 0.2087 |
| osl_mahalanobis_diag | 100 | 0.533420 | 0.353990 | 6.0647 | 0.1919 | 0.2077 |
| osl_mahalanobis_full | 100 | 0.431140 | 0.311820 | 7.0642 | 0.1938 | 0.2051 |

### sift1m

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.545760 | - | 2.4015 | 0.1961 | 0.2034 |
| osl_mahalanobis_diag | 100 | 0.477340 | - | 3.0805 | 0.1959 | 0.1054 |
| osl_mahalanobis_full | 100 | 0.503080 | - | 3.4847 | 0.1013 | 0.2125 |

Note: `first_300_learn_and_first_1000_query_from_raw_sift`

## Notes

- `same_label` is only defined for datasets with labels.
- `mars_15d_band4` uses `neighbor_k = 50`; the other runs use `neighbor_k = 100`.
- The final package keeps only the three OSL methods.
