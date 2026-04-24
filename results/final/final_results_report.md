# Final Results

This report contains the final results for all datasets and all methods.

## Retrieval Results

### cshsi_Dioni

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.219120 | 0.013929 | 0.6004 | 0.0037 | 0.1004 |
| osl_mahalanobis_diag | 100 | 0.295950 | 0.019286 | 0.4864 | 0.0040 | 0.1012 |
| osl_mahalanobis_full | 100 | 0.276950 | 0.013929 | 0.4937 | 0.0035 | 0.0954 |
| random_dense | 100 | 0.682320 | 0.028214 | 0.0009 | 0.0041 | 0.0973 |
| random_sparse | 100 | 0.573710 | 0.022500 | 0.0042 | 0.0041 | 0.0879 |

### cshsi_Houston13

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.242100 | 0.007368 | 0.4072 | 0.0021 | 0.0963 |
| osl_mahalanobis_diag | 100 | 0.577590 | 0.007368 | 0.5899 | 0.0033 | 0.0975 |
| osl_mahalanobis_full | 100 | 0.559920 | 0.009474 | 0.4987 | 0.0033 | 0.1033 |
| random_dense | 100 | 0.749970 | 0.011579 | 0.0003 | 0.0040 | 0.0922 |
| random_sparse | 100 | 0.579880 | 0.006842 | 0.0039 | 0.0048 | 0.0765 |

### cshsi_Houston18

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.203480 | 0.163860 | 0.4924 | 0.0022 | 0.0122 |
| osl_mahalanobis_diag | 100 | 0.558960 | 0.188912 | 0.5008 | 0.0034 | 0.0994 |
| osl_mahalanobis_full | 100 | 0.646450 | 0.225579 | 0.5006 | 0.0879 | 0.1000 |
| random_dense | 100 | 0.723090 | 0.222421 | 0.0004 | 0.0046 | 0.0930 |
| random_sparse | 100 | 0.609400 | 0.215754 | 0.0043 | 0.0041 | 0.0878 |

### cshsi_Loukia

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.374650 | 0.012045 | 0.5980 | 0.0033 | 0.1005 |
| osl_mahalanobis_diag | 100 | 0.290390 | 0.014773 | 0.6035 | 0.0875 | 0.0128 |
| osl_mahalanobis_full | 100 | 0.292940 | 0.015455 | 0.5954 | 0.0039 | 0.0979 |
| random_dense | 100 | 0.638920 | 0.022727 | 0.0008 | 0.0044 | 0.0942 |
| random_sparse | 100 | 0.575940 | 0.022273 | 0.0044 | 0.0040 | 0.0898 |

### cshsi_paviaC

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.364300 | 0.029574 | 0.4046 | 0.0034 | 0.1010 |
| osl_mahalanobis_diag | 100 | 0.327870 | 0.023617 | 0.3902 | 0.0038 | 0.0950 |
| osl_mahalanobis_full | 100 | 0.388210 | 0.027447 | 0.5977 | 0.0034 | 0.0975 |
| random_dense | 100 | 0.731680 | 0.049574 | 0.0005 | 0.0037 | 0.0158 |
| random_sparse | 100 | 0.504750 | 0.037872 | 0.0049 | 0.0038 | 0.0174 |

### cshsi_paviaU

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.274320 | 0.109105 | 0.4977 | 0.0038 | 0.1023 |
| osl_mahalanobis_diag | 100 | 0.496550 | 0.138684 | 0.4863 | 0.0041 | 0.1006 |
| osl_mahalanobis_full | 100 | 0.377080 | 0.105526 | 0.5867 | 0.0042 | 0.1011 |
| random_dense | 100 | 0.748010 | 0.141053 | 0.0007 | 0.0041 | 0.0915 |
| random_sparse | 100 | 0.420800 | 0.150421 | 0.0047 | 0.0039 | 0.0825 |

### glove_6b_300d

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.197070 | - | 0.5000 | 0.0041 | 0.1010 |
| osl_mahalanobis_diag | 100 | 0.244550 | - | 0.6794 | 0.0116 | 0.1035 |
| osl_mahalanobis_full | 100 | 0.185640 | - | 0.8009 | 0.0839 | 0.1133 |
| random_dense | 100 | 0.180130 | - | 0.0013 | 0.0043 | 0.0991 |
| random_sparse | 100 | 0.180700 | - | 0.0042 | 0.0044 | 0.0802 |

### mars_15d_band4

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 50 | 0.280133 | - | 4.2449 | 0.0062 | 0.0021 |
| osl_mahalanobis_diag | 50 | 0.445867 | - | 4.3937 | 0.0066 | 0.0022 |
| osl_mahalanobis_full | 50 | 0.314467 | - | 4.3796 | 0.0048 | 0.0017 |
| random_dense | 50 | 0.495200 | - | 0.2281 | 0.0044 | 0.0021 |
| random_sparse | 50 | 0.267067 | - | 0.0722 | 0.0038 | 0.0022 |

### mnist

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.522490 | 0.361720 | 0.6073 | 0.0039 | 0.1925 |
| osl_mahalanobis_diag | 100 | 0.526490 | 0.340230 | 0.8668 | 0.0039 | 0.1859 |
| osl_mahalanobis_full | 100 | 0.374460 | 0.294890 | 0.8774 | 0.0044 | 0.1029 |
| random_dense | 100 | 0.408310 | 0.288780 | 0.0029 | 0.0043 | 0.0717 |
| random_sparse | 100 | 0.424770 | 0.291260 | 0.0045 | 0.0066 | 0.0697 |

### sift1m

| method | neighbor_k | precision | same_label | fit_time_sec | encode_time_sec | query_time_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| osl_euclidean | 100 | 0.567250 | - | 0.4967 | 0.0040 | 0.1026 |
| osl_mahalanobis_diag | 100 | 0.513500 | - | 0.4880 | 0.0038 | 0.1032 |
| osl_mahalanobis_full | 100 | 0.484320 | - | 0.4897 | 0.0040 | 0.1115 |
| random_dense | 100 | 0.468960 | - | 0.0006 | 0.0037 | 0.0167 |
| random_sparse | 100 | 0.482730 | - | 0.0042 | 0.0041 | 0.0806 |

## Notes

- `same_label` is only defined for datasets with labels.
- `mars_15d_band4` uses `neighbor_k = 50`; the other main runs use `neighbor_k = 100`.
- These files are final outputs only. No tuning traces or search-process artifacts are included here.
