# Results

## Test results, configurations selected by 5-fold CV hamming loss

| dataset | method | hamming_loss | subset_accuracy | micro_f1 | macro_f1 | ranking_loss | avg_precision | config |
|---|---|---|---|---|---|---|---|---|
| emotions | ML-kNN | 0.2153 | 0.2574 | 0.6390 | 0.6139 | 0.1701 | 0.7859 | euclid, pca=-, alpha=0, gamma=0, k=10, s=2 |
| emotions | ML-kNN + PCA | 0.2096 | 0.2871 | 0.6684 | 0.6337 | 0.1637 | 0.8017 | euclid, pca=30, alpha=0, gamma=0, k=10, s=1 |
| emotions | NTK (raw) | 0.2360 | 0.2426 | 0.6104 | 0.5505 | 0.2111 | 0.7519 | ntk_raw, pca=30, alpha=1, gamma=0, k=15, s=0.5 |
| emotions | NTK (distance) | 0.2063 | 0.2673 | 0.6469 | 0.6136 | 0.1779 | 0.7856 | ntk_dist, pca=30, alpha=1, gamma=0, k=10, s=2 |
| emotions | Hybrid (value fusion) | 0.2294 | 0.2079 | 0.6029 | 0.5775 | 0.1863 | 0.7797 | mix_raw, pca=72, alpha=0.75, gamma=0.01, k=15, s=0.5 |
| emotions | Hybrid (kernel distance) | 0.2137 | 0.2574 | 0.6357 | 0.6180 | 0.1663 | 0.7894 | mix_dist, pca=72, alpha=0.1, gamma=0.01, k=10, s=2 |
| scene | ML-kNN | 0.0969 | 0.5978 | 0.7171 | 0.7208 | 0.0922 | 0.8529 | euclid, pca=-, alpha=0, gamma=0, k=10, s=0.5 |
| scene | ML-kNN + PCA | 0.0879 | 0.6154 | 0.7293 | 0.7304 | 0.0864 | 0.8593 | euclid, pca=50, alpha=0, gamma=0, k=10, s=0.5 |
| scene | NTK (raw) | 0.1182 | 0.4933 | 0.6177 | 0.6007 | 0.1192 | 0.8098 | ntk_raw, pca=30, alpha=1, gamma=0, k=10, s=1 |
| scene | NTK (distance) | 0.1028 | 0.5711 | 0.6902 | 0.6909 | 0.1188 | 0.8260 | ntk_dist, pca=30, alpha=1, gamma=0, k=5, s=2 |
| scene | Hybrid (value fusion) | 0.0885 | 0.6329 | 0.7317 | 0.7349 | 0.0876 | 0.8513 | mix_raw, pca=80, alpha=0.1, gamma=0.001, k=10, s=0.5 |
| scene | Hybrid (kernel distance) | 0.0860 | 0.6246 | 0.7373 | 0.7388 | 0.0888 | 0.8576 | mix_dist, pca=50, alpha=0.1, gamma=0.01, k=10, s=2 |
| yeast | ML-kNN | 0.1987 | 0.1679 | 0.6259 | 0.3392 | 0.1707 | 0.7577 | euclid, pca=-, alpha=0, gamma=0, k=10, s=2 |
| yeast | ML-kNN + PCA | 0.1987 | 0.1647 | 0.6200 | 0.3309 | 0.1709 | 0.7584 | euclid, pca=80, alpha=0, gamma=0, k=10, s=0.5 |
| yeast | NTK (raw) | 0.1984 | 0.1668 | 0.6312 | 0.3464 | 0.1703 | 0.7570 | ntk_raw, pca=50, alpha=1, gamma=0, k=10, s=0.5 |
| yeast | NTK (distance) | 0.1989 | 0.1592 | 0.6175 | 0.3335 | 0.1710 | 0.7580 | ntk_dist, pca=80, alpha=1, gamma=0, k=10, s=0.5 |
| yeast | Hybrid (value fusion) | 0.1994 | 0.1636 | 0.6185 | 0.3330 | 0.1721 | 0.7573 | mix_raw, pca=80, alpha=0.1, gamma=0.001, k=10, s=0.5 |
| yeast | Hybrid (kernel distance) | 0.1992 | 0.1549 | 0.6202 | 0.3341 | 0.1718 | 0.7545 | mix_dist, pca=80, alpha=0.5, gamma=0.001, k=10, s=0.5 |
| medical | ML-kNN | 0.0223 | 0.2946 | 0.5038 | 0.1745 | 0.0652 | 0.6991 | euclid, pca=-, alpha=0, gamma=0, k=10, s=0.5 |
| medical | ML-kNN + PCA | 0.0229 | 0.2853 | 0.4536 | 0.1248 | 0.0736 | 0.6745 | euclid, pca=80, alpha=0, gamma=0, k=5, s=1 |
| medical | NTK (raw) | 0.0256 | 0.1132 | 0.2382 | 0.0625 | 0.0889 | 0.5912 | ntk_raw, pca=80, alpha=1, gamma=0, k=10, s=1 |
| medical | NTK (distance) | 0.0241 | 0.2667 | 0.4243 | 0.1031 | 0.0818 | 0.6424 | ntk_dist, pca=50, alpha=1, gamma=0, k=5, s=0.5 |
| medical | Hybrid (value fusion) | 0.0253 | 0.1860 | 0.3049 | 0.0486 | 0.0755 | 0.6436 | mix_raw, pca=50, alpha=0.1, gamma=0.001, k=5, s=2 |
| medical | Hybrid (kernel distance) | 0.0240 | 0.2295 | 0.3941 | 0.1006 | 0.0645 | 0.6603 | mix_dist, pca=80, alpha=0.75, gamma=0.01, k=15, s=0.5 |
| enron | ML-kNN | 0.0543 | 0.0674 | 0.3983 | 0.0642 | 0.0954 | 0.6133 | euclid, pca=-, alpha=0, gamma=0, k=15, s=2 |
| enron | ML-kNN + PCA | 0.0492 | 0.1002 | 0.5113 | 0.0961 | 0.0875 | 0.6631 | euclid, pca=30, alpha=0, gamma=0, k=15, s=0.5 |
| enron | NTK (raw) | 0.0587 | 0.0017 | 0.2885 | 0.0250 | 0.1140 | 0.5365 | ntk_raw, pca=80, alpha=1, gamma=0, k=10, s=1 |
| enron | NTK (distance) | 0.0488 | 0.0829 | 0.5185 | 0.0940 | 0.0870 | 0.6692 | ntk_dist, pca=30, alpha=1, gamma=0, k=15, s=2 |
| enron | Hybrid (value fusion) | 0.0492 | 0.1002 | 0.5113 | 0.0961 | 0.0875 | 0.6631 | euclid, pca=30, alpha=0, gamma=0, k=15, s=0.5 |
| enron | Hybrid (kernel distance) | 0.0489 | 0.0777 | 0.5059 | 0.0955 | 0.0870 | 0.6666 | mix_dist, pca=30, alpha=0.25, gamma=0.01, k=15, s=0.5 |
| corel5k | ML-kNN | 0.0093 | 0.0060 | 0.0616 | 0.0140 | 0.1301 | 0.2679 | euclid, pca=-, alpha=0, gamma=0, k=5, s=2 |
| corel5k | ML-kNN + PCA | 0.0092 | 0.0020 | 0.1043 | 0.0204 | 0.1150 | 0.3368 | euclid, pca=30, alpha=0, gamma=0, k=10, s=1 |
| corel5k | NTK (raw) | 0.0094 | 0.0000 | 0.0331 | 0.0080 | 0.1254 | 0.2947 | ntk_raw, pca=80, alpha=1, gamma=0, k=10, s=2 |
| corel5k | NTK (distance) | 0.0092 | 0.0020 | 0.0954 | 0.0173 | 0.1164 | 0.3215 | ntk_dist, pca=30, alpha=1, gamma=0, k=10, s=2 |
| corel5k | Hybrid (value fusion) | 0.0092 | 0.0020 | 0.1043 | 0.0204 | 0.1150 | 0.3368 | euclid, pca=30, alpha=0, gamma=0, k=10, s=1 |
| corel5k | Hybrid (kernel distance) | 0.0092 | 0.0060 | 0.1079 | 0.0210 | 0.1152 | 0.3328 | mix_dist, pca=30, alpha=0.1, gamma=0.01, k=10, s=1 |
| bibtex | ML-kNN | 0.0150 | 0.0402 | 0.0990 | 0.0545 | 0.2419 | 0.2670 | euclid, pca=-, alpha=0, gamma=0, k=10, s=0.5 |
| bibtex | ML-kNN + PCA | 0.0130 | 0.1360 | 0.3236 | 0.1398 | 0.1043 | 0.5036 | euclid, pca=80, alpha=0, gamma=0, k=15, s=1 |
| bibtex | NTK (raw) | 0.0135 | 0.1133 | 0.2630 | 0.0894 | 0.1710 | 0.3856 | ntk_raw, pca=50, alpha=1, gamma=0, k=15, s=1 |
| bibtex | NTK (distance) | 0.0131 | 0.1368 | 0.3235 | 0.1363 | 0.1625 | 0.4581 | ntk_dist, pca=50, alpha=1, gamma=0, k=5, s=0.5 |
| bibtex | Hybrid (value fusion) | 0.0128 | 0.1447 | 0.3544 | 0.1701 | 0.1128 | 0.5084 | mix_raw, pca=80, alpha=0.1, gamma=0.001, k=10, s=1 |
| bibtex | Hybrid (kernel distance) | 0.0130 | 0.1435 | 0.3378 | 0.1521 | 0.1178 | 0.4930 | mix_dist, pca=50, alpha=0.1, gamma=0.01, k=10, s=1 |

## Test results, configurations selected by 5-fold CV avg precision

| dataset | method | hamming_loss | subset_accuracy | micro_f1 | macro_f1 | ranking_loss | avg_precision | config |
|---|---|---|---|---|---|---|---|---|
| emotions | ML-kNN | 0.2153 | 0.2574 | 0.6390 | 0.6139 | 0.1719 | 0.7845 | euclid, pca=-, alpha=0, gamma=0, k=10, s=0.5 |
| emotions | ML-kNN + PCA | 0.2153 | 0.2574 | 0.6390 | 0.6139 | 0.1719 | 0.7845 | euclid, pca=72, alpha=0, gamma=0, k=10, s=0.5 |
| emotions | NTK (raw) | 0.2459 | 0.2228 | 0.5826 | 0.5093 | 0.2113 | 0.7491 | ntk_raw, pca=30, alpha=1, gamma=0, k=10, s=2 |
| emotions | NTK (distance) | 0.2211 | 0.2277 | 0.6105 | 0.5744 | 0.1637 | 0.7987 | ntk_dist, pca=30, alpha=1, gamma=0, k=15, s=1 |
| emotions | Hybrid (value fusion) | 0.2153 | 0.2574 | 0.6390 | 0.6139 | 0.1719 | 0.7845 | euclid, pca=72, alpha=0, gamma=0, k=10, s=0.5 |
| emotions | Hybrid (kernel distance) | 0.2153 | 0.2574 | 0.6390 | 0.6139 | 0.1719 | 0.7845 | euclid, pca=72, alpha=0, gamma=0, k=10, s=0.5 |
| scene | ML-kNN | 0.0962 | 0.5560 | 0.6928 | 0.6827 | 0.0869 | 0.8516 | euclid, pca=-, alpha=0, gamma=0, k=15, s=0.5 |
| scene | ML-kNN + PCA | 0.0903 | 0.6137 | 0.7282 | 0.7285 | 0.0830 | 0.8603 | euclid, pca=50, alpha=0, gamma=0, k=15, s=1 |
| scene | NTK (raw) | 0.1197 | 0.4532 | 0.5973 | 0.5872 | 0.1178 | 0.8070 | ntk_raw, pca=30, alpha=1, gamma=0, k=15, s=1 |
| scene | NTK (distance) | 0.0999 | 0.5401 | 0.6812 | 0.6768 | 0.1020 | 0.8380 | ntk_dist, pca=30, alpha=1, gamma=0, k=10, s=0.5 |
| scene | Hybrid (value fusion) | 0.0849 | 0.6530 | 0.7451 | 0.7461 | 0.0782 | 0.8647 | mix_raw, pca=50, alpha=0.1, gamma=0.001, k=10, s=0.5 |
| scene | Hybrid (kernel distance) | 0.0903 | 0.6137 | 0.7282 | 0.7285 | 0.0830 | 0.8603 | euclid, pca=50, alpha=0, gamma=0, k=15, s=1 |
| yeast | ML-kNN | 0.2000 | 0.1658 | 0.6254 | 0.3438 | 0.1702 | 0.7594 | euclid, pca=-, alpha=0, gamma=0, k=10, s=1 |
| yeast | ML-kNN + PCA | 0.2007 | 0.1505 | 0.6211 | 0.3364 | 0.1712 | 0.7561 | euclid, pca=50, alpha=0, gamma=0, k=10, s=0.5 |
| yeast | NTK (raw) | 0.1968 | 0.1494 | 0.6267 | 0.3411 | 0.1719 | 0.7599 | ntk_raw, pca=80, alpha=1, gamma=0, k=10, s=0.5 |
| yeast | NTK (distance) | 0.1989 | 0.1592 | 0.6175 | 0.3335 | 0.1710 | 0.7580 | ntk_dist, pca=80, alpha=1, gamma=0, k=10, s=0.5 |
| yeast | Hybrid (value fusion) | 0.1965 | 0.1483 | 0.6269 | 0.3416 | 0.1716 | 0.7598 | mix_raw, pca=80, alpha=0.9, gamma=0.001, k=10, s=0.5 |
| yeast | Hybrid (kernel distance) | 0.1996 | 0.1647 | 0.6301 | 0.3478 | 0.1722 | 0.7547 | mix_dist, pca=50, alpha=0.1, gamma=0.01, k=10, s=0.5 |
| medical | ML-kNN | 0.0227 | 0.2496 | 0.4728 | 0.1321 | 0.0658 | 0.6992 | euclid, pca=-, alpha=0, gamma=0, k=10, s=1 |
| medical | ML-kNN + PCA | 0.0247 | 0.2667 | 0.4362 | 0.1333 | 0.0651 | 0.6746 | euclid, pca=80, alpha=0, gamma=0, k=10, s=0.5 |
| medical | NTK (raw) | 0.0251 | 0.1690 | 0.3394 | 0.0814 | 0.0833 | 0.5994 | ntk_raw, pca=80, alpha=1, gamma=0, k=15, s=0.5 |
| medical | NTK (distance) | 0.0239 | 0.2434 | 0.3983 | 0.1018 | 0.0804 | 0.6590 | ntk_dist, pca=80, alpha=1, gamma=0, k=5, s=1 |
| medical | Hybrid (value fusion) | 0.0231 | 0.2946 | 0.4640 | 0.1246 | 0.0562 | 0.6994 | mix_raw, pca=80, alpha=0.25, gamma=0.001, k=10, s=0.5 |
| medical | Hybrid (kernel distance) | 0.0243 | 0.1814 | 0.3343 | 0.0899 | 0.0678 | 0.6672 | mix_dist, pca=50, alpha=0.1, gamma=0.001, k=10, s=0.5 |
| enron | ML-kNN | 0.0543 | 0.0674 | 0.3983 | 0.0642 | 0.0960 | 0.6140 | euclid, pca=-, alpha=0, gamma=0, k=15, s=0.5 |
| enron | ML-kNN + PCA | 0.0494 | 0.1019 | 0.5240 | 0.1000 | 0.0875 | 0.6693 | euclid, pca=30, alpha=0, gamma=0, k=10, s=1 |
| enron | NTK (raw) | 0.0585 | 0.0017 | 0.2890 | 0.0250 | 0.1140 | 0.5365 | ntk_raw, pca=80, alpha=1, gamma=0, k=10, s=2 |
| enron | NTK (distance) | 0.0491 | 0.0846 | 0.5198 | 0.0969 | 0.0876 | 0.6700 | ntk_dist, pca=30, alpha=1, gamma=0, k=15, s=1 |
| enron | Hybrid (value fusion) | 0.0494 | 0.1019 | 0.5240 | 0.1000 | 0.0875 | 0.6693 | euclid, pca=30, alpha=0, gamma=0, k=10, s=1 |
| enron | Hybrid (kernel distance) | 0.0495 | 0.1036 | 0.5235 | 0.0982 | 0.0871 | 0.6699 | mix_dist, pca=30, alpha=0.1, gamma=0.01, k=10, s=1 |
| corel5k | ML-kNN | 0.0093 | 0.0020 | 0.0472 | 0.0113 | 0.1275 | 0.2721 | euclid, pca=-, alpha=0, gamma=0, k=15, s=2 |
| corel5k | ML-kNN + PCA | 0.0092 | 0.0100 | 0.1070 | 0.0214 | 0.1092 | 0.3409 | euclid, pca=30, alpha=0, gamma=0, k=15, s=2 |
| corel5k | NTK (raw) | 0.0093 | 0.0060 | 0.0512 | 0.0117 | 0.1235 | 0.2990 | ntk_raw, pca=80, alpha=1, gamma=0, k=15, s=2 |
| corel5k | NTK (distance) | 0.0093 | 0.0040 | 0.0914 | 0.0164 | 0.1131 | 0.3255 | ntk_dist, pca=30, alpha=1, gamma=0, k=15, s=2 |
| corel5k | Hybrid (value fusion) | 0.0092 | 0.0080 | 0.1034 | 0.0209 | 0.1094 | 0.3444 | mix_raw, pca=30, alpha=0.25, gamma=0.01, k=15, s=2 |
| corel5k | Hybrid (kernel distance) | 0.0092 | 0.0060 | 0.1003 | 0.0217 | 0.1097 | 0.3440 | mix_dist, pca=30, alpha=0.1, gamma=0.01, k=15, s=2 |
| bibtex | ML-kNN | 0.0150 | 0.0402 | 0.0893 | 0.0547 | 0.2407 | 0.2739 | euclid, pca=-, alpha=0, gamma=0, k=15, s=0.5 |
| bibtex | ML-kNN + PCA | 0.0130 | 0.1308 | 0.3039 | 0.1097 | 0.1037 | 0.5040 | euclid, pca=80, alpha=0, gamma=0, k=15, s=2 |
| bibtex | NTK (raw) | 0.0136 | 0.1074 | 0.2440 | 0.0697 | 0.1674 | 0.3848 | ntk_raw, pca=30, alpha=1, gamma=0, k=15, s=2 |
| bibtex | NTK (distance) | 0.0133 | 0.1276 | 0.2885 | 0.0948 | 0.1143 | 0.4910 | ntk_dist, pca=30, alpha=1, gamma=0, k=15, s=2 |
| bibtex | Hybrid (value fusion) | 0.0132 | 0.1312 | 0.3099 | 0.1180 | 0.0989 | 0.5164 | mix_raw, pca=80, alpha=0.25, gamma=0.001, k=15, s=2 |
| bibtex | Hybrid (kernel distance) | 0.0130 | 0.1308 | 0.3039 | 0.1097 | 0.1037 | 0.5040 | euclid, pca=80, alpha=0, gamma=0, k=15, s=2 |

## Differences vs. ML-kNN + PCA (test split, paired bootstrap, 2000 resamples; selection by CV Hamming loss)

| dataset | family | alpha | d_hamming_loss | d_hamming_loss_lo | d_hamming_loss_hi | d_avg_precision | d_avg_precision_lo | d_avg_precision_hi |
|---|---|---|---|---|---|---|---|---|
| emotions | ML-kNN | 0.0000 | 0.0058 | -0.0099 | 0.0206 | -0.0158 | -0.0379 | 0.0074 |
| emotions | NTK (raw) | 1.0000 | 0.0264 | 0.0058 | 0.0479 | -0.0498 | -0.0768 | -0.0216 |
| emotions | NTK (distance) | 1.0000 | -0.0033 | -0.0182 | 0.0124 | -0.0161 | -0.0400 | 0.0078 |
| emotions | Hybrid (value fusion) | 0.7500 | 0.0198 | 0.0008 | 0.0388 | -0.0221 | -0.0466 | 0.0016 |
| emotions | Hybrid (kernel distance) | 0.1000 | 0.0041 | -0.0099 | 0.0182 | -0.0124 | -0.0348 | 0.0101 |
| scene | ML-kNN | 0.0000 | 0.0089 | 0.0035 | 0.0145 | -0.0064 | -0.0162 | 0.0037 |
| scene | NTK (raw) | 1.0000 | 0.0302 | 0.0224 | 0.0383 | -0.0495 | -0.0662 | -0.0327 |
| scene | NTK (distance) | 1.0000 | 0.0149 | 0.0095 | 0.0206 | -0.0332 | -0.0463 | -0.0211 |
| scene | Hybrid (value fusion) | 0.1000 | 0.0006 | -0.0046 | 0.0059 | -0.0080 | -0.0180 | 0.0029 |
| scene | Hybrid (kernel distance) | 0.1000 | -0.0020 | -0.0042 | 0.0001 | -0.0017 | -0.0058 | 0.0023 |
| yeast | ML-kNN | 0.0000 | 0.0000 | -0.0023 | 0.0023 | -0.0007 | -0.0048 | 0.0034 |
| yeast | NTK (raw) | 1.0000 | -0.0003 | -0.0037 | 0.0031 | -0.0014 | -0.0064 | 0.0035 |
| yeast | NTK (distance) | 1.0000 | 0.0002 | -0.0017 | 0.0021 | -0.0004 | -0.0036 | 0.0030 |
| yeast | Hybrid (value fusion) | 0.1000 | 0.0007 | -0.0014 | 0.0030 | -0.0011 | -0.0054 | 0.0033 |
| yeast | Hybrid (kernel distance) | 0.5000 | 0.0005 | -0.0017 | 0.0026 | -0.0039 | -0.0085 | 0.0008 |
| medical | ML-kNN | 0.0000 | -0.0007 | -0.0021 | 0.0008 | 0.0246 | 0.0027 | 0.0482 |
| medical | NTK (raw) | 1.0000 | 0.0027 | 0.0013 | 0.0040 | -0.0833 | -0.1108 | -0.0546 |
| medical | NTK (distance) | 1.0000 | 0.0012 | 0.0001 | 0.0023 | -0.0321 | -0.0491 | -0.0156 |
| medical | Hybrid (value fusion) | 0.1000 | 0.0024 | 0.0010 | 0.0037 | -0.0309 | -0.0514 | -0.0096 |
| medical | Hybrid (kernel distance) | 0.7500 | 0.0011 | 0.0000 | 0.0022 | -0.0142 | -0.0319 | 0.0032 |
| enron | ML-kNN | 0.0000 | 0.0051 | 0.0032 | 0.0068 | -0.0498 | -0.0649 | -0.0352 |
| enron | NTK (raw) | 1.0000 | 0.0095 | 0.0075 | 0.0114 | -0.1266 | -0.1432 | -0.1104 |
| enron | NTK (distance) | 1.0000 | -0.0004 | -0.0014 | 0.0007 | 0.0061 | -0.0009 | 0.0123 |
| enron | Hybrid (value fusion) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| enron | Hybrid (kernel distance) | 0.2500 | -0.0003 | -0.0009 | 0.0004 | 0.0035 | 0.0001 | 0.0071 |
| corel5k | ML-kNN | 0.0000 | 0.0001 | -0.0000 | 0.0002 | -0.0690 | -0.0853 | -0.0531 |
| corel5k | NTK (raw) | 1.0000 | 0.0002 | 0.0000 | 0.0004 | -0.0421 | -0.0609 | -0.0237 |
| corel5k | NTK (distance) | 1.0000 | 0.0000 | -0.0001 | 0.0002 | -0.0154 | -0.0255 | -0.0055 |
| corel5k | Hybrid (value fusion) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| corel5k | Hybrid (kernel distance) | 0.1000 | 0.0000 | -0.0001 | 0.0001 | -0.0041 | -0.0089 | 0.0007 |
| bibtex | ML-kNN | 0.0000 | 0.0020 | 0.0018 | 0.0022 | -0.2366 | -0.2504 | -0.2221 |
| bibtex | NTK (raw) | 1.0000 | 0.0005 | 0.0004 | 0.0007 | -0.1180 | -0.1285 | -0.1067 |
| bibtex | NTK (distance) | 1.0000 | 0.0002 | 0.0000 | 0.0003 | -0.0455 | -0.0544 | -0.0364 |
| bibtex | Hybrid (value fusion) | 0.1000 | -0.0002 | -0.0003 | -0.0000 | 0.0048 | -0.0026 | 0.0120 |
| bibtex | Hybrid (kernel distance) | 0.1000 | 0.0001 | -0.0001 | 0.0002 | -0.0106 | -0.0179 | -0.0033 |

## How often each method beats ML-kNN + PCA (number of datasets; 95% CI excludes 0)

| selected by CV | family | hamming loss: better / same / worse | avg precision: better / same / worse |
|---|---|---|---|
| hamming loss | ML-kNN | 0 / 4 / 3 | 1 / 3 / 3 |
| hamming loss | NTK (raw) | 0 / 1 / 6 | 0 / 1 / 6 |
| hamming loss | NTK (distance) | 0 / 4 / 3 | 0 / 3 / 4 |
| hamming loss | Hybrid (value fusion) | 1 / 4 / 2 | 0 / 6 / 1 |
| hamming loss | Hybrid (kernel distance) | 0 / 6 / 1 | 1 / 5 / 1 |
| avg precision | ML-kNN | 1 / 3 / 3 | 1 / 3 / 3 |
| avg precision | NTK (raw) | 1 / 2 / 4 | 0 / 1 / 6 |
| avg precision | NTK (distance) | 0 / 5 / 2 | 0 / 4 / 3 |
| avg precision | Hybrid (value fusion) | 2 / 4 / 1 | 2 / 5 / 0 |
| avg precision | Hybrid (kernel distance) | 0 / 7 / 0 | 0 / 7 / 0 |

## Choosing by test score vs. by cross-validation

| dataset | family | n_configs | test_hl_cv_selected | test_hl_best_in_grid | alpha_of_best_in_grid |
|---|---|---|---|---|---|
| emotions | ML-kNN + PCA | 27 | 0.2096 | 0.2030 | 0.0000 |
| emotions | Hybrid (value fusion) | 729 | 0.2294 | 0.1955 | 0.2500 |
| emotions | Hybrid (kernel distance) | 729 | 0.2137 | 0.1931 | 0.7500 |
| scene | ML-kNN + PCA | 27 | 0.0879 | 0.0879 | 0.0000 |
| scene | Hybrid (value fusion) | 729 | 0.0885 | 0.0849 | 0.1000 |
| scene | Hybrid (kernel distance) | 729 | 0.0860 | 0.0860 | 0.1000 |
| yeast | ML-kNN + PCA | 27 | 0.1987 | 0.1967 | 0.0000 |
| yeast | Hybrid (value fusion) | 729 | 0.1994 | 0.1961 | 0.7500 |
| yeast | Hybrid (kernel distance) | 729 | 0.1992 | 0.1967 | 0.0000 |
| medical | ML-kNN + PCA | 27 | 0.0229 | 0.0227 | 0.0000 |
| medical | Hybrid (value fusion) | 729 | 0.0253 | 0.0227 | 0.0000 |
| medical | Hybrid (kernel distance) | 729 | 0.0240 | 0.0227 | 0.0000 |
| enron | ML-kNN + PCA | 27 | 0.0492 | 0.0475 | 0.0000 |
| enron | Hybrid (value fusion) | 729 | 0.0492 | 0.0475 | 0.0000 |
| enron | Hybrid (kernel distance) | 729 | 0.0489 | 0.0475 | 0.0000 |
| corel5k | ML-kNN + PCA | 27 | 0.0092 | 0.0090 | 0.0000 |
| corel5k | Hybrid (value fusion) | 729 | 0.0092 | 0.0090 | 0.0000 |
| corel5k | Hybrid (kernel distance) | 729 | 0.0092 | 0.0090 | 0.0000 |
| bibtex | ML-kNN + PCA | 27 | 0.0130 | 0.0128 | 0.0000 |
| bibtex | Hybrid (value fusion) | 729 | 0.0128 | 0.0128 | 0.1000 |
| bibtex | Hybrid (kernel distance) | 729 | 0.0130 | 0.0128 | 0.0000 |

## Datasets, run times, and the all-negative predictor

| dataset | n_train | n_test | n_features | n_labels | label_cardinality | n_configs | seconds | all_negative_test_hl | all_negative_test_subset_acc |
|---|---|---|---|---|---|---|---|---|---|
| emotions | 391 | 202 | 72 | 6 | 1.8133 | 1440 | 12.4000 | 0.3292 | 0.0000 |
| scene | 1211 | 1196 | 294 | 6 | 1.0619 | 1440 | 92.7000 | 0.1810 | 0.0000 |
| yeast | 1500 | 917 | 103 | 14 | 4.2280 | 1440 | 146.3000 | 0.3037 | 0.0000 |
| medical | 333 | 645 | 1449 | 45 | 1.2553 | 1440 | 39.9000 | 0.0276 | 0.0000 |
| enron | 1123 | 579 | 1001 | 53 | 3.3865 | 1440 | 50.4000 | 0.0634 | 0.0000 |
| corel5k | 4500 | 500 | 499 | 374 | 3.5216 | 1440 | 2204.3000 | 0.0094 | 0.0000 |
| bibtex | 4880 | 2515 | 1836 | 159 | 2.3803 | 1440 | 1341.7000 | 0.0154 | 0.0000 |
