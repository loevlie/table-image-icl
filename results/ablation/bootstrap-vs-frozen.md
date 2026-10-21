**CBIS-DDSM** (AUROC, 3 splits; reference: TabICL v2, table + frozen DINOv2 PCA)

| method | AUROC [95% CI] | vs reference [95% CI] | P(better) |
|---|---|---|---|
| baseline: ours, table only | 0.643 [0.625, 0.662] | -0.156 [-0.175, -0.136] | 0.00 |
| baseline: ours, table + image | 0.728 [0.712, 0.744] | -0.071 [-0.085, -0.055] | 0.00 |
| baseline: ours, image of another row | 0.627 [0.607, 0.647] | -0.172 [-0.190, -0.152] | 0.00 |
| baseline: ours, photos of random rows | 0.651 [0.632, 0.669] | -0.148 [-0.166, -0.130] | 0.00 |
| TabPFN v2, table only | 0.659 [0.641, 0.678] | -0.140 [-0.157, -0.121] | 0.00 |
| TabPFN v2, table + frozen DINOv2 PCA | 0.796 [0.781, 0.809] | -0.003 [-0.009, +0.003] | 0.15 |
| TabICL v2, table only | 0.638 [0.618, 0.656] | -0.162 [-0.182, -0.142] | 0.00 |
| TabICL v2, table + frozen DINOv2 PCA | 0.799 [0.784, 0.812] | +0.000 [+0.000, +0.000] | 0.00 |
| baseline-seed1: ours, table only | 0.646 [0.628, 0.665] | -0.153 [-0.171, -0.134] | 0.00 |
| baseline-seed1: ours, table + image | 0.683 [0.668, 0.699] | -0.116 [-0.132, -0.098] | 0.00 |
| baseline-seed1: ours, image of another row | 0.646 [0.627, 0.666] | -0.154 [-0.172, -0.134] | 0.00 |
| baseline-seed1: ours, photos of random rows | 0.657 [0.639, 0.674] | -0.142 [-0.159, -0.124] | 0.00 |
| distractor: ours, table only | 0.648 [0.629, 0.666] | -0.152 [-0.170, -0.132] | 0.00 |
| distractor: ours, table + image | 0.704 [0.687, 0.720] | -0.095 [-0.110, -0.078] | 0.00 |
| distractor: ours, image of another row | 0.619 [0.600, 0.638] | -0.180 [-0.199, -0.159] | 0.00 |
| distractor: ours, photos of random rows | 0.651 [0.632, 0.669] | -0.148 [-0.167, -0.130] | 0.00 |
| fine-grained: ours, table only | 0.647 [0.628, 0.665] | -0.153 [-0.171, -0.133] | 0.00 |
| fine-grained: ours, table + image | 0.683 [0.665, 0.700] | -0.116 [-0.133, -0.098] | 0.00 |
| fine-grained: ours, image of another row | 0.640 [0.621, 0.660] | -0.159 [-0.179, -0.138] | 0.00 |
| fine-grained: ours, photos of random rows | 0.654 [0.636, 0.672] | -0.145 [-0.163, -0.126] | 0.00 |
| sibling-ordinal: ours, table only | 0.642 [0.623, 0.660] | -0.158 [-0.177, -0.138] | 0.00 |
| sibling-ordinal: ours, table + image | 0.708 [0.692, 0.723] | -0.091 [-0.107, -0.075] | 0.00 |
| sibling-ordinal: ours, image of another row | 0.637 [0.618, 0.657] | -0.162 [-0.181, -0.142] | 0.00 |
| sibling-ordinal: ours, photos of random rows | 0.652 [0.633, 0.670] | -0.148 [-0.166, -0.130] | 0.00 |
| distinct: ours, table only | 0.647 [0.629, 0.666] | -0.152 [-0.171, -0.132] | 0.00 |
| distinct: ours, table + image | 0.702 [0.685, 0.718] | -0.097 [-0.113, -0.081] | 0.00 |
| distinct: ours, image of another row | 0.632 [0.613, 0.652] | -0.167 [-0.186, -0.147] | 0.00 |
| distinct: ours, photos of random rows | 0.655 [0.637, 0.673] | -0.144 [-0.161, -0.126] | 0.00 |
| learnable: ours, table only | 0.649 [0.630, 0.668] | -0.151 [-0.170, -0.130] | 0.00 |
| learnable: ours, table + image | 0.711 [0.695, 0.728] | -0.088 [-0.103, -0.071] | 0.00 |
| learnable: ours, image of another row | 0.632 [0.614, 0.652] | -0.167 [-0.186, -0.146] | 0.00 |
| learnable: ours, photos of random rows | 0.654 [0.636, 0.671] | -0.146 [-0.163, -0.128] | 0.00 |

**HAM10000** (AUROC, 3 splits; reference: TabICL v2, table + frozen DINOv2 PCA)

| method | AUROC [95% CI] | vs reference [95% CI] | P(better) |
|---|---|---|---|
| baseline: ours, table only | 0.735 [0.722, 0.749] | -0.127 [-0.143, -0.111] | 0.00 |
| baseline: ours, table + image | 0.874 [0.865, 0.884] | +0.012 [+0.001, +0.022] | 0.98 |
| baseline: ours, image of another row | 0.620 [0.606, 0.635] | -0.243 [-0.259, -0.225] | 0.00 |
| baseline: ours, photos of random rows | 0.737 [0.725, 0.751] | -0.125 [-0.140, -0.110] | 0.00 |
| TabPFN v2, table only | 0.745 [0.732, 0.759] | -0.118 [-0.133, -0.101] | 0.00 |
| TabPFN v2, table + frozen DINOv2 PCA | 0.852 [0.842, 0.861] | -0.011 [-0.016, -0.006] | 0.00 |
| TabICL v2, table only | 0.741 [0.728, 0.755] | -0.122 [-0.137, -0.106] | 0.00 |
| TabICL v2, table + frozen DINOv2 PCA | 0.863 [0.853, 0.872] | +0.000 [+0.000, +0.000] | 0.00 |
| baseline-seed1: ours, table only | 0.737 [0.725, 0.751] | -0.125 [-0.141, -0.110] | 0.00 |
| baseline-seed1: ours, table + image | 0.878 [0.869, 0.887] | +0.016 [+0.005, +0.027] | 1.00 |
| baseline-seed1: ours, image of another row | 0.613 [0.600, 0.628] | -0.249 [-0.265, -0.233] | 0.00 |
| baseline-seed1: ours, photos of random rows | 0.740 [0.727, 0.754] | -0.123 [-0.138, -0.108] | 0.00 |
| distractor: ours, table only | 0.737 [0.723, 0.751] | -0.125 [-0.141, -0.110] | 0.00 |
| distractor: ours, table + image | 0.872 [0.863, 0.882] | +0.010 [-0.001, +0.020] | 0.97 |
| distractor: ours, image of another row | 0.621 [0.607, 0.636] | -0.241 [-0.257, -0.224] | 0.00 |
| distractor: ours, photos of random rows | 0.735 [0.721, 0.750] | -0.127 [-0.143, -0.111] | 0.00 |
| fine-grained: ours, table only | 0.737 [0.724, 0.751] | -0.126 [-0.141, -0.110] | 0.00 |
| fine-grained: ours, table + image | 0.879 [0.870, 0.888] | +0.017 [+0.006, +0.027] | 1.00 |
| fine-grained: ours, image of another row | 0.620 [0.606, 0.635] | -0.243 [-0.258, -0.226] | 0.00 |
| fine-grained: ours, photos of random rows | 0.747 [0.734, 0.760] | -0.116 [-0.130, -0.100] | 0.00 |
| sibling-ordinal: ours, table only | 0.736 [0.723, 0.750] | -0.127 [-0.142, -0.111] | 0.00 |
| sibling-ordinal: ours, table + image | 0.872 [0.861, 0.881] | +0.009 [-0.002, +0.019] | 0.95 |
| sibling-ordinal: ours, image of another row | 0.621 [0.608, 0.636] | -0.241 [-0.256, -0.224] | 0.00 |
| sibling-ordinal: ours, photos of random rows | 0.738 [0.725, 0.753] | -0.124 [-0.139, -0.108] | 0.00 |
| distinct: ours, table only | 0.736 [0.723, 0.750] | -0.127 [-0.142, -0.111] | 0.00 |
| distinct: ours, table + image | 0.869 [0.859, 0.879] | +0.007 [-0.004, +0.017] | 0.90 |
| distinct: ours, image of another row | 0.635 [0.621, 0.649] | -0.228 [-0.243, -0.211] | 0.00 |
| distinct: ours, photos of random rows | 0.736 [0.723, 0.751] | -0.126 [-0.141, -0.111] | 0.00 |
| learnable: ours, table only | 0.740 [0.726, 0.753] | -0.123 [-0.139, -0.108] | 0.00 |
| learnable: ours, table + image | 0.876 [0.866, 0.885] | +0.013 [+0.002, +0.024] | 0.99 |
| learnable: ours, image of another row | 0.621 [0.607, 0.636] | -0.241 [-0.257, -0.225] | 0.00 |
| learnable: ours, photos of random rows | 0.740 [0.727, 0.754] | -0.122 [-0.138, -0.107] | 0.00 |

**CheXpert** (AUROC, 3 splits; reference: TabICL v2, table + frozen DINOv2 PCA)

| method | AUROC [95% CI] | vs reference [95% CI] | P(better) |
|---|---|---|---|
| baseline: ours, table only | 0.720 [0.713, 0.726] | -0.009 [-0.013, -0.006] | 0.00 |
| baseline: ours, table + image | 0.722 [0.716, 0.728] | -0.006 [-0.010, -0.003] | 0.00 |
| baseline: ours, image of another row | 0.719 [0.713, 0.725] | -0.010 [-0.014, -0.006] | 0.00 |
| baseline: ours, photos of random rows | 0.721 [0.715, 0.727] | -0.008 [-0.012, -0.004] | 0.00 |
| TabPFN v2, table only | 0.728 [0.722, 0.734] | -0.001 [-0.005, +0.002] | 0.26 |
| TabPFN v2, table + frozen DINOv2 PCA | 0.734 [0.728, 0.740] | +0.005 [+0.002, +0.008] | 1.00 |
| TabICL v2, table only | 0.722 [0.715, 0.728] | -0.007 [-0.011, -0.004] | 0.00 |
| TabICL v2, table + frozen DINOv2 PCA | 0.729 [0.723, 0.735] | +0.000 [+0.000, +0.000] | 0.00 |
| baseline-seed1: ours, table only | 0.720 [0.714, 0.726] | -0.009 [-0.013, -0.005] | 0.00 |
| baseline-seed1: ours, table + image | 0.722 [0.716, 0.728] | -0.006 [-0.010, -0.003] | 0.00 |
| baseline-seed1: ours, image of another row | 0.720 [0.714, 0.726] | -0.009 [-0.013, -0.005] | 0.00 |
| baseline-seed1: ours, photos of random rows | 0.720 [0.713, 0.725] | -0.009 [-0.013, -0.005] | 0.00 |
| distractor: ours, table only | 0.721 [0.715, 0.727] | -0.007 [-0.011, -0.004] | 0.00 |
| distractor: ours, table + image | 0.722 [0.716, 0.728] | -0.007 [-0.011, -0.004] | 0.00 |
| distractor: ours, image of another row | 0.720 [0.714, 0.725] | -0.009 [-0.013, -0.005] | 0.00 |
| distractor: ours, photos of random rows | 0.720 [0.714, 0.726] | -0.009 [-0.013, -0.006] | 0.00 |
| fine-grained: ours, table only | 0.718 [0.712, 0.724] | -0.011 [-0.014, -0.007] | 0.00 |
| fine-grained: ours, table + image | 0.721 [0.714, 0.726] | -0.008 [-0.012, -0.004] | 0.00 |
| fine-grained: ours, image of another row | 0.716 [0.710, 0.722] | -0.013 [-0.017, -0.009] | 0.00 |
| fine-grained: ours, photos of random rows | 0.718 [0.712, 0.724] | -0.010 [-0.014, -0.007] | 0.00 |
| sibling-ordinal: ours, table only | 0.718 [0.712, 0.724] | -0.010 [-0.014, -0.007] | 0.00 |
| sibling-ordinal: ours, table + image | 0.721 [0.714, 0.727] | -0.008 [-0.012, -0.004] | 0.00 |
| sibling-ordinal: ours, image of another row | 0.718 [0.711, 0.724] | -0.011 [-0.015, -0.007] | 0.00 |
| sibling-ordinal: ours, photos of random rows | 0.720 [0.714, 0.726] | -0.009 [-0.013, -0.005] | 0.00 |
| distinct: ours, table only | 0.720 [0.713, 0.725] | -0.009 [-0.013, -0.006] | 0.00 |
| distinct: ours, table + image | 0.722 [0.716, 0.728] | -0.007 [-0.010, -0.003] | 0.00 |
| distinct: ours, image of another row | 0.719 [0.713, 0.725] | -0.010 [-0.014, -0.006] | 0.00 |
| distinct: ours, photos of random rows | 0.720 [0.714, 0.726] | -0.009 [-0.013, -0.004] | 0.00 |
| learnable: ours, table only | 0.719 [0.713, 0.725] | -0.010 [-0.013, -0.006] | 0.00 |
| learnable: ours, table + image | 0.719 [0.712, 0.725] | -0.010 [-0.014, -0.007] | 0.00 |
| learnable: ours, image of another row | 0.715 [0.709, 0.721] | -0.014 [-0.018, -0.010] | 0.00 |
| learnable: ours, photos of random rows | 0.717 [0.711, 0.723] | -0.011 [-0.015, -0.008] | 0.00 |

**Mango Mass** (RMSE, 3 splits; reference: TabICL v2, table + frozen DINOv2 PCA)

| method | RMSE [95% CI] | vs reference [95% CI] | P(better) |
|---|---|---|---|
| baseline: ours, table only | 0.076 [0.071, 0.081] | +0.029 [+0.024, +0.033] | 0.00 |
| baseline: ours, table + image | 0.044 [0.041, 0.047] | -0.003 [-0.006, +0.000] | 0.95 |
| baseline: ours, image of another row | 0.103 [0.098, 0.109] | +0.056 [+0.051, +0.062] | 0.00 |
| baseline: ours, photos of random rows | 0.076 [0.071, 0.081] | +0.029 [+0.025, +0.034] | 0.00 |
| TabPFN v2, table only | 0.076 [0.071, 0.080] | +0.029 [+0.024, +0.033] | 0.00 |
| TabPFN v2, table + frozen DINOv2 PCA | 0.051 [0.047, 0.054] | +0.003 [+0.002, +0.005] | 0.00 |
| TabICL v2, table only | 0.076 [0.071, 0.081] | +0.029 [+0.024, +0.033] | 0.00 |
| TabICL v2, table + frozen DINOv2 PCA | 0.047 [0.044, 0.050] | +0.000 [+0.000, +0.000] | 0.00 |
| baseline-seed1: ours, table only | 0.076 [0.071, 0.081] | +0.029 [+0.024, +0.033] | 0.00 |
| baseline-seed1: ours, table + image | 0.041 [0.038, 0.043] | -0.007 [-0.009, -0.004] | 1.00 |
| baseline-seed1: ours, image of another row | 0.101 [0.096, 0.107] | +0.054 [+0.048, +0.060] | 0.00 |
| baseline-seed1: ours, photos of random rows | 0.076 [0.071, 0.081] | +0.029 [+0.025, +0.034] | 0.00 |
| distractor: ours, table only | 0.076 [0.071, 0.081] | +0.029 [+0.024, +0.033] | 0.00 |
| distractor: ours, table + image | 0.045 [0.042, 0.048] | -0.002 [-0.005, +0.001] | 0.91 |
| distractor: ours, image of another row | 0.096 [0.091, 0.101] | +0.049 [+0.044, +0.054] | 0.00 |
| distractor: ours, photos of random rows | 0.077 [0.071, 0.081] | +0.029 [+0.025, +0.034] | 0.00 |
| fine-grained: ours, table only | 0.076 [0.071, 0.081] | +0.029 [+0.024, +0.033] | 0.00 |
| fine-grained: ours, table + image | 0.047 [0.044, 0.051] | +0.000 [-0.003, +0.004] | 0.41 |
| fine-grained: ours, image of another row | 0.088 [0.083, 0.094] | +0.041 [+0.037, +0.046] | 0.00 |
| fine-grained: ours, photos of random rows | 0.076 [0.071, 0.081] | +0.029 [+0.025, +0.034] | 0.00 |
| sibling-ordinal: ours, table only | 0.076 [0.071, 0.081] | +0.029 [+0.024, +0.034] | 0.00 |
| sibling-ordinal: ours, table + image | 0.041 [0.038, 0.044] | -0.006 [-0.009, -0.003] | 1.00 |
| sibling-ordinal: ours, image of another row | 0.104 [0.098, 0.110] | +0.057 [+0.051, +0.063] | 0.00 |
| sibling-ordinal: ours, photos of random rows | 0.078 [0.073, 0.083] | +0.031 [+0.026, +0.035] | 0.00 |
| distinct: ours, table only | 0.076 [0.071, 0.081] | +0.029 [+0.024, +0.033] | 0.00 |
| distinct: ours, table + image | 0.045 [0.042, 0.047] | -0.002 [-0.006, +0.001] | 0.92 |
| distinct: ours, image of another row | 0.104 [0.098, 0.109] | +0.057 [+0.051, +0.062] | 0.00 |
| distinct: ours, photos of random rows | 0.077 [0.072, 0.081] | +0.029 [+0.025, +0.034] | 0.00 |
| learnable: ours, table only | 0.076 [0.071, 0.081] | +0.029 [+0.024, +0.033] | 0.00 |
| learnable: ours, table + image | 0.040 [0.037, 0.043] | -0.007 [-0.010, -0.004] | 1.00 |
| learnable: ours, image of another row | 0.110 [0.104, 0.116] | +0.063 [+0.057, +0.069] | 0.00 |
| learnable: ours, photos of random rows | 0.077 [0.072, 0.081] | +0.030 [+0.025, +0.034] | 0.00 |
