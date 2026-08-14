# Identifiable away-from-prior switch

## resource_first

| Method | Switch pre/post | Reward DiD [95% CI] | Estimate pre/post | MAE pre/post | Early/late post reward | Early/late post MAE | Post agr/joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| framework_v8 | 16.040/17.030 | 2.470 [-3.177, 8.117] | 0.388/0.364 | 0.445/0.586 | 19.560/14.500 | 0.572/0.600 | 0.960/42.347 |
| framework_v8_no_cross_episode | 16.900/18.350 | 1.990 [-7.450, 11.430] | 0.404/0.447 | 0.430/0.503 | 18.200/18.500 | 0.517/0.489 | 0.980/44.885 |

Continuous − no-cross reward DiD: **0.480** [-12.596, 13.556].

## resource_second

| Method | Switch pre/post | Reward DiD [95% CI] | Estimate pre/post | MAE pre/post | Early/late post reward | Early/late post MAE | Post agr/joint |
|---|---:|---:|---:|---:|---:|---:|---:|
| framework_v8 | 20.670/25.030 | 4.480 [0.070, 8.890] | 0.553/0.529 | 0.386/0.479 | 24.300/25.760 | 0.485/0.472 | 1.000/69.889 |
| framework_v8_no_cross_episode | 20.470/19.190 | -2.350 [-15.994, 11.294] | 0.521/0.538 | 0.354/0.488 | 19.180/19.200 | 0.504/0.471 | 0.900/63.497 |

Continuous − no-cross reward DiD: **6.830** [-8.845, 22.505].
