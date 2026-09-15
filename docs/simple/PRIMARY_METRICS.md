# Universal buyer primary-outcome report

Only Simple Env `reward` and AgenticPay `buyer_score` are primary outcomes.

## simple — reward

| Variant | Clusters | Mean | vs best(CoT, full) per cluster | vs universal_framework_v5_9 |
|---|---:|---:|---:|---:|
| cot_prompt | 128 | 0.4065 | -0.1006 [-0.1280, -0.0748] | -0.0916 [-0.1373, -0.0472] |
| direct_prompt | 128 | 0.0961 | -0.4110 [-0.4579, -0.3630] | -0.4020 [-0.4559, -0.3478] |
| full_framework | 128 | 0.4464 | -0.0607 [-0.0800, -0.0434] | -0.0517 [-0.0924, -0.0109] |
| universal_framework_v5_9 | 128 | 0.4981 | -0.0089 [-0.0498, +0.0319] | — |
| universal_framework_v6_3_conservative_awr | 128 | 0.5130 | +0.0060 [-0.0390, +0.0501] | +0.0149 [-0.0296, +0.0589] |

## agenticpay — buyer_score

| Variant | Clusters | Mean | vs best(CoT, full) per cluster | vs universal_framework_v1 |
|---|---:|---:|---:|---:|
| cot_prompt | 28 | 31.7199 | -6.3053 [-10.4527, -2.5961] | +15.9892 [+5.8345, +26.9635] |
| direct_prompt | 28 | 32.9531 | -5.0721 [-10.7318, +0.4266] | +17.2225 [+7.1038, +28.0168] |
| full_framework | 28 | 21.3108 | -16.7144 [-27.4859, -7.3591] | +5.5801 [-5.6253, +16.7353] |
| universal_framework_v1 | 28 | 15.7307 | -22.2945 [-33.0293, -12.3483] | — |
| universal_framework_v2_conservative_awr | 28 | 18.1775 | -19.8477 [-31.3662, -8.7753] | +2.4468 [-6.4345, +10.6209] |
