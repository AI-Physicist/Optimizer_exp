# Real Text 1000-Step Multi-Seed Summary

| optimizer | num_seeds | seeds | final_loss_mean | final_loss_std | final_test_loss_mean | final_test_ece_mean | final_test_acc_mean | final_param_norm_l2_mean | avg_step_time_mean | peak_memory_mb_max | max_step |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adamw | 3 | 1,2,3 | 1.882399 | 0.054142 | 1.976876 | 0.027979 | 0.430339 | 314.877440 | 0.008417 | 338.01 | 1000 |
| adafactor | 3 | 1,2,3 | 2.373334 | 0.006285 | 2.396520 | 0.038506 | 0.305054 | 953.322904 | 0.016245 | 325.27 | 1000 |
| sgd | 3 | 1,2,3 | 2.438027 | 0.003895 | 2.467166 | 0.020486 | 0.311279 | 117.060054 | 0.008116 | 331.61 | 1000 |
| rmsprop | 3 | 1,2,3 | 2.554173 | 0.038953 | 2.652665 | 0.073781 | 0.253723 | 289.527384 | 0.008416 | 331.61 | 1000 |
