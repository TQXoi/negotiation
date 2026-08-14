# Next change

在 canary 与 paired pilot 完成前不做下一项算法修改。根据失败 trajectory，只允许二选一：

1. 若 agreement 仍因 over-anchor 下降，修订 response-support / regret gate；
2. 若收益因过度保守下降，修订 uncertainty schedule，而不改 belief updater。
