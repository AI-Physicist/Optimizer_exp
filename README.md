我在比较：
  不同 optimizer 在同一个 Transformer 上的优化行为
固定：
  - model
  - data
  - batch size
  - steps
  - scheduler

变化：
  - optimizer
  - learning rate

评价指标：
  - loss vs step
  - final loss
  - 是否发散
  - step time
  - peak memory

1. 先跑通（已完成 ✅）
2. 短程 sweep 找 lr
3. 固定 lr 做正式实验
4. 多 seed
5. 画图 + 总结
