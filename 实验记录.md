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

中期汇报：
- 我先搭了一个最小 Transformer 训练系统，在 synthetic next-token 任务上比较了 AdamW、SGD、RMSprop、Adafactor 及其变体。
- 结果表明，几个 optimizer 的差异不大，因为这个任务理论下界已知且较简单，所有合理方法都能逼近下界。
- 不过 RMSprop 类方法略占优，说明逐坐标预条件在这个任务中更关键。
- 下一步我准备迁移到真实文本数据，验证这种排序是否保持。
