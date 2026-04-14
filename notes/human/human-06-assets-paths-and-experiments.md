# Human Assets Paths And Experiments

## 导航

- 文档类型：`human` 阶段文档
- 对应 AI 文档：[../ai/ai-06-assets-paths-and-experiments.md](../ai/ai-06-assets-paths-and-experiments.md)
- 上一篇：[human-05-ppo-and-runner.md](human-05-ppo-and-runner.md)
- 下一篇：[human-07-manual-tuning-guide.md](human-07-manual-tuning-guide.md)
- 总索引：[../index.md](../index.md)

## 作用

说明当前仓库里哪些目录保存模型、资产、日志、截图和参考资料，以及哪些目录默认不能当成主项目改动目标。

## Mermaid 路径关系图

```mermaid
graph LR
    scripts["训练/测试脚本\n../../Go2Pvcnn/scripts/"]
    assets["运行资产\n../../assets/"]
    logs["训练日志与模型\n../../logs/"]
    other["外部模型/权重\n../../other_model/"]
    images["截图与可视化产物\n../../furniture_test_images/"]
    notes["知识索引\n../"]
    raw["参考实现\n../../raw/"]
    ref["只读参考\n../../onlyReference/"]

    scripts -->|"读取 robot / terrain / object 资产"| assets
    scripts -->|"保存 run 目录 / checkpoint"| logs
    scripts -->|"可选加载额外模型"| other
    scripts -->|"测试时落图像/视频"| images
    notes -->|"引用主线代码与结果路径"| logs
    notes -->|"标记 reference-only 边界"| raw
    notes -->|"标记 reference-only 边界"| ref
```

## 重点目录

- [assets](../../assets)
- [logs](../../logs)
- [other_model](../../other_model)
- [furniture_test_images](../../furniture_test_images)
- [raw](../../raw)
- [onlyReference](../../onlyReference)

## 上游输入

- 训练和测试脚本写出的产物
- 手工准备的 USD 资产和 checkpoint

## 下游消费者

- 恢复训练
- 回放、分析和人工检查
- 后续实验复现实验环境

## 待补充

- 当前项目依赖的关键资产清单
- 哪些路径是机器相关绝对路径
- 哪些目录必须视为 reference-only
