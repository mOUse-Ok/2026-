# M6C-B Offline Replay

该目录是隔离的 Python 离线实现，不被 CMake、llama-cli、`ExpertHintQueue` 或 runtime priority mode 引用。

实现边界：

- 一个 bounded Task store；
- Legacy/Aging 两个带 position map 的 indexed binary heap；
- S0 与 synthetic-only S1 共用结构事务；
- A.3 Evidence Hash、reconstructability 与固定 service-slot S0 replay；
- 不运行推理，不进行参数化 S1 Evidence 比较，不形成性能结论。

运行定向测试：

```bash
python3 -m unittest discover \
  -s llama.cpp/trace/tests \
  -p 'test_m6c_offline*.py' -v
```

从仓库根目录运行 30-Run S0 Gate，输出目录必须是一个尚不存在的新目录。M6C-B.2 的 v4 运行会逐 stream 执行 write、flush、close，然后重新打开最终路径验证行数、JSON、size 和 SHA；不会把打开 writer 的 Hash 写入报告：

```bash
PYTHONPATH=llama.cpp/trace python3 -m m6c_offline.runner \
  --output-dir llama.cpp/trace_output/<new-m6c-b-output-dir>
```

runner 写出 `m6c_b_report.json` 后，会用独立实现重新读取 Evidence index、机器报告和全部最终 decision stream，并写出 `artifact_validation_report.json`。也可独立重复执行 validator，输出必须使用新路径：

```bash
PYTHONPATH=llama.cpp/trace python3 -m m6c_offline.artifact_validator \
  --report llama.cpp/trace_output/<m6c-b-output-dir>/m6c_b_report.json \
  --output /tmp/<new-artifact-validation-report>.json
```

默认策略只声明 close 后重新读取的本地文件完整性，不请求 `fsync`，也不声称覆盖所有存储层的持久性。

所有输出固定声明：

```text
counterfactual_type = fixed_arrival_fixed_service_slot_policy_replay
physical_system_reexecuted = false
performance_claim = false
```

M6C-B.2 的成功枚举是 `M6C_B_EVIDENCE_REPAIRED_AND_CONFIRMED`。即使成功，也必须停止并等待人工决定；本工具不会创建或执行 M6C-C。
