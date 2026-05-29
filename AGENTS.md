# AGENTS.md

## Project Language

- 默认使用中文沟通和记录项目结论。
- 涉及代码、接口、文件名、命令、模型名时保留英文原名。

## Project Ground Rules

- 这是辅助抓取视觉与控制项目，当前视觉路线是：
  - 固定外部 RGB 相机，eye-to-hand。
  - VGA 640x480 OV5640 UYVY 原图作为唯一图像坐标母图。
  - 触发式模型 A 做语义检测/定位。
  - 触发式模型 B 做 ROI RGB 抓取矩形估计。
  - 平面标定负责 VGA 像素坐标到机械臂桌面坐标映射。
  - 轨迹规划、RL、力触觉闭环负责接触阶段抓取与安全递送。
- 不把尚未实测的内容写成事实。所有板端、RUHMI、Tensor Arena、延迟、内存、后处理成本都必须以实验记录为准。
- 每次实验尽量记录：
  - 日期、硬件、固件/SDK 版本、模型来源。
  - 输入图像规格。
  - 构建和烧录步骤。
  - 串口/LCD/日志现象。
  - 推理延迟、内存占用、输出格式。
  - 问题、下一步。
- 对外部官方示例、模型、SDK 的结论优先引用官方文档或本地实际运行结果。

## CodeGraph

This project has a CodeGraph MCP server (`codegraph_*` tools) configured. CodeGraph is a tree-sitter-parsed knowledge graph of every symbol, edge, and file. Reads are sub-millisecond and return structural information grep cannot.

### When to prefer codegraph over native search

Use codegraph for **structural** questions: what calls what, what would break, where is X defined, what is X's signature. Use native grep/read only for **literal text** queries such as string contents, comments, log messages, or after a specific file is already open.

| Question | Tool |
|---|---|
| "Where is X defined?" / "Find symbol named X" | `codegraph_search` |
| "What calls function Y?" | `codegraph_callers` |
| "What does Y call?" | `codegraph_callees` |
| "How does X reach/become Y? / trace the flow from X to Y" | `codegraph_trace` |
| "What would break if I changed Z?" | `codegraph_impact` |
| "Show me Y's signature / source / docstring" | `codegraph_node` |
| "Give me focused context for a task/area" | `codegraph_context` |
| "See several related symbols' source at once" | `codegraph_explore` |
| "What files exist under path/" | `codegraph_files` |
| "Is the index healthy?" | `codegraph_status` |

### Rules of thumb

- Answer structural questions directly with CodeGraph.
- For architecture questions, start with `codegraph_context`, then use one `codegraph_explore` if source bodies are needed.
- For flow tracing, start with `codegraph_trace` instead of reconstructing paths manually.
- Do not grep first when looking up a symbol by name.
- Do not loop `codegraph_node` over many symbols; use `codegraph_explore`.
- The index can lag file writes by about 500 ms.

### If `.codegraph/` does not exist

Ask the user: "I notice this project doesn't have CodeGraph initialized. Want me to run `codegraph init -i` to build the index?"
