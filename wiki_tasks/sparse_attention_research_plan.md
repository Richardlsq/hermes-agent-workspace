# 稀疏注意力研究任务追踪

## 任务状态
- 任务开始: 2026-06-01
- 任务目标: 撰写稀疏注意力(Sparse Attention)综述
- 输出位置: hermes-agent-workspace/wiki/sparse_attention/
  - 实际可写位置: hermes-agent-workspace repo(用户后续同步到 /mnt/e/SynologyDrive/.../wiki)
- 目标行数: ≥ 300 行

## 关键 arXiv 论文编号(已记忆核对)
- Sparse Transformer (OpenAI, Child et al. 2019): 1904.10509
- Reformer (Kitaev et al. 2020): 2001.04451
- Longformer (Beltagy et al. 2020): 2004.05150
- BigBird (Zaheer et al. 2020): 2007.14062
- Performer (Choromanski et al. 2020): 2009.14794
- Linformer (Wang et al. 2020): 2006.04768
- Mistral 7B (Jiang et al. 2023): 2310.06825
- MoBA (Mixture of Block Attention, DeepSeek 2025): 2502.13189
- NSA (Native Sparse Attention, DeepSeek 2025): 2502.11089
- Mamba (Gu & Dao 2023): 2312.00752
- Mamba-2 (Dao & Gu 2024): 2405.21060

## MiniMax MSA(M3, 2026-06-01 发布)
- 在训练数据截止 2026-01 之后,用户提供技术规格:
  - 1M 上下文 prefill 9.7× / decode 15.6× 加速
  - 单 token 计算量压缩到 1/20
  - 块级稀疏 + 动态筛选
  - 方法名: MiniMax Sparse Attention (MSA)

## 进展
- [x] 任务接收与计划
- [ ] 撰写综述 markdown
- [ ] 验证行数 ≥ 300
- [ ] 在 hermes-agent-workspace 创建索引
