# Agent 前端设计方案（审核稿）

## 1. 方向定义
- 方向名：Cold Relay Minimalism（冷继电极简）
- 核心目标：把 Agent 的“工作流可见性”变成界面主轴，而非传统卡片看板。
- 目标情绪：克制、可信、可追踪。

## 2. DFII 评估
- Aesthetic Impact: 5
- Context Fit: 5
- Implementation Feasibility: 4
- Performance Safety: 4
- Consistency Risk: 2
- 评分：15/15（按公式封顶，Excellent）

## 3. 差异化锚点
- 记忆点：`9 节点 Pulse Rail`（映射 LangGraph 9 节点）。
- 非模板化策略：用“workflow-as-interface”替代“dashboard-as-cards”。

## 4. 设计令牌
- 字体：`Space Grotesk`（展示）+ `Noto Sans SC`（正文）+ `IBM Plex Mono`（数据）
- 颜色：深炭背景 + 冷青强调 + 雾灰中性
- 节奏：4/8/12/20/32/52
- 动效：仅保留流程脉冲与关键 hover，避免微动效泛滥

## 5. 页面信息架构
- `/` Console：Query Composer + Pulse Rail + Report Viewer
- `/memory`：偏好写入、画像读取
- `/ingest`：文档回填
- `/settings`：运行时与设计令牌快照

## 6. 代码落地点
- 入口：`frontend/src/main.tsx`
- 壳层：`frontend/src/App.tsx`
- 核心页面：`frontend/src/pages/DashboardPage.tsx`
- 工作流锚点：`frontend/src/components/PulseRail.tsx`
- 视觉系统：`frontend/src/styles/tokens.css` + `frontend/src/styles/global.css`

## 7. 审核重点
1. 是否认可“Pulse Rail 作为主视觉锚点”。
2. 是否保持当前冷色极简（如需更强品牌色可替换 `--accent`）。
3. 是否需要新增历史报告列表或 trace 时间线页面。
