"""身份层:角色实体 + 多模态素材 + 概率身份云(移植自 mneme-release anchor)。

- store.CharacterStore:characters/character_assets/character_cloud 三表的 per-user 存取。
- cloud.CloudEngine:概率身份云,做粗召回排序 + 注册基底,不做判决(判决在 MLLM 仲裁)。
- types:证据值类型(FacePick/VoiceSample/CastEvidence)+ CandidateCard + normalized。
- draft.DraftStore(⑤):身份链/roster/评估/暂存素材的会话态(Redis;禁 sqlite)。
- chains.ChainBook(⑤):链证据台账 + 刷新触发 + 同框碰撞重裁/降级。
- registry.AnchorRegistry(⑥):cast 映射 + 候选构造 + roster 更新。
- inspect(⑥):present_casts + 同框碰撞检测(Rule 6)。
- recognize(⑥):批量 BIND 仲裁协议(build/parse)+ enroll_evidence。
- commit.commit_session(⑦):会话末两阶段终审 + 结算落 MySQL(归属映射 + wearer)。
"""
