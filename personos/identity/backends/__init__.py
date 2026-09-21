"""视频生产管线的可插拔后端:mm_runner(Omni)/ face_detector / voiceprint。

factory.make_backends(profile) 按 profile 装配:
- "mock":纯 numpy 假后端,不下模型、不联网,供单测/骨架冒烟;
- "real":真后端(AdaFace/InsightFace 检测识别 + ECAPA 声纹 + xhs MAAS Omni),需模型权重 + GPU/CPU。

契约(对齐 mneme):face_detector.detect(frame_rgb: ndarray)->list[FaceDet];
voiceprint.embed(wav_bytes)->ndarray(归一化);mm_runner.chat(prompt, *, video_url/images_b64)->str。
"""
