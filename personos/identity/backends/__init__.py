"""Pluggable backends for the video pipeline: mm_runner (multimodal LLM), face_detector, voiceprint.

factory.make_backends(profile) assembles them by profile:
- "mock": pure-numpy fakes that download no models and make no network calls, for
  unit tests and smoke runs of the skeleton;
- "real": the real thing (AdaFace/InsightFace detection and recognition + ECAPA
  voiceprints + a hosted multimodal LLM), which needs model weights and a GPU/CPU.

The contract is: face_detector.detect(frame_rgb: ndarray) -> list[FaceDet];
voiceprint.embed(wav_bytes) -> ndarray (normalized);
mm_runner.chat(prompt, *, video_url/images_b64) -> str.
"""
