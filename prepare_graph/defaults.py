from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PARENT = Path(__file__).resolve().parents[2]

DEFAULT_YOUCOOK2_ROOT = Path("/home/zhangxuy/MyProject/Dataset/YouCook2")
DEFAULT_OUTPUT_ROOT = Path("/home/zhangxuy/MyProject/stg/output/YouTest")
DEFAULT_SCENE_GRAPH_DIR = DEFAULT_OUTPUT_ROOT / "scene_graphs"
DEFAULT_FRAME_CACHE_DIR = DEFAULT_OUTPUT_ROOT / "intermediate" / "clip_frames"

DEFAULT_HSGG_REPO_ROOT = Path("/home/zhangxuy/MyProject/stg/HSGG-main")
DEFAULT_MODEL_ROOT = Path("/home/zhangxuy/MyProject/models")
DEFAULT_HSGG_LLMDET_MODEL = DEFAULT_MODEL_ROOT / "grounding-dino-base"
DEFAULT_HSGG_LLM_MODEL = DEFAULT_MODEL_ROOT / "LLaVA-OneVision-1.5-8B"
DEFAULT_HSGG_DEPTH_MODEL = DEFAULT_MODEL_ROOT / "DA3-BASE"
DEFAULT_VIDEO_ROOT = DEFAULT_YOUCOOK2_ROOT / "YouCookIIVideos"

DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_ROOT / "manifests" / "youcook2_pilot_manifest.json"
DEFAULT_DEV_DRAFTS_PATH = DEFAULT_OUTPUT_ROOT / "annotations" / "qa_drafts_dev.jsonl"
DEFAULT_TEST_DRAFTS_PATH = DEFAULT_OUTPUT_ROOT / "annotations" / "qa_drafts_test.jsonl"
DEFAULT_PREDICTIONS_PATH = DEFAULT_OUTPUT_ROOT / "predictions" / "pilot_predictions.jsonl"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_ROOT / "metrics" / "pilot_metrics.json"
DEFAULT_CONTEXT_INDEX_PATH = DEFAULT_OUTPUT_ROOT / "contexts" / "context_index.jsonl"

DEFAULT_TOTAL_CLIPS = 12
DEFAULT_DEV_CLIPS = 6
DEFAULT_TEST_CLIPS = 6
DEFAULT_TARGET_FPS = 1.0
DEFAULT_FRAME_STRIDE = 5
DEFAULT_QA_PER_CLIP = 6
DEFAULT_MIN_DURATION_SEC = 60.0
DEFAULT_MAX_DURATION_SEC = 120.0
DEFAULT_TARGET_DURATION_SEC = 90.0
DEFAULT_MIN_STEPS_PER_CLIP = 2
DEFAULT_RANDOM_SEED = 7

DEFAULT_BUDGETS = (320, 640, 960)
DEFAULT_REPRESENTATIONS = ("summary", "frame_wise_graph", "persistent_graph")
DEFAULT_HSGG_CUDA_DEVICE = 0
DEFAULT_HSGG_SAVE_INTERVAL = 20

DEFAULT_SCENE_GRAPH_PATTERN = "{video_id}/{clip_id}"
DEFAULT_ANNOTATION_CANDIDATES = (
    "annotations/youcookii_annotations_trainval.json",
    "annotations/youcook2_annotations_trainval.json",
    "youcookii_annotations_trainval.json",
    "youcook2_annotations_trainval.json",
)

OUTPUT_SUBDIRS = (
    "manifests",
    "annotations",
    "contexts",
    "predictions",
    "metrics",
    "logs",
    "intermediate",
    "scene_graphs",
)

DEFAULT_OPENEQA_DATASET_REPO = "ellisbrown/OpenEQA"
DEFAULT_OPENEQA_RAW_ROOT = REPO_ROOT / "data" / "openeqa" / "raw"
DEFAULT_OPENEQA_PARQUET_DIR = DEFAULT_OPENEQA_RAW_ROOT / "parquet"
DEFAULT_OPENEQA_ARCHIVE_DIR = DEFAULT_OPENEQA_RAW_ROOT / "archives"
DEFAULT_OPENEQA_EPISODES_DIR = DEFAULT_OPENEQA_RAW_ROOT / "episodes"
DEFAULT_OPENEQA_DOWNLOAD_MANIFEST = DEFAULT_OPENEQA_RAW_ROOT / "download_manifest.json"
DEFAULT_OPENEQA_PARQUET_FILENAME = "v0/test-00000-of-00001.parquet"
DEFAULT_OPENEQA_PARQUET_PATH = DEFAULT_OPENEQA_PARQUET_DIR / "v0" / "test-00000-of-00001.parquet"
DEFAULT_OPENEQA_ARCHIVE_FILENAMES = ("hm3d-v0.tar.gz", "scannet-v0.tar.gz")
DEFAULT_OPENEQA_OUTPUT_ROOT = REPO_ROOT / "output" / "openeqa" / "pilot"
DEFAULT_OPENEQA_MANIFEST_PATH = DEFAULT_OPENEQA_OUTPUT_ROOT / "manifests" / "openeqa_pilot_manifest.json"
DEFAULT_OPENEQA_QUESTIONS_PATH = DEFAULT_OPENEQA_OUTPUT_ROOT / "manifests" / "pilot_questions.jsonl"
DEFAULT_OPENEQA_SCENE_GRAPH_DIR = DEFAULT_OPENEQA_OUTPUT_ROOT / "scene_graphs"
DEFAULT_OPENEQA_FRAMES_DIR = DEFAULT_OPENEQA_OUTPUT_ROOT / "intermediate" / "clip_frames"
DEFAULT_OPENEQA_STG_DIR = DEFAULT_OPENEQA_OUTPUT_ROOT / "stg"
DEFAULT_OPENEQA_VISUALIZATION_DIR = DEFAULT_OPENEQA_OUTPUT_ROOT / "visualizations"
DEFAULT_OPENEQA_EVAL_DIR = DEFAULT_OPENEQA_OUTPUT_ROOT / "evaluation"
DEFAULT_OPENEQA_PILOT_PER_SOURCE = 3
