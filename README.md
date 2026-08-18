# FSD 多二维码方向检测（YUV + ordered_corners）

本项目将 `create_Mb_Tiny_RFB_fd_3_nodilation` 改造成单阶段多二维码检测器：

- 输入：三通道 YUV444，`N×3×320×240`
- 输出：`confidence [N,4720,2]` 与 `ordered_corners [N,4720,8]`
- 同一张图支持 0～N 个二维码
- 不输出独立 bbox；匹配、NMS、评估所需 bbox 均由四角点派生
- P0/P1/P2/P3 固定为二维码自身 TL/TR/BR/BL

## Strict V2优化流程

- `CODEAGENT_PREPARE_QR_STRICT_V2_DATASET.md`：锁定两个业务视频，构建增强合成、
  显式负样本和hard-negative，并冻结source-balanced manifest。
- `CODEAGENT_TRAIN_QR_STRICT_V2_OPTIMIZED.md`：审计坐标/YUV/anchor，使用不改变输出结构的
  loss和采样优化训练，只在公共validation选择阈值，最后才一次性运行锁定视频。

相关工具：`strict_eval_guard.py`、`dataset_v2_manifest.py`、
`audit_qr_training_pipeline.py`、`mine_hard_negatives.py`、
`select_qr_threshold.py`和`qr_refine.py`。

## 唯一数据格式

训练、验证和评估只接受 `qr_ordered_corners_v1`：

~~~json
{
  "schema_version": "qr_ordered_corners_v1",
  "image": "images/000001.jpg",
  "width": 240,
  "height": 320,
  "num_qrcodes": 1,
  "instances": [
    {
      "class_id": 0,
      "label": "qrcode",
      "corners": [[20,30],[80,30],[80,90],[20,90]],
      "corner_order": [
        "qr_top_left","qr_top_right","qr_bottom_right","qr_bottom_left"
      ]
    }
  ]
}
~~~

负样本必须显式使用 `num_qrcodes: 0` 和 `instances: []`。未知或旧 schema 直接报错，
不能静默解释成负样本。

四个既有数据集原本分别使用 `instances`、`objects` 和顶层 `corners`。先运行：

~~~bash
python canonicalize_qr_datasets.py \
  --dataset barber=/path/to/barber \
  --dataset boofcv=/path/to/boofcv \
  --dataset mendeley=/path/to/mendeley \
  --dataset synth=/path/to/qr_full_data \
  --deduplicate-identical-images \
  --conflicting-duplicate-keeper \
    barber:val:images/barber_945dae373601f3c3.jpg \
  --output-root /path/to/qr_canonical_v1
~~~

转换不会覆盖源数据集；新目录使用软链接引用源 images/labels，并对 JSON/TXT、角点数量、
方向字段、group split 和精确重复进行校验。显式去重模式只接受 SHA256 与 canonical 标签
同时完全一致的跨 split 重复，并按 `test > val > train` 自动保留。标签冲突默认失败；已人工
确认的单个冲突只能用完整 dataset/split/image keeper 路径显式授权，并写入转换报告。
服务器完整步骤见
`CODEAGENT_CANONICALIZE_QR_DATASETS.md`。

## 多目标训练

每个 prior 与全部 GT 的派生 bbox 计算 IoU，每个 prior 只分配给一个 GT，并强制每个 GT
至少匹配一个唯一 prior。只有正 prior 回归对应实例的8个角点；显式零二维码图片参与
hard-negative mining。

## 从单 Y 模型迁移

模型首个卷积由1输入通道改为3输入通道。加载旧单Y checkpoint 时，原权重复制到Y通道，
U/V权重置零，随后训练学习色度信息。模型输出仍为 `confidence(2)+ordered_corners(8)`。

## 快速检查

~~~bash
python tests/test_qr_schema.py
python tests/test_canonicalize_qr_datasets.py
python tests/test_qr_geometry.py
python tests/test_qr_model_adapter.py
python tests/test_qr_dataset.py
python -m py_compile *.py tests/*.py
bash -n run_prepare_dataset.sh run_train.sh run_infer.sh
~~~

转换验证通过后的训练步骤见 `CODEAGENT_RUN_QR_FSD.md`。

## 视频等比例 Pad 推理

非 3:4 视频可先居中补边到精确 3:4，再由模型缩放到 240×320，避免直接 resize 导致
画面变形。输出尺寸会同时保证为偶数，兼容常用 MP4 codec：

~~~bash
python infer_video.py \
  --fsd-repo /path/to/fsd \
  --checkpoint /path/to/qr_fsd_best.pth \
  --input input.mp4 \
  --padded-input-output input_padded_3x4.mp4 \
  --output detection_padded_3x4.mp4 \
  --pad-to-portrait-3x4 --pad-value 127
~~~

逐帧 JSONL 会记录源尺寸、padded 尺寸、四边 padding 和检测结果。服务器上两个
`from_chenshuo` 视频的完整流程见 `CODEAGENT_INFER_FROM_CHENSHUO_PADDED.md`。
