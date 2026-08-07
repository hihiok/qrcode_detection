# CodeAgent：FSD 多二维码检测 + 三通道 YUV

## 1. 目标

基于 create_Mb_Tiny_RFB_fd_3_nodilation：

1. 输入固定 W×H=240×320，Tensor 为 N×3×320×240。
2. transform 固定为 OpenCV BGR→YUV444，三个通道均除以 255；训练、评估、图片和视频推理一致。
3. 每图允许 0～N 个二维码。
4. 模型只输出 confidence(2)+ordered_corners(8)，不增加 bbox、landmark 或第二阶段。
5. P0/P1/P2/P3 固定为二维码自身 TL/TR/BR/BL，不能按图像位置重新排序。
6. priors 为 4720；bbox 只从角点派生，用于多 GT 匹配和 NMS。

## 2. 获取代码

首次执行：

~~~bash
PROJECT_ROOT=/mnt/ssd1/z00919662/qrcode_detection
git clone \
  --branch agent/multi-qr-yuv \
  --single-branch \
  https://github.com/hihiok/qrcode_detection.git \
  "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
git rev-parse --abbrev-ref HEAD
git status --short
~~~

已有目录时不要删除用户数据或覆盖未提交修改：

~~~bash
cd /mnt/ssd1/z00919662/qrcode_detection
git status --short
# 仅在输出为空时继续
git fetch origin agent/multi-qr-yuv
git switch agent/multi-qr-yuv
git pull --ff-only
~~~

## 3. 路径

~~~bash
export PROJECT_ROOT=/mnt/ssd1/z00919662/qrcode_detection
export FSD_ROOT=/mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param
export DATA_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset
export OLD_QR_CHECKPOINT=$FSD_ROOT/models/qr_fsd_240x320_corners8/qr_fsd_best.pth
export OUTPUT_DIR=$FSD_ROOT/models/qr_fsd_multi_yuv
cd "$PROJECT_ROOT"
~~~

确认 $FSD_ROOT/vision/ssd/mb_tiny_RFB_fd_3.py 存在。

## 4. 静态与单元测试

沿用原 ultraface 环境，不升级 PyTorch：

~~~bash
python -V
python -c "import torch, cv2, numpy; print(torch.__version__, cv2.__version__, numpy.__version__)"
python -m py_compile *.py tests/*.py
bash -n run_prepare_dataset.sh run_train.sh run_infer.sh
python tests/test_qr_geometry.py
python tests/test_qr_model_adapter.py
python tests/test_qr_dataset.py
~~~

必须确认：

~~~text
feature maps: 40x30, 20x15, 10x8, 5x4
priors: 4720
input: [1,3,320,240]
confidence: [1,4720,2]
ordered_corners: [1,4720,8]
bbox output: NONE
~~~

## 5. 标注格式

推荐 annotations.jsonl 每行：

~~~json
{"image":"images/a.jpg","width":240,"height":320,"instances":[{"label":"qrcode","corners":[[20,30],[80,30],[80,90],[20,90]],"corner_order":["qr_top_left","qr_top_right","qr_bottom_right","qr_bottom_left"]}]}
~~~

- 多二维码：instances 放多个对象。
- 无二维码负样本："instances":[]。
- 兼容旧格式：单个 corners[4,2] 或多个 corners[M,4,2]。
- LabelMe 每个二维码画一个四点 polygon，点击顺序必须 P0→P1→P2→P3；一张图可画多个。

若需要从 LabelMe 转换：

~~~bash
python prepare_qr_dataset.py labelme \
  --input /absolute/path/to/labelme_source \
  --output "$DATA_ROOT" \
  --rotate-cw landscape
~~~

验证并人工检查：

~~~bash
python validate_qr_dataset.py --data-root "$DATA_ROOT" --visualize 50
~~~

报告 train/val/test 的图片数、二维码实例数、0 二维码负样本数和每图二维码数量直方图。
同一视频相邻帧不能跨 train/val/test。

## 6. 多二维码合成冒烟数据

~~~bash
SMOKE_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset_smoke_multi
python prepare_qr_dataset.py synthetic \
  --output "$SMOKE_ROOT" \
  --background-dir /data/pub1/z00919662/dataset/coco_ADE_12cls \
  --train-count 40 --val-count 8 --test-count 8 \
  --max-qrs-per-image 5 --negative-ratio 0.15 \
  --rotate-landscape-cw
python validate_qr_dataset.py --data-root "$SMOKE_ROOT" --visualize 16
~~~

## 7. checkpoint 迁移

优先从之前的 $OLD_QR_CHECKPOINT 微调。代码会自动把旧输入首层 [out,1,k,k] 转为
[out,3,k,k]：Y 权重复制旧权重，U/V 权重置零。除首层 1→3 外，resume 必须严格加载。

若从原人脸 FSD bbox checkpoint 初始化，使用 --pretrained-fd：首层同样自动 1→3；
旧 regression bbox(4) head 跳过，新 corners(8) head 初始化；其他不匹配立即报错。

## 8. 一轮冒烟训练

~~~bash
mkdir -p "$OUTPUT_DIR/smoke"
CUDA_VISIBLE_DEVICES=0 python -u train_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" \
  --data-root "$SMOKE_ROOT" \
  --checkpoint-dir "$OUTPUT_DIR/smoke" \
  --resume "$OLD_QR_CHECKPOINT" \
  --input-mode yuv --input-size-key 240 \
  --batch-size 4 --num-workers 0 --epochs 1 --gpus 0
~~~

确认 total/corner/classification loss 有限，且含负样本的 batch 也有 classification loss。

## 9. 正式训练

~~~bash
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0,1 nohup python -u train_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" \
  --data-root "$DATA_ROOT" \
  --checkpoint-dir "$OUTPUT_DIR" \
  --resume "$OLD_QR_CHECKPOINT" \
  --input-mode yuv --input-size-key 240 \
  --batch-size 64 --num-workers 16 --epochs 200 --gpus 0,1 \
  > "$OUTPUT_DIR/nohup.out" 2>&1 &
~~~

不要只看最后 epoch，使用验证集 total loss 最低的 qr_fsd_best.pth。

## 10. 多二维码评估

~~~bash
python eval_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" \
  --checkpoint "$OUTPUT_DIR/qr_fsd_best.pth" \
  --data-root "$DATA_ROOT" --split test \
  --input-mode yuv --device cuda:0 \
  --score-threshold 0.5 --match-iou-threshold 0.5 \
  --max-detections 20 \
  --output "$OUTPUT_DIR/test_metrics.json"
~~~

报告 TP/FP/FN、precision、recall、F1、负样本图片误检率、bbox/polygon IoU、严格 P0
和 ordered-corner error、success@5px/10px。预测与 GT 必须一对一匹配。

## 11. 图片与 MP4 推理

~~~bash
python infer_fsd_qr.py \
  --fsd-repo "$FSD_ROOT" --checkpoint "$OUTPUT_DIR/qr_fsd_best.pth" \
  --input /path/to/images --output "$OUTPUT_DIR/image_preview" \
  --input-mode yuv --score-threshold 0.8 --max-detections 20

python infer_video.py \
  --fsd-repo "$FSD_ROOT" --checkpoint "$OUTPUT_DIR/qr_fsd_best.pth" \
  --input /mnt/ssd1/z00919662/qrcode_detection/video_input/vrtest.mp4 \
  --output /mnt/ssd1/z00919662/qrcode_detection/video_output/vrtest_multi_yuv.mp4 \
  --score-threshold 0.8 --nms-threshold 0.3 --max-detections 20
~~~

任意分辨率输入会等比例 letterbox 到 240×320，再把角点映射回原图，不能强行 resize。

## 12. 最终报告

在 $OUTPUT_DIR/CODEAGENT_REPORT.md 写明：

- 实际 commit、路径、Python/PyTorch/CUDA/GPU
- 三个 split 的图片/实例/负样本统计
- YUV transform 数值范围与首层 1→3 加载日志
- 模型输入输出 shape 与 4720 priors
- best epoch、训练/验证曲线
- 多二维码和无二维码测试指标
- 至少 50 张可视化，单列台球、球网等 hard-negative 结果
- MP4 输出路径和仍存在的误检/漏检
