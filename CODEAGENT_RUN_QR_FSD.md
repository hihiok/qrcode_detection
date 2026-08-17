# CodeAgent：FSD 多二维码检测 + 三通道 YUV

## 1. 目标

基于 create_Mb_Tiny_RFB_fd_3_nodilation：

1. 输入固定 W×H=240×320，Tensor 为 N×3×320×240。
2. transform 固定为 OpenCV BGR→YUV444，三个通道均除以 255；训练、评估、图片和视频推理一致。
3. 每图允许 0～N 个二维码。
4. 模型只输出 confidence(2)+ordered_corners(8)，不增加 bbox、landmark 或第二阶段。
5. P0/P1/P2/P3 固定为二维码自身 TL/TR/BR/BL，不能按图像位置重新排序。
6. priors 为 4720；bbox 只从角点派生，用于多 GT 匹配和 NMS。

## 2. 网络、代理、SSL 与获取代码

执行任何 git/pip/curl 网络操作前，先读取服务器本地配置；不要把代理账号或密码提交到 GitHub：

~~~bash
test -f /mnt/ssd1/z00919662/qrcode_detection/proxy.md
test -f /mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md
sed -n '1,220p' /mnt/ssd1/z00919662/qrcode_detection/proxy.md
sed -n '1,220p' /mnt/ssd1/z00919662/qrcode_detection/CODEAGENT_DISABLE_SSL.md
~~~

按这两个本地文件设置 http_proxy/https_proxy、Git proxy，并在公司证书链导致 clone 失败时，
于 git clone 之前执行：

~~~bash
git config --global http.sslVerify false
~~~

仅用于该受控服务器环境；最终报告说明实际应用了哪些代理和 SSL 设置，但不得回显密码。

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
export BARBER_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_accepted_1027_240x320
export BOOFCV_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset/boofcv_qr_multi_240x320_rotation_validated
export MENDELEY_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset/mendeley_qr_multi_240x320_rotation_validated

# 此路径已由用户在服务器确认；直接使用现有 4000/400/400 合成数据，禁止重新生成或覆盖。
export SYNTH_ROOT=/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_full_data

export OLD_QR_CHECKPOINT=$FSD_ROOT/models/qr_fsd_240x320_corners8/qr_fsd_best.pth
export OUTPUT_DIR=$FSD_ROOT/models/qr_fsd_multi_yuv
cd "$PROJECT_ROOT"
~~~

确认 $FSD_ROOT/vision/ssd/mb_tiny_RFB_fd_3.py 存在，并确认以上四个数据集目录均存在。
合成集必须严格核对 train=4000、val=400、test=400；若数量不同，停止并人工确认。

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

### 5.1 已准备好的真实数据集与旧合成集

正式训练必须联合使用以下三个已经完成方向解析和旋转一致性验证的真实数据集，并加入下方一个合成数据集：

~~~text
/mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_accepted_1027_240x320
/mnt/ssd1/z00919662/qrcode_detection/dataset/boofcv_qr_multi_240x320_rotation_validated
/mnt/ssd1/z00919662/qrcode_detection/dataset/mendeley_qr_multi_240x320_rotation_validated
~~~

另外加入此前训练使用、现已确认路径的单二维码合成集：

~~~text
/mnt/ssd1/z00919662/qrcode_detection/dataset/qr_full_data
~~~

因此正式训练固定为“三个真实数据集 + 一个合成数据集”。必须验证 `$SYNTH_ROOT`
的 train/val/test 图片数为 4000/400/400，禁止重新生成或覆盖。

每个根目录必须包含 `train/annotations.jsonl`、`val/annotations.jsonl` 和
`test/annotations.jsonl`，图片路径相对各自 split 目录。不要重新推断或按图像坐标重排
P0/P1/P2/P3，也不要覆盖四个源目录。训练脚本允许重复传入
`--data-root`，会分别加载各数据源并用 `ConcatDataset` 合并 train 和 val。

正式训练前逐一执行：

~~~bash
for dataset_root in "$BARBER_ROOT" "$BOOFCV_ROOT" "$MENDELEY_ROOT" "$SYNTH_ROOT"; do
  test -f "$dataset_root/train/annotations.jsonl"
  test -f "$dataset_root/val/annotations.jsonl"
  test -f "$dataset_root/test/annotations.jsonl"
  python validate_qr_dataset.py --data-root "$dataset_root" --visualize 30
done
~~~

额外断言合成集 train/val/test 图片数分别为 4000/400/400。检查跨数据源和跨 split
的图片 SHA256；同一张图及同一视频的相邻帧不得跨 train/val/test。
报告各数据源每个 split 的图片数、二维码实例数、负样本数和
每图实例数直方图。若某个目录结构不符合规范，先停止并在报告中说明，不得静默跳过。

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
  --data-root "$BARBER_ROOT" \
  --data-root "$BOOFCV_ROOT" \
  --data-root "$MENDELEY_ROOT" \
  --data-root "$SYNTH_ROOT" \
  --checkpoint-dir "$OUTPUT_DIR" \
  --resume "$OLD_QR_CHECKPOINT" \
  --input-mode yuv --input-size-key 240 \
  --batch-size 64 --num-workers 16 --epochs 200 --gpus 0,1 \
  > "$OUTPUT_DIR/nohup.out" 2>&1 &
~~~

不要只看最后 epoch，使用验证集 total loss 最低的 qr_fsd_best.pth。

## 10. 多二维码评估

真实集与旧合成集分别评估，不能只汇报合并指标：

~~~bash
for dataset_root in "$BARBER_ROOT" "$BOOFCV_ROOT" "$MENDELEY_ROOT" "$SYNTH_ROOT"; do
  dataset_name=$(basename "$dataset_root")
  python eval_fsd_qr.py \
    --fsd-repo "$FSD_ROOT" \
    --checkpoint "$OUTPUT_DIR/qr_fsd_best.pth" \
    --data-root "$dataset_root" --split test \
    --input-mode yuv --device cuda:0 \
    --score-threshold 0.5 --match-iou-threshold 0.5 \
    --max-detections 20 \
    --output "$OUTPUT_DIR/test_metrics_${dataset_name}.json"
done
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
- 四个数据源各 split 的图片/实例/负样本统计；合成集必须确认 4000/400/400
- YUV transform 数值范围与首层 1→3 加载日志
- 模型输入输出 shape 与 4720 priors
- best epoch、训练/验证曲线
- 多二维码和无二维码测试指标
- 至少 50 张可视化，单列台球、球网等 hard-negative 结果
- MP4 输出路径和仍存在的误检/漏检
