# FSD 多二维码方向检测（YUV + ordered_corners）

本项目将 create_Mb_Tiny_RFB_fd_3_nodilation 改造成单阶段多二维码检测器：

- 输入：三通道 YUV444，N×3×320×240
- 输出：confidence [N,4720,2] 与 ordered_corners [N,4720,8]
- 同一张图支持 0～N 个二维码
- 不输出独立 bbox；匹配、NMS、评估所需 bbox 均由四角点 min/max 派生
- 每个角点保持二维码自身语义顺序 P0(TL)→P1(TR)→P2(BR)→P3(BL)

## 数据格式

~~~json
{
  "image": "images/000001.jpg",
  "width": 240,
  "height": 320,
  "instances": [
    {
      "label": "qrcode",
      "corners": [[20,30],[80,30],[80,90],[20,90]],
      "corner_order": ["qr_top_left","qr_top_right","qr_bottom_right","qr_bottom_left"]
    }
  ]
}
~~~

负样本使用 "instances":[]。读取器也兼容旧版单二维码 corners[4,2] 和多二维码
corners[M,4,2]。

## 多目标训练

每个 prior 与全部 GT 的派生 bbox 计算 IoU，每个 prior 只分配给一个 GT，并强制每个 GT
至少匹配一个唯一 prior。只有正 prior 回归其对应实例的 8 个角点；hard-negative mining
也会从零二维码图片中选择背景 prior，能学习压制台球、球网等误检。

## 从单 Y 模型迁移

模型首个卷积由 1 输入通道改为 3 输入通道。加载旧单 Y checkpoint 时，原权重复制到
Y 通道，U/V 权重置零，随后训练学习色度信息。旧的单 Y corners(8) 二维码模型可以直接
作为 resume 权重；原人脸 bbox(4) 模型也可作为 pretrained-fd 权重。

## 快速检查

~~~bash
python tests/test_qr_geometry.py
python tests/test_qr_model_adapter.py
python tests/test_qr_dataset.py
python -m py_compile *.py tests/*.py
bash -n run_prepare_dataset.sh run_train.sh run_infer.sh
~~~

## 视频推理

~~~bash
python infer_video.py \
  --fsd-repo /mnt/ssd1/z00919662/AI-face-detect/ultraface_3323_ref_param \
  --checkpoint /path/to/qr_fsd_best.pth \
  --input input.mp4 --output output.mp4 \
  --score-threshold 0.8 --max-detections 20
~~~

完整服务器执行步骤见 CODEAGENT_RUN_QR_FSD.md。
