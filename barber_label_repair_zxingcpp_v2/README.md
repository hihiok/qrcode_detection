# BarBeR ZXing-C++ label repair V2

Audit-only recovery of QR semantic corner order for the remaining BarBeR images.

Safety invariants:

- V3's 799 images / 915 instances are immutable calibration gold.
- The original BarBeR polygon is the only coordinate source.
- ZXing coordinates are evidence only; output vertices reuse original label tokens.
- Calibration must pass before audit starts.
- Only images whose every instance is Grade ZA enter combined_proposed.
- No OpenCV import, no payload text access, no apply mode.

The implementation discovers V3 proposed labels recursively below the V3 work
directory, but requires the expected 1219/799/915/420 counts. A mismatch is a
fatal error. The count override exists only for unit-sized development fixtures
and must not be used for production.

Run:

    python -m barber_label_repair_zxingcpp_v2.cli all \
      --dataset /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_validated \
      --v3-work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_v3_work \
      --work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v2_work

If calibration fails, exit code is 2 and the 420-image audit is not run.

The integration test creates a synthetic QR entirely in memory and sends it
through the real zxing-cpp 3.1.1 binding. It also verifies that blank RGB input
returns no result with and without return_errors.
