# BarBeR ZXing-C++ label repair — VGG geometry gate

Audit-only recovery of QR semantic corner order for the remaining BarBeR images.

Safety invariants:

- V3's 799 images / 915 instances are immutable calibration gold.
- Original BarBeR VIA/VGG polygons are rebuilt from the source images and are
  the only automatic coordinate source.
- Processed dataset TXT labels are immutable legacy inputs only. They are not
  used by audit, review rendering, recovered labels, or finalization.
- ZXing coordinates are evidence only; output vertices reuse transformed VGG
  polygon tokens.
- All 799 V3 gold images / 915 instances must match transformed VGG vertices
  before the 420-image audit is allowed to start.
- Calibration must pass before audit starts.
- Only images whose every instance is Grade ZA enter combined_proposed.
- No OpenCV import, no payload text access, no apply mode.

The implementation discovers V3 proposed labels recursively below the V3 work
directory, but requires the expected 1219/799/915/420 counts. A mismatch is a
fatal error. The count override exists only for unit-sized development fixtures
and must not be used for production.

Run:

    python -m barber_label_repair_zxingcpp_v2.cli doctor \
      --dataset /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_qr_multi_240x320_rotation_validated \
      --barber-root /mnt/ssd1/z00919662/qrcode_detection/dataset/BarBeR \
      --v3-work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_v3_work \
      --work /mnt/ssd1/z00919662/qrcode_detection/dataset/barber_label_repair_zxingcpp_v3_geometry_work

See `CODEAGENT_RUN_BARBER_ZXINGCPP_V3_GEOMETRY.md` for the required two-step
doctor/audit flow and safe reuse of the already-passed V2 calibration.

After audit, generate the human-review pack for every remaining ZB/ZM image:

    python -m barber_label_repair_zxingcpp_v2.review pack \
      --dataset /path/to/barber_qr_multi_240x320_rotation_validated \
      --work /path/to/barber_label_repair_zxingcpp_v2_work

The pack contains annotated preview pages and a pending decision JSONL. Automatic
ZA labels are never mixed with unreviewed ZB/ZM labels. After a human fills every
decision, `review finalize` creates a new complete 1219-image dataset; it never
overwrites the source dataset. Degenerate polygons and parse errors require an
explicit corrected label under `manual_corrected_labels/<split>/labels/`.

The integration test creates a synthetic QR entirely in memory and sends it
through the real zxing-cpp 3.1.1 binding. It also verifies that blank RGB input
returns no result with and without return_errors.
