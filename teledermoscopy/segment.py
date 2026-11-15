"""CLI segmentation cho một ảnh: python -m teledermoscopy.segment --help."""

import argparse
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SegFormer MiT-B2 — smoke test phân đoạn một ảnh; ngưỡng cố định 0.5.")
    parser.add_argument("--image", type=Path, required=True, help="Đường dẫn ảnh đầu vào.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint real_only/best.pt đã xác minh.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Thư mục lưu mask, overlay và JSON.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    try:
        from teledermoscopy.vision.segmentation import SegmentationPredictor, read_rgb_image, save_result

        rgb = read_rgb_image(args.image)
        predictor = SegmentationPredictor(args.checkpoint, args.device)
        print(f"Thiết bị thực tế: {predictor.device}", flush=True)
        result = predictor.predict(rgb)
        summary = save_result(result, args.image, args.checkpoint, args.output_dir)
        if summary["mask_state"] in {"empty", "full"}:
            print(f"Lưu ý: mask {summary['mask_state']}; giữ nguyên kết quả và ngưỡng 0.5.", file=sys.stderr)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as error:
        print(f"Lỗi ({type(error).__name__}): {error}", file=sys.stderr)
        if args.device == "cuda":
            print("Lệnh yêu cầu CUDA đã thất bại; không tự chuyển sang CPU. Có thể chạy riêng --device cpu.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
