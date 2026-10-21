"""BƯỚC 2 – Huấn luyện YOLOv8n-seg (Google Colab T4 hoặc GPU NVIDIA cục bộ).

Tương đương câu lệnh trong quy trình:
    yolo task=segment mode=train model=yolov8n-seg.pt data=dataset.yaml epochs=100 imgsz=640 batch=16 save=true
nhưng bổ sung các tham số bắt buộc cho bài toán tính tỷ lệ diện tích:

    overlap_mask=False  <- QUAN TRỌNG. Mặc định True: ultralytics gộp mọi instance vào MỘT mask, instance nhỏ
                           vẽ đè instance lớn => nhãn huấn luyện của "leaf" bị KHOÉT mất phần vết bệnh.
                           Mô hình học ra mask lá "thủng lỗ", làm sai mẫu số Area_Leaf. Tắt để mỗi
                           instance giữ nguyên mask đầy đủ của nó.
    seed, deterministic <- tái lập kết quả khi nộp hồ sơ dự thi.
    patience            <- dừng sớm nếu mAP không cải thiện, tiết kiệm quota GPU Colab.

Ví dụ (Colab, lưu kết quả thẳng vào Google Drive để không mất khi bị ngắt kết nối):
    !pip install -r ai/requirements.txt
    !python ai/scripts/s05_train.py --data /content/datasets/agrismart/dataset.yaml --epochs 100 --batch 16 \
        --project /content/drive/MyDrive/AgriSmart/runs --save-period 10
    # bị ngắt giữa chừng:
    !python ai/scripts/s05_train.py --data /content/datasets/agrismart/dataset.yaml \
        --resume /content/drive/MyDrive/AgriSmart/runs/agrismart_v8n/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default=None, help="0 | cpu | None(tự chọn)")
    ap.add_argument("--project", default="runs/segment")
    ap.add_argument("--name", default="agrismart_v8n")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-period", type=int, default=-1, help="lưu checkpoint mỗi N epoch (-1: tắt)")
    ap.add_argument("--workers", type=int, default=2, help="Colab chỉ có 2 vCPU")
    ap.add_argument("--resume", default=None, metavar="LAST_PT",
                    help="huấn luyện tiếp từ checkpoint (…/weights/last.pt) khi Colab bị ngắt kết nối")
    args = ap.parse_args()

    from ultralytics import YOLO  # import muộn để --help chạy được khi chưa cài torch

    if args.resume:
        # Tiếp tục đúng lần huấn luyện cũ (giữ nguyên tham số, epoch, trạng thái optimizer)
        model = YOLO(args.resume)
        model.train(resume=True, data=args.data)
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        print(json.dumps({"best_weights": str(best), "save_dir": str(model.trainer.save_dir)}, ensure_ascii=False))
        return

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        overlap_mask=False,
        save=True,
        save_period=args.save_period,
        plots=True,
        # Augmentation trực tuyến của ultralytics (bổ sung cho augmentation offline ở s03):
        degrees=0.0,  # đã xoay ±30° offline, không xoay thêm để tránh mất vết bệnh sát mép
        fliplr=0.5,
        hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=10,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(json.dumps({"best_weights": str(best), "save_dir": str(model.trainer.save_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
