"""AgriSmart Shield – lõi xử lý dùng chung cho huấn luyện, đánh giá và suy luận.

Các module trong gói này CHỈ phụ thuộc numpy/opencv (không phụ thuộc torch/ultralytics) để kiểm thử
đơn vị độc lập. Máy chủ trang web (web/) dùng trực tiếp các module này: một bản cài đặt duy nhất cho
Severity Index, Knowledge Engine và giải mã mô hình. onnxruntime / ai-edge-litert / ultralytics chỉ được
nạp khi thật sự dùng (seg_runtime.py).
"""

__version__ = "3.0.0"

LEAF_CLASS = "leaf"
