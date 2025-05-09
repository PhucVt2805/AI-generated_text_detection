import torch
import os
from config.log_config import logger

def check_compute_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            logger.info(f"{os.path.basename(__file__)}: GPU CUDA hỗ trợ BF16, sử dụng BF16")
        else:
            compute_dtype = torch.float16
            logger.info(f"{os.path.basename(__file__)}: GPU CUDA không hỗ trợ BF16, sử dụng FP16")
    else:
        if torch.backends.mps.is_available():
            logger.info(f"{os.path.basename(__file__)}: GPU MPS không hỗ trợ FP16, sử dụng FP32")
        else:
            logger.info(f"{os.path.basename(__file__)}: Không có GPU, sử dụng FP32")
        compute_dtype = torch.float32
    return compute_dtype