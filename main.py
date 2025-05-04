from log_config import logger
import os
from train import train

if __name__ == "__main__":
    path_model = train(
        pretrained_model_name_or_path='distilbert-base-uncased',
        data_path='test.parquet',
        output_dir='./result',
        num_train_epochs=3,
        batch_size=4,
        learning_rate=2e-5,
        max_length=512,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        gradient_accumulation_steps=1,
        use_gradient_checkpointing=False,
        num_labels=2
    )
    if path_model:
        logger.info(f'{os.path.basename(__file__)}: Đã lưu mô hình tại {path_model}')
    else:
        logger.error(f'{os.path.basename(__file__)}: Huấn luyện mô hình thất bại')