import os
from src.train import train
from config.log_config import logger
from src.inference import ensemble_predict
from datasets import load_dataset
import traceback

def main(data_path):
    models = ['distilbert-base-uncased', 'microsoft/deberta-v3-large', 'roberta-large']
    base_output_dir = 'results'

    # Tải tập dữ liệu và chia thành tập train và val
    try:
        dataset = load_dataset(data_path.split('.')[-1], data_files=data_path)
        split = dataset['train'].train_test_split(test_size=0.1)
        train_data = split['train'][split.column_names['train'][0]]
        train_labels = split['train'][split.column_names['train'][1]]
        val_data = split['test'][split.column_names['test'][0]]
        val_labels = split['test'][split.column_names['test'][1]]
        logger.info(f'{os.path.basename(__file__)}: Đã tải tập dữ liệu từ {data_path} và chia thành tập train={len(train_data)} và val={len(val_data)}')
    except Exception as e:
        logger.error(f'{os.path.basename(__file__)}: Lỗi khi tải tập dữ liệu từ {data_path}: {e}')
        logger.error(f'{os.path.basename(__file__)}: Traceback: {traceback.format_exc()}')
        raise e
    
     # --- Bước 1: Huấn luyện các mô hình ---
    logger.info(f"{os.path.basename(__file__)}: Bắt đầu quá trình Huấn luyện")
    trained_model_dirs = []
    for model_name in models:
        model_output_dir = os.path.join(base_output_dir, model_name.replace('/', '_'))
        os.makedirs(model_output_dir, exist_ok=True)

        logger.info(f"{os.path.basename(__file__)}: Bắt đầu huấn luyện mô hình: {model_name} và lưu vào thư mục: {model_output_dir} ---")

        try:
            path_model = train(
                pretrained_model_name_or_path=model_name,
                train_data=train_data,
                train_labels=train_labels,
                val_data=val_data,
                val_labels=val_labels,
                output_dir=model_output_dir,
                num_train_epochs=1,
                batch_size=2,
                use_4bit_quantization=False
            )

            if path_model:
                print(f"Đã lưu mô hình {model_name} thành công tại {path_model}")
                logger.info(f'{os.path.basename(__file__)}: Đã lưu mô hình {model_name} thành công tại {path_model}')
                trained_model_dirs.append(path_model)
            else:
                logger.error(f'{os.path.basename(__file__)}: Huấn luyện mô hình {model_name} thất bại.')

        except Exception as e:
            logger.error(f'{os.path.basename(__file__)}: Quá trình huấn luyện mô hình {model_name} gặp lỗi: {e}')
            logger.error(f'{os.path.basename(__file__)}: Traceback: {traceback.format_exc()}')

    if len(trained_model_dirs) < len(models):
        logger.warning(f"{os.path.basename(__file__)}: Không phải tất cả các mô hình đều được huấn luyện thành công.Chỉ có {', '.join(trained_model_dirs)} mô hình sẽ được sử dụng cho ensemble.")

     # --- Bước 2: Thực hiện Ensemble Inference ---
    logger.info(f"{os.path.basename(__file__)}:\n--- Bắt đầu quá trình Ensemble Inference ---")

    texts_to_predict = [
        "Đây là một ví dụ văn bản tích cực cho việc phát triển mã nguồn mở.",
        "Đây là một ví dụ văn bản tiêu cực, gặp lỗi khi cài đặt thư viện.",
        "Một văn bản khác cần phân loại, liên quan đến tài liệu hướng dẫn.",
        "Fix bugs in the latest release.",
        "The new feature works perfectly!"
    ]
    ensemble_base_model_names = [os.path.basename(dir_path) for dir_path in trained_model_dirs]
    name_mapping = {m.replace('/', '_'): m for m in models}
    ensemble_base_model_names_mapped = [name_mapping.get(dir_name, dir_name) for dir_name in ensemble_base_model_names]


    predictions, avg_probabilities = ensemble_predict(
        model_dirs=trained_model_dirs,
        base_model_names=ensemble_base_model_names_mapped,
        texts=texts_to_predict,
        max_length=512,
        use_4bit_quantization=False
    )

    print("\n--- Kết quả dự đoán Ensemble ---")
    if predictions:
        predictions_int = [int(p) for p in predictions]
        for text, prediction, probs in zip(texts_to_predict, predictions_int, avg_probabilities):
            label_text = "Tích cực" if prediction == 1 else ("Tiêu cực" if prediction == 0 else "Không xác định")
            print(f"Văn bản: '{text}'")
            print(f"  Dự đoán cuối cùng: {prediction} ({label_text})")
            print(f"  Xác suất trung bình (Lớp 0, Lớp 1): {probs}")
            print("-" * 20)
    else:
        print("Không có dự đoán nào được thực hiện do không tải được mô hình nào.")



if __name__ == "__main__":
    main('data/test.parquet')
