import os
import torch
import traceback
from peft import PeftModel
from utils.utils import check_compute_dtype
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from config.log_config import logger
import numpy as np

def load_model(
        model_dir: str,
        model_name: str,
        cache_dir: str = './pretrained_models',
        use_fl32: bool = True,
        use_4bit_quantization: bool = False,
    ):
    """
    Load the base model and apply the fine-tuned PEFT adapters from the folder.

    Args:
        model_dir (str): Path to the directory where the fine-tuned model is stored.
        model_name (str): Name of the base model (required for loading the base model).
        cache_dir (str): Path to the directory where the base model is cached.
        use_4bit_quantization (bool): Whether to use 4bit quantization when loading the model (must match when training).

    Returns:
        model (transformers.PreTrainedModel): Loaded model with PEFT adapters applied.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.

    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir, use_fast=True, local_files_only=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                if "[PAD]" not in tokenizer.vocab:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                else:
                    tokenizer.pad_token = "[PAD]"
                logger.warning(f"{os.path.basename(__file__)}: Đã thêm pad_token vào tokenizer từ {model_dir}")

        logger.info(f"{os.path.basename(__file__)}: Đã tải Tokenizer từ {model_dir}")
        if use_fl32:
            compute_dtype = torch.float32
        else:
            compute_dtype = check_compute_dtype()
        logger.info(f"{os.path.basename(__file__)}: Đã thiết lập compute_dtype={compute_dtype}")
        bnb_config = None
        model_load_kwargs = {
            'cache_dir': cache_dir,
            'device_map': "auto",
            'torch_dtype': compute_dtype,
            'num_labels': 2,
            'local_files_only': False
        }
        if use_4bit_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_threshold=6.0
            )
            model_load_kwargs["quantization_config"] = bnb_config
            logger.info(f"{os.path.basename(__file__)}: Đã tạo cấu hình Bits and Bytes cho 4bit quantization với bnb_4bit_compute_dtype={compute_dtype}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_load_kwargs
        )
        logger.info(f"{os.path.basename(__file__)}: Đã tải mô hình base {model_name}")

        current_model_vocab_size = model.get_input_embeddings().num_embeddings
        if current_model_vocab_size < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"{os.path.basename(__file__)}: Đã điều chỉnh kích thước embedding đầu vào của mô hình base từ {current_model_vocab_size} thành {len(tokenizer)}.")
    
        model = PeftModel.from_pretrained(model, model_dir)
        logger.info(f"{os.path.basename(__file__)}: Đã tải PEFT adapters từ {model_dir} và áp dụng vào mô hình base")
        model = model.merge_and_unload()
        logger.info("Đã merge PEFT adapters vào mô hình base.")
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        elif torch.backends.mps.is_available():
            model = model.to("mps")
        logger.info(f"{os.path.basename(__file__)}: Đã tải hoàn tất mô hình từ {model_dir}")

        return model, tokenizer
    
    except Exception as e:
        logger.error(f"{os.path.basename(__file__)}: Lỗi khi tải mô hình từ {model_dir}: {e}")
        logger.error(f'{os.path.basename(__file__)}: Traceback: {traceback.format_exc()}')
        return None, None


def predict_single_model(text: str, model, tokenizer, max_length: int = 512):
    """
    Perform inference for a document on a loaded model. Returns probabilities for each class.
    
    Args:
        text (str): Text to predict.
        model (transformers.PreTrainedModel): Loaded model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
        max_length (int): Maximum length when tokenizing.

    Returns:
        np.ndarray: Probabilities for each class.
    """
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)

        return probabilities.squeeze().cpu().type(torch.float32).numpy()

    except Exception as e:
        logger.error(f"Lỗi khi thực hiện inference cho văn bản '{text[:50]}...' với mô hình {model.__class__.__name__}: {e}")
        logger.error(f'{os.path.basename(__file__)}: Traceback: {traceback.format_exc()}')
        return None
    

def ensemble_predict(model_dirs: list[str], base_model_names: list[str], texts: list[str], max_length: int = 512, use_fl32: bool = True, use_4bit_quantization: bool = True):
    """
    Perform predictions using Ensemble (average probability) technique on multiple models.

    Args:
        model_dirs (list[str]): List of paths to the directory where the fine-tuned model is stored.
        base_model_names (list[str]): List of names of the corresponding base models (required for loading the base model).
        texts (list[str]): List of texts to predict.
        max_length (int): Maximum length when tokenizing.
        use_4bit_quantization (bool): Whether to use 4bit quantization when loading the model (must match when training).

    Returns:
        list[int]: List of final predictions (0 or 1) for each text.
        list[np.ndarray]: List of average probabilities for each text.
    """
    if len(model_dirs) != len(base_model_names):
        logger.error(f"{os.path.basename(__file__)}: Số lượng thư mục mô hình và tên mô hình base phải khớp nhau.")
        return [], []

    loaded_models = []
    for model_dir, base_name in zip(model_dirs, base_model_names):
        model, tokenizer = load_model(
            model_dir=os.path.join(os.getcwd(), model_dir),
            model_name=base_name,
            use_4bit_quantization=use_4bit_quantization,
            use_fl32=use_fl32
        )
        if model and tokenizer:
            loaded_models.append((model, tokenizer, base_name)) # Lưu cả tên base model để dễ debug
        else:
            logger.warning(f"{os.path.basename(__file__)}: Không thể tải mô hình từ {model_dir}. Bỏ qua mô hình này trong ensemble.")

    if not loaded_models:
        logger.error(f"{os.path.basename(__file__)}: Không có mô hình nào được tải thành công. Không thể thực hiện ensemble.")
        return [], []

    logger.info(f"{os.path.basename(__file__)}: Đã tải thành công {len(loaded_models)} mô hình cho ensemble.")

    all_texts_avg_probs = []

    for i, text in enumerate(texts):
        logger.info(f"{os.path.basename(__file__)}: Đang xử lý văn bản {i+1}/{len(texts)}: '{text[:100]}...'")
        model_probs = []

        for model, tokenizer, model_name in loaded_models:
            probs = predict_single_model(text, model, tokenizer, max_length)
            if probs is not None:
                model_probs.append(probs)
                logger.info(f"{os.path.basename(__file__)}: - Mô hình {model_name}: Xác suất = {probs}")

        if not model_probs:
            logger.warning(f"{os.path.basename(__file__)}: Không nhận được dự đoán từ bất kỳ mô hình nào cho văn bản '{text[:100]}...'. Bỏ qua văn bản này.")
            all_texts_avg_probs.append(np.array([0.5, 0.5]))
            continue

        averaged_probs = np.mean(model_probs, axis=0)
        all_texts_avg_probs.append(averaged_probs)
        logger.info(f"{os.path.basename(__file__)}: -> Kết quả Ensemble (xác suất trung bình): {averaged_probs}")

    final_predictions = [np.argmax(probs) for probs in all_texts_avg_probs]

    return final_predictions, all_texts_avg_probs

if __name__ == "__main__":
    example_data = [
        'Text to test',
        'Another text to test',
        'Yet another text to test'
    ]
    predictions, avg_probabilities = ensemble_predict(
        model_dirs=['results/distilbert-base-uncased', 'results/microsoft_deberta-v3-large'],
        base_model_names=['distilbert-base-uncased', 'microsoft/deberta-v3-large'],
        texts=example_data,
        max_length=512,
        use_4bit_quantization=False,
        use_fl32=True
    )
    print("\n--- Kết quả dự đoán Ensemble ---")
    if predictions:
        predictions_int = [int(p) for p in predictions]
        for text, prediction, probs in zip(example_data, predictions_int, avg_probabilities):
            label_text = "AI tạo ra" if prediction == 1 else ("Con người tạo ra" if prediction == 0 else "Không xác định")
            print(f"Văn bản: '{text}'")
            print(f"  Dự đoán cuối cùng: {prediction} ({label_text})")
            print(f"  Xác suất trung bình (Lớp 0, Lớp 1): {probs}")
            print("-" * 20)
    else:
        print("Không có dự đoán nào được thực hiện do không tải được mô hình nào.")
