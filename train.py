import os
import torch
from log_config import logger
import numpy as np
from datasets import Dataset, ClassLabel, load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer

MODEL_TARGET_LAYER = {
    "microsoft/deberta-v3-large": ["query_proj", "key_proj", "value_proj", "dense"],
    "roberta-large": ["query", "key", "value", "dense"],
    "distilbert-base-uncased": ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
    }


def compute_metrics(eval_pred):
    """
    Computes weighted F1 score and accuracy for a sequence classification task.

    This function is designed to be used as the `compute_metrics` argument
    in a Hugging Face Trainer. It takes the raw model predictions (typically logits)
    and the ground truth labels, calculates the predicted class indices,
    and then computes the weighted F1 score and overall accuracy.

    Args:
        eval_pred (tuple): A tuple containing model outputs and true labels.
            The first element (eval_pred[0]) is typically the raw predictions (logits or probabilities)
            from the model, a numpy array of shape (batch_size, num_labels).
            The second element (eval_pred[1]) is the ground truth labels,
            a numpy array of shape (batch_size,).

    Returns:
        dict: A dictionary containing the evaluation metrics.
            Keys include:
                'f1': The weighted F1 score calculated using `sklearn.metrics.f1_score`.
                'acc': The accuracy score calculated using `sklearn.metrics.accuracy_score`.
    """
    logit, labels = eval_pred
    # Apply argmax to get the predicted class index for each sample
    predictions = np.argmax(logit, axis=-1)

    # Compute weighted F1 score
    f1 = f1_score(labels, predictions, average="weighted")

    # Compute accuracy score
    acc = accuracy_score(labels, predictions)

    # Return metrics in a dictionary
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted'),
    }


def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenizes input text and adds the corresponding labels to the output.

    This function is designed to be used with `datasets.Dataset.map`.
    When `batched=True` is used with the `map` method, the `examples`
    argument will be a dictionary where keys correspond to column names
    (e.g., "text", "label"), and values are lists containing multiple
    examples for that batch.

    Args:
        examples (dict): A dictionary provided by `datasets.map`.
            Expected to contain at least "text" (list of strings)
            and "label" (list of integers or other label types) columns.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use
            for processing the text.
        max_length (int, optional): The maximum sequence length for tokenization.
            Defaults to 512. Texts longer than this will be truncated,
            and shorter texts will be padded to this length.

    Returns:
        dict: A dictionary containing the tokenized inputs and the original labels.
            Includes standard tokenizer outputs like "input_ids" and "attention_mask",
            and a new key "labels" containing the corresponding labels from the input `examples`.
    """
    # Tokenize the text
    tokenized_inputs = tokenizer(
        examples["data"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    tokenized_inputs["labels"] = examples["label"]

    return tokenized_inputs


def train(
        pretrained_model_name_or_path: str,
        data_path: str,
        output_dir: str,
        num_train_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        gradient_accumulation_steps: int = 1,
        use_gradient_checkpointing: bool = False,
        num_labels: int = 2
    ):
    """
    Train a text classification model using the Hugging Face Transformers and PEFT (LoRA) libraries.

    This function loads a pre-trained model, configures LoRA, 4-bit quantization (optional),
    prepares the data by tokenizing, sets the training parameters, and uses the Hugging Face Trainer
    to perform the training and evaluation.

    Args:
        pretrained_model_name_or_path (str): Path or name of the pretrained model
        data_path (str): Path to the dataset file (CSV, JSON, etc.) containing the training and validation data with columns=['data', 'label'].
        output_dir (str): Directory to save the training results (checkpoint, final model).
        num_train_epochs (int, optional): Number of training epochs. Default is 3.
        batch_size (int, optional): Batch size for training and evaluation. Default is 8.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 5e-5.
        max_length (int, optional): Maximum sequence length for tokenization. Default is 512.
        lora_r (int, optional): LoRA rank. Default is 16.
        lora_alpha (int, optional): LoRA alpha. Default is 16.
        lora_dropout (float, optional): LoRA dropout rate. Default is 0.05.
        use_4bit_quantization (bool, optional): Whether to use 4-bit quantization. Default is True.
        gradient_accumulation_steps (int, optional): Number of steps for gradient accumulation. Default is 1.
        use_mixed_precision (bool, optional): Whether to use mixed precision training. Default is True.
        use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Default is False.
        num_labels (int, optional): Number of labels for the classification task. Default is 2.

    Returns:
        None: The function saves the trained model and evaluation results to the specified output directory.
    """

    os.environ['TRANSFORMERS_CACHE'] = './model'

    dataset = load_dataset(data_path.split('.')[-1], data_files=data_path)
    split = dataset['train'].train_test_split(test_size=0.2)
    train_data = split['train']['data']
    train_labels = split['train']['label']
    val_data = split['test']['data']
    val_labels = split['test']['label']
    logger.info(f'{os.path.basename(__file__)}: Đã tải tập dữ liệu từ {data_path} và chia thành tập train={len(train_data)} và val={len(val_data)}')

    # 1. Choose the dtype that matches your hardware
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            use_mixed_precision = True
            use_4bit_quantization = True
            logger.info(f'{os.path.basename(__file__)}: GPU CUDA hỗ trợ BF16')
        else:
            compute_dtype = torch.float16
            use_mixed_precision = True
            use_4bit_quantization = True
            logger.info(f'{os.path.basename(__file__)}: GPU không hỗ trợ BF16, sử dụng FP16')
        quantization = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        compute_dtype = torch.float32
        logger.info(f'{os.path.basename(__file__)}: GPU MPS sử dụng FP32')
        quantization = False
        use_4bit_quantization = False
        use_mixed_precision = False
    else:
        device = torch.device("cpu")
        compute_dtype = torch.float32
        logger.info(f'{os.path.basename(__file__)}: Không có GPU, sử dụng CPU')
        quantization = False
        use_4bit_quantization = False
        use_mixed_precision = False


    # 2. Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info(f'{os.path.basename(__file__)}: Đã thêm pad_token vào tokenizer')
    logger.info(f'{os.path.basename(__file__)}: Đã load Tokenizer từ {pretrained_model_name_or_path}')
    
    bnb_config = None
    model_kwargs = {
        'num_labels': num_labels,
        'device_map': "auto",
    }

    if use_4bit_quantization and quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_threshold=6.0
        )
        model_kwargs['quantization_config'] = bnb_config
        model_kwargs['torch_dtype'] = compute_dtype
        logger.info(f'{os.path.basename(__file__)}: Đã tạo và áp dụng cấu hình Bits and Bytes cho 4bit quantization với bnb_4bit_compute_dtype={compute_dtype}')
    else:
        if use_mixed_precision and device.type != "cpu":
            model_kwargs['torch_dtype'] = compute_dtype
            logger.info(f'{os.path.basename(__file__)}: Đã thiết lập torch_dtype cho mô hình là {compute_dtype} cho mixed precision.')
        else:
            model_kwargs['torch_dtype'] = compute_dtype
            logger.info(f'{os.path.basename(__file__)}: Đã thiết lập torch_dtype cho mô hình là float32')
    
    # 3. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        **model_kwargs,
    )
    logger.info(f'{os.path.basename(__file__)}: Đã tải mô hình {pretrained_model_name_or_path} với cấu hình 4bit quantization')

    current_model_vocab_size = model.get_input_embeddings().num_embeddings
    if current_model_vocab_size < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f'{os.path.basename(__file__)}: Đã điều chỉnh kích thước embedding đầu vào của mô hình từ {current_model_vocab_size} thành {len(tokenizer)} cho phù hợp với tokenizer')
    elif current_model_vocab_size > len(tokenizer):
        logger.warning(f'{os.path.basename(__file__)}: Kích thước embedding đầu vào của mô hình ({current_model_vocab_size}) lớn hơn kích thước từ vựng của tokenizer ({len(tokenizer)}).')

    # 4. LoRA configuration
    target_modules = None
    for prefix, modules in MODEL_TARGET_LAYER.items():
        if pretrained_model_name_or_path.lower().startswith(prefix.lower()):
            target_modules = modules
            logger.info(f'{os.path.basename(__file__)}: Đã xác định các module mục tiêu cho LoRA: {target_modules}')
            break
    if target_modules is None: logger.warning(f'{os.path.basename(__file__)}: Không tìm thấy mô hình nào trong danh sách mục tiêu cho LoRA. Sẽ để theo mặc định.')

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=target_modules,
        modules_to_save=["classifier"]
    )
    logger.info(f'{os.path.basename(__file__)}: Đã tạo cấu hình LoRA với r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, task_type="{lora_config.task_type}"')

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Dataset Creation and Tokenization
    train_datasets = Dataset.from_dict({"data": train_data, "label": train_labels})
    val_datasets = Dataset.from_dict({"data": val_data, "label": val_labels})
    logger.info(f'{os.path.basename(__file__)}: Đã tạo tập dữ liệu huấn luyện ({len(train_datasets)} mẫu) và kiểm tra ({len(val_datasets)} mẫu) từ danh sách văn bản và nhãn')

    tokenizer_train = train_datasets.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        batched=True,
        remove_columns=["data"],
        desc="Tokenizing train dataset"
    )

    tokenizer_val = val_datasets.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        batched=True,
        remove_columns=["data"],
        desc="Tokenizing validation dataset"
    )
    logger.info(f'{os.path.basename(__file__)}: Đã token hóa tập dữ liệu huấn luyện và kiểm tra')

    tokenizer_train = tokenizer_train.cast_column("label", ClassLabel(num_classes=num_labels))
    tokenizer_val = tokenizer_val.cast_column("label", ClassLabel(num_classes=num_labels))
    logger.info(f'{os.path.basename(__file__)}: Đã ép kiểu cột "label" sang ClassLabel.')

    tokenizer_train.set_format("torch")
    tokenizer_val.set_format("torch")
    logger.info(f'{os.path.basename(__file__)}: Đã thiết lập định dạng torch cho tập dữ liệu huấn luyện và kiểm tra')

    # 6. Training Arguments Setup
    if use_mixed_precision:
        if compute_dtype == torch.bfloat16:
            mixed_precision = "bf16"
            logger.info(f'{os.path.basename(__file__)}: Đã thiết lập mixed precision là bf16')
        elif compute_dtype == torch.float16:
            mixed_precision = "fp16"
            logger.info(f'{os.path.basename(__file__)}: Đã thiết lập mixed precision là fp16')
    else:
        mixed_precision = "no"
        logger.info(f'{os.path.basename(__file__)}: Không sử dụng mixed precision')

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True, #Ghi đè lên thư mục đầu ra nếu nó đã tồn tại
        do_train=True, #Bật chế độ huấn luyện
        do_eval=True,  #Bật chế độ đánh giá
        per_device_train_batch_size=batch_size, #Kích thước batch cho mỗi thiết bị trong quá trình huấn luyện
        per_device_eval_batch_size=batch_size, #Kích thước batch cho mỗi thiết bị trong quá trình đánh giá
        gradient_accumulation_steps=gradient_accumulation_steps, #Số bước tích lũy gradient trước khi cập nhật trọng số
        learning_rate=learning_rate, #Tốc độ học
        weight_decay=0.01, #Hệ số giảm trọng số
        max_grad_norm=1.0, #Giới hạn độ lớn gradient
        num_train_epochs=num_train_epochs, #Số lượng epoch huấn luyện
        logging_dir=os.path.join(output_dir, "logs"), #Thư mục để lưu trữ các tệp log
        logging_strategy="epoch", #Chiến lược ghi log
        save_strategy="steps", #Chiến lược lưu mô hình
        save_steps=500, #Số bước để lưu mô hình
        gradient_checkpointing=use_gradient_checkpointing, #Bật gradient checkpointing
        eval_strategy="steps", #Chiến lược đánh giá
        eval_steps=500, #Số bước để đánh giá mô hình
        load_best_model_at_end=True, #Tải mô hình tốt nhất ở cuối quá trình huấn luyện
        metric_for_best_model="eval_f1", #Chỉ số để xác định mô hình tốt nhất
        greater_is_better=True, #Chỉ số lớn hơn là tốt hơn
        fp16=mixed_precision == "fp16", #Sử dụng fp16 nếu mixed_precision là "fp16"
        bf16=mixed_precision == "bf16", #Sử dụng bf16 nếu mixed_precision là "bf16"
        report_to="none", #Không báo cáo đến bất kỳ dịch vụ nào
        label_names=["label"], #Tên cột nhãn
    )

    # 7. Trainer Initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenizer_train,
        eval_dataset=tokenizer_val,
        compute_metrics=compute_metrics,
        processing_class=tokenizer
    )
    logger.info(f'{os.path.basename(__file__)}: Đã tạo Trainer với các tham số huấn luyện và tập dữ liệu đã được token hóa')

    # 8. Training
    logger.info(f'{os.path.basename(__file__)}: Bắt đầu quá trình huấn luyện mô hình')
    train_result = trainer.train()
    logger.info(f'{os.path.basename(__file__)}: Đã hoàn thành quá trình huấn luyện mô hình')
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_model(output_dir)
    logger.info(f'{os.path.basename(__file__)}: Đã lưu mô hình tại {output_dir}')
    
    # 9. Evaluation
    logger.info(f'{os.path.basename(__file__)}: Bắt đầu quá trình đánh giá mô hình')
    eval_result = trainer.evaluate()
    logger.info(f'{os.path.basename(__file__)}: Đã hoàn thành quá trình đánh giá mô hình\nEvaluation metrics: {eval_result}')

    return output_dir