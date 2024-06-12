from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, set_peft_model_state_dict, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from safetensors.torch import load_file
from transformers import BitsAndBytesConfig
from tqdm import tqdm
import evaluate


# global configuration for this project
CACHE_DIR = "./cache"
DATASET_NAME = "samsum"
MODEL_NAME = "philschmid/flan-t5-xxl-sharded-fp16" # google/flan-t5-xl
EXP_NAME = f"{MODEL_NAME.split('/')[1]}-{DATASET_NAME}"
CKPT_DIR = f"./ckpt/{EXP_NAME}"
LOG_DIR =  f"./log/{EXP_NAME}"
LOAD_MODEL_IN_8BIT = True
INFERENCE_MODE = True


# load dataset from hub
dataset = load_dataset(path=DATASET_NAME, cache_dir=f"{CACHE_DIR}/dataset", trust_remote_code=True)
print(dataset)
'''
DatasetDict({
    train: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 14732
    })
    test: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 819
    })
    validation: Dataset({
        features: ['id', 'dialogue', 'summary'],
        num_rows: 818
    })
})
'''


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=f"{CACHE_DIR}/tokenizer")

# basic information of tokenizer
print(f"model_max_length: {tokenizer.model_max_length}") # 512
print(f"bos_token_id: {tokenizer.bos_token_id}")
print(f"unk_token_id: {tokenizer.unk_token_id}")
print(f"eos_token_id: {tokenizer.eos_token_id}")
print(f"pad_token_id: {tokenizer.pad_token_id}")

# tokenize example sentence
example_sentence = "Hello, how are you?"
print(f"example sentence: {example_sentence}")
print(f"tokenized result: {tokenizer(example_sentence)}")


# preprocess dataset: determine maximum source sequence size
def tokenize_source_func(example):
    """
    if batched:
        example will be:
        {
            "key1": str,
            "key2": str
        }
    else:
        example will be:
        {
            "key1": [str, str, str],
            "key2": [str, str, str]
        }
    """
    return tokenizer(example["dialogue"], truncation=True)

# concatenate train dataset and test dataset, and tokenize "dialogue" of each sample
tokenized_sources = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    tokenize_source_func,
    batched=True,
    batch_size=1000,
    remove_columns=["dialogue", "summary"], # To save memory use, we drop these two features because we do not need these two features in processed dataset (tokenized_inputs)
    cache_file_name=f"{CACHE_DIR}/dataset/tokenized_sources"
)

# get input sequence size of each sample
source_lengths = [len(x) for x in tokenized_sources["input_ids"]]

# to avoid some extreme long input sequences dominate the maximum sequence length, we use percentile to determine maximun sequence size
max_source_length = int(np.percentile(source_lengths, 85))
print(f"max_source_length: {max_source_length}")


# preprocess dataset: determine maximum target sequence size
def tokenize_target_func(example):
    """
    if batched:
        example will be:
        {
            "key1": str,
            "key2": str
        }
    else:
        example will be:
        {
            "key1": [str, str, str],
            "key2": [str, str, str]
        }
    """
    return tokenizer(example["summary"], truncation=True)

# concatenate train dataset and test dataset, and tokenize "dialogue" of each sample
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    tokenize_target_func,
    batched=True,
    batch_size=1000,
    remove_columns=["dialogue", "summary"], # To save memory use, we drop these two features because we do not need these two features in processed dataset (tokenized_targets)
    cache_file_name=f"{CACHE_DIR}/dataset/tokenized_targets"
)

# get target sequence size of each sample
target_lengths = [len(x) for x in tokenized_targets["input_ids"]]

# to avoid some extreme long target sequences dominate the maximum sequence length, we use percentile to determine maximun sequence size
max_target_length = int(np.percentile(target_lengths, 90))
print(f"max_target_length: {max_target_length}")


# preprocess dataset: prepare the dataset for model training and evaluation
# that is, the dataset should have three features: "input_ids", "attention_mask", and "labels"
def preprocess_dataset_func(example):
    # before tokenize source sentence, we add task prefix which is used in t5 training
    sources = ["summarize: " + sent for sent in example["dialogue"]]

    # tokenize source sentences to get sources sequences with same length
    inputs = tokenizer(sources, max_length=max_source_length, truncation=True)

    # tokenize target sentences to get target sequences with same length
    labels = tokenizer(example["summary"], max_length=max_target_length, truncation=True)
    
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels["input_ids"]}

# each split (train, test, validation) in dataset will be preprocessed in the same way 
tokenized_dataset = dataset.map(
    preprocess_dataset_func,
    batched=True,
    remove_columns=["dialogue", "summary", "id"], # to save memory, we do not need these original features in preprocessed dataset (tokenized_dataset)
    cache_file_names={
        "train": f"{CACHE_DIR}/dataset/tokenized_dataset_train",
        "test": f"{CACHE_DIR}/dataset/tokenized_dataset_test",
        "validation": f"{CACHE_DIR}/dataset/tokenized_dataset_validation"
    }
)
print(tokenized_dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask', 'labels'],
        num_rows: 14732
    })
    test: Dataset({
        features: ['input_ids', 'attention_mask', 'labels'],
        num_rows: 819
    })
    validation: Dataset({
        features: ['input_ids', 'attention_mask', 'labels'],
        num_rows: 818
    })
})
"""


# we are ready to train the model! now, we load the model
if not LOAD_MODEL_IN_8BIT:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,cache_dir=f"{CACHE_DIR}/model")
else:
    config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        cache_dir=f"{CACHE_DIR}/model",
        quantization_config=config
    )
    # if you use load_in_8bit, you may got the error of not finding "libcusparse.so.11"
    # try to install cusparse (pip install nvidia-pyindex, pip install nvidia-cusparse)
    # and you will find the file under /home/johnnyhwu/ssd2/lora-finetuning/env/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.11
    # export this path: export LD_LIBRARY_PATH="/home/johnnyhwu/ssd2/lora-finetuning/env/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

    # prepare int-8 model for training
    if not INFERENCE_MODE:
        model = prepare_model_for_kbit_training(model)


# define lora config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "k", "v", "o"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# add lora module to original model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
"""
trainable params: 18,874,368 || all params: 2,868,631,552 || trainable%: 0.6580
"""


if not INFERENCE_MODE: 

    # create a data collator which will take care of padding of inputs and labels
    # https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )


    # deinfe training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4, # higher learning rate
        num_train_epochs=10,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        logging_dir=LOG_DIR,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        dataloader_num_workers=8,
        gradient_checkpointing=False,
        report_to="tensorboard"
    )

    # create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )

    # silence the warnings. please re-enable for inference!
    model.config.use_cache = False

    # train model
    trainer.train()


if INFERENCE_MODE:

    # after training, we want to evaluate model on testing dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=f"{CACHE_DIR}/tokenizer")
    model.config.use_cache = True

    # then, we load adapter weight
    adapters_weights = load_file(f"ckpt/flan-t5-xxl-sharded-fp16-samsum/checkpoint-18410/adapter_model.safetensors")

    # finally, we set original model with new weight
    set_peft_model_state_dict(model, adapters_weights)

    # we set model to inference mode, "do not" move model to cuda (.cuda()), it will done automatically
    model.eval()

    # remember that we have a dataset (DatasetDict) loaded from huggingface
    print(dataset)
    """
    DatasetDict({
        train: Dataset({
            features: ['id', 'dialogue', 'summary'],
            num_rows: 14732
        })
        test: Dataset({
            features: ['id', 'dialogue', 'summary'],
            num_rows: 819
        })
        validation: Dataset({
            features: ['id', 'dialogue', 'summary'],
            num_rows: 818
        })
    })
    """

    # we randomly choose a testing sample
    sample = dataset["test"][42]
    print(sample)
    """
    {
        'id': '13829773',
        'dialogue': 
            "Ola: Hello Kate, sorry for not keeping in touch properly. As expected, we have hardly any connectivity here in Cuba. But we're doing fine and enjoying our trip. How are the things at home?\r\n
            Kate: At long last! Started to worry. Nothing new happening, if you disregard all that Xmas craze. Momo has recovered from her injury and frolicking again.\r\n
            Kate: <file_photo>\r\n
            Kate: Good old Momo! Yes, it is your scarf!\r\n
            Ola: NO!!! It's one of my favorites! The one from Laos!\r\n
            Kate: Too late. Momo thinks it belongs to her now. Get yourself a new one. They surely have nice ones there.\r\n
            Ola: Not at all. Only cheapish cotton blouses with horrible multi-coloured embroidery or some equally horrible crochetted tops. No shawls or scarfs.\r\n
            Ola: <file_photo>\r\n
            Kate: Wait a sec!\r\n
            Kate: <file_photo>\r\n
            Kate: Isn't it similar?! Mum would probably like it. Why don't you?\r\n
            Ola: Not a bad idea. But the quality is usually crappy.\r\n
            Kate: And if you go to some boutique shop or something? Not at a market as in your pics?\r\n
            Ola: I might try and find some. Would you like one too?\r\n
            Kate: Not really. And Mum would prefer to be the only one with an authentic Cuban blouse :))\r\n
            Ola: OK I'll have a look. Greets to everyone at home pls.\r\n
            Kate: Take care!",
        'summary': "Ola is in Cuba and is enjoying her trip. She has problems with connectivity there. Momo has recovered from her injury. Ola doesn't like the clothes in Cuba. Ola will try to find a blouse for mum in Cuba, as Kate suggested."
    }
    """

    # tokenize model's input (dialogue)
    # because the "inputs" argument in model.generate() expects tensor type, we should specify "return_tensors" here, or we will get the "input_ids" in a list
    input_ids = tokenizer(sample["dialogue"], truncation=True, return_tensors="pt")["input_ids"].cuda()

    # model inference
    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=128,
        do_sample=False,
        use_cache=True,
    )
    print(outputs)
    """
    tensor([[    0,  5424,     9,    19,    16, 13052,     5,  8822,    32,    65,
            16599,    45,   160,  2871,     5,  5424,     9,    31,     7, 25816,
                19,  2767,     5,  5424,     9,    56,   653,    12,   805,     3,
                9,   126,    80,     5,     1]])
    """
    print(tokenizer.decode(outputs[0]))
    """
    <pad> Ola is in Cuba. Momo has recovered from her injury. Ola's scarf is gone. Ola will try to buy a new one.</s>
    """


    # now, we want to evaluate model on testing dataset with rouge metric
    predictions, references = [], []
    for i in tqdm(range(len(dataset["test"])), desc="Evaluation"):
        sample = dataset["test"][i]
        input_ids = tokenizer(sample["dialogue"], truncation=True, return_tensors="pt")["input_ids"].cuda()
        outputs = model.generate(inputs=input_ids, max_new_tokens=128, do_sample=False, use_cache=True)
        prediction = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True)
        label = sample["summary"]
        predictions.append(prediction)
        references.append(label)

    # load rouge metric
    metric = evaluate.load("rouge")

    # compute metric
    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

    # print results
    print(f"rogue1: {rogue['rouge1']* 100:.2f}%")
    print(f"rouge2: {rogue['rouge2']* 100:.2f}%")
    print(f"rougeL: {rogue['rougeL']* 100:.2f}%")
    print(f"rougeLsum: {rogue['rougeLsum']* 100:.2f}%")