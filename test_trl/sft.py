# imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
from datasets import load_dataset

# load model and dataset - dataset needs to be in a specific format
model = AutoModelForCausalLM.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/Qwen2-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/Qwen2-7B-Instruct")

dataset = load_dataset("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/jiaokechen/openai_humaneval", split="test")

# load trainer
trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    model_init_kwargs=None
)

# train
trainer.train()