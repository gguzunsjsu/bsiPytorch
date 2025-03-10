import time
import torch
import collections
import collections.abc
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

paths = [str(x) for x in Path(".").glob("**/*.txt")]
'''
We choose to train a byte-level Byte-pair encoding tokenizer (the same as GPT-2), with the same special tokens as RoBERTa. Let’s arbitrarily pick its size to be 52,000.
We recommend training a byte-level BPE (rather than let’s say, a WordPiece tokenizer like BERT) because it will start building its vocabulary from an alphabet of single bytes, so all words will be decomposable into tokens (no more `<unk>` tokens!).
'''
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",

])

tokenizer.save_model("EsperBERTo")
tokenizer = ByteLevelBPETokenizer(
    "./EsperBERTo/vocab.json",
    "./EsperBERTo/merges.txt",
)
#We now have both a `vocab.json`, which is a list of the most frequent tokens ranked by frequency, and a `merges.txt` list of merges.

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print('Torch availability: ',torch.cuda.is_available())

#As the model is BERT-like, we’ll train it on a task of *Masked language modeling*, i.e. the predict how to fill arbitrary tokens that we randomly mask in the dataset. 

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)

#As we are training from scratch, we only initialize from a config, not from an existing pretrained model or checkpoint.
model = RobertaForMaskedLM(config=config) 

print('number of parameters: ', model.num_parameters())
# => 84 million parameters

#We'll build our dataset by applying our tokenizer to our text file.
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo.txt",
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

#start training
start = time.time()
trainer.train()
end = time.time()
print('time to train model: ', end-start)

trainer.save_model("./EsperBERTo")

#checking that our LM trained
fill_mask = pipeline(
    "fill-mask",
    model="./EsperBERTo",
    tokenizer="./EsperBERTo"
)

print(fill_mask("La suno <mask>."))
# The sun <mask>.
# =>

print(fill_mask("Jen la komenco de bela <mask>."))
# This is the beginning of a beautiful <mask>.
# =>