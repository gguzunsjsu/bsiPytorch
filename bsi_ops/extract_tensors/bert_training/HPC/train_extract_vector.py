import time
import torch
import collections
import collections.abc
import numpy as np
import pickle
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
import os

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

save_dir = "EsperBERTo_3"
os.makedirs(save_dir, exist_ok=True)
tokenizer.save_model(save_dir)
# tokenizer.save_model("EsperBERTo_3")
tokenizer = ByteLevelBPETokenizer(
    "./EsperBERTo_3/vocab.json",
    "./EsperBERTo_3/merges.txt",
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


tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo_3", max_len=512)

#As we are training from scratch, we only initialize from a config, not from an existing pretrained model or checkpoint.
model = RobertaForMaskedLM(config=config, output_hidden_states=True) 

print('number of parameters: ', model.num_parameters())
# => 84 million parameters

# printing model
print('model architecture: \n',model)

#We'll build our dataset by applying our tokenizer to our text file.
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./oscar.eo.txt",
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Modify the trainer to save intermediate vectors
class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intermediate_vectors = []

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        loss = outputs.loss

        print('loss: \n', loss)
        print('outputs: \n', outputs)
        # Store intermediate vectors (last hidden states)
        # intermediate_vectors = outputs.last_hidden_state
        # intermediate_vectors = outputs.hidden_states[-1]
        if 'hidden_states' in outputs:
            intermediate_vectors = outputs.hidden_states[-1]
            self.intermediate_vectors.append(intermediate_vectors.detach().cpu().numpy())
        
        # Check if loss is not None
        if loss is not None:
            return loss
        else:
            # Return a dummy loss if necessary
            return torch.tensor(0.0, requires_grad=True)
            # return loss

    def save_intermediate_vectors(self, output_dir):
        intermediate_vectors = torch.tensor(np.concatenate(self.intermediate_vectors, axis=0))
        torch.save(intermediate_vectors, output_dir / "intermediate_vectors.pt")
        with open(output_dir / "intermediate_vectors.pkl", 'wb') as f:
            pickle.dump(intermediate_vectors, f)


training_args = TrainingArguments(
    output_dir="./EsperBERTo_3",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = MyTrainer(
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

trainer.save_model("./EsperBERTo_3")

# Save intermediate vectors
trainer.save_intermediate_vectors(Path("./EsperBERTo_3"))

# #checking that our LM trained
# fill_mask = pipeline(
#     "fill-mask",
#     model="./EsperBERTo",
#     tokenizer="./EsperBERTo"
# )

# print(fill_mask("La suno <mask>."))
# # The sun <mask>.
# # =>

# print(fill_mask("Jen la komenco de bela <mask>."))
# # This is the beginning of a beautiful <mask>.
# # =>