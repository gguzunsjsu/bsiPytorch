import torch
from transformers import BertForSequenceClassification, BertTokenizer
from memory_profiler import memory_usage
import time

device = torch.device("cpu")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.eval()

inputs = tokenizer("Hello world!", return_tensors="pt").to(device)


def run_layer_by_layer(model, inputs):
    encoder_inputs = model.bert.embeddings(inputs["input_ids"])
    print(f"Encoder inputs {encoder_inputs}")

    for i, layer_module in enumerate(model.bert.encoder.layer[:11]):
        encoder_inputs = layer_module(
            encoder_inputs,
            attention_mask=inputs["attention_mask"].bool()
        )[0]

    encoder_inputs = model.bert.encoder.layer[11](
        encoder_inputs,
        attention_mask=inputs["attention_mask"].bool()
    )[0]

    if model.bert.pooler is not None:
        pooled_output = model.bert.pooler(encoder_inputs)
        logits = model.classifier(pooled_output)
        print(f"logits {logits}")
    else:
        logits = model.classifier(encoder_inputs[:, 0, :])


mem_before = memory_usage()[0]
run_layer_by_layer(model, inputs)
mem_after = memory_usage()[0]
print(f"Memory usage before: {mem_before} MiB")
print(f"Memory usage after : {mem_after} MiB")
print(f"Difference         : {mem_after - mem_before} MiB")

# import torch
# from transformers import BertForSequenceClassification
#
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#
# print("----Parameter count by encoder layer----")
#
# for i, layer_module in enumerate(model.bert.encoder.layer):
#     layer_param_count = sum(p.numel() for p in layer_module.parameters())
#     print(f"Layer {i} has {layer_param_count} parameters")
#
# # print((p.numel() for p in model.bert.pooler.parameters()))
#
# if model.bert.pooler is not None:
#     pooler_param_count = sum(p.numel() for p in model.bert.pooler.parameters())
#     print(f"Pooler -> Number of pooler parameters {pooler_param_count}")
# else:
#     print(f"No seperate pooler layer in this model")
#
# if hasattr(model, 'classifier'):
#     classifier_param_count = sum(p.numel() for p in model.classifier.parameters())
#     print(f"Classifier -> Number of parameters {classifier_param_count}")
#
