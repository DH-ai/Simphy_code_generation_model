from transformers import AutoTokenizer, T5ForConditionalGeneration,pipeline,AutoModelForSeq2SeqLM
import torch


model_name = "Salesforce/codet5p-6b"
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
# model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True

).to(device)

print(model.get_memory_footprint())


text = "Create a python script to print 10 whole numbers"
# input_ids = tokenizer(text, return_tensors="pt").input_ids
encoding = tokenizer(text,return_tensors="pt").to(device=device)
print(type(encoding))
encoding['decoder_input_ids']= encoding['input_ids'].clone()
# simply generate a single sequence
# generated_ids = model.generate(input_ids, max_length=8)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
outputs= model.genrate(**encoding,max_length=720)

print(tokenizer.decode(outputs[0],skip_special_tokens=True))
# classifier = pipeline("text-generation",model=model,tokenizer=tokenizer)


# print(classifier(text))