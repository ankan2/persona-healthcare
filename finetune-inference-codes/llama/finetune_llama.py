# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Features of this fine-tuning code:
1. Modified alpaca style prompt. 
2. Based on the values of truncate_doc_to and max_seq_len, filters the dataset to contain only those examples that can fit within max_seq_len after truncating the document to truncate_doc_to number of words
3. Train on completion only
4. Preprocessing: After truncating the document to truncate_doc_to words, if the document does not contain any one of the relevant sentences (as labelled in the dataset), we remove that example
'''

from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import numpy as np
import evaluate
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import pandas as pd
import huggingface_hub

# huggingface_hub.login(token='<hugging_face_tokens>')
# nltk.download('punkt')
# nltk.download('wordnet')

tqdm.pandas()

def compute_meteor(predictions, references, alpha=0.9, beta=3, gamma=0.5):
  scores = [
	  meteor_score.single_meteor_score(
		  word_tokenize(ref), word_tokenize(pred), alpha=alpha, beta=beta, gamma=gamma
	  )
	  for ref, pred in zip(references, predictions)
  ]
  return {"meteor": scores}


# Define and parse arguments.
@dataclass
class ScriptArguments:
	"""
	The name of the Casual LM model we wish to fine with SFTTrainer
	"""

	model_name: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
	dataset_name: Optional[str] = field(
		default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
	)
	dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
	log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
	learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
	batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
	seq_length: Optional[int] = field(default=1500, metadata={"help": "Input sequence length"})
	gradient_accumulation_steps: Optional[int] = field(
		default=16, metadata={"help": "the number of gradient accumulation steps"}
	)
	load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
	load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
	use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
	trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
	output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
	peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
	peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
	logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
	use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
	num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
	max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
	save_steps: Optional[int] = field(
		default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
	)
	save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
	push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
	hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})

	# adding new arguments
	checkpoint_after_num_epochs: Optional[int] = field(default=None, metadata={"help": "the number of epochs after which to checkpoint"})
	truncate_doc_to: Optional[int] = field(default=1200, metadata={"help": "the number of words to truncate the document to"})
	target_domain: Optional[str] = field(default="all", metadata={"help": "target domain"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
	raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
	quantization_config = BitsAndBytesConfig(
		load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
	)
	# Copy the model to each device
	# device_map = {"": Accelerator().local_process_index}
	device_map = "auto"
	torch_dtype = torch.bfloat16
else:
	device_map = None
	quantization_config = None
	torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
	script_args.model_name,
	quantization_config=quantization_config,
	device_map=device_map,
	trust_remote_code=script_args.trust_remote_code,
	torch_dtype=torch_dtype,
	use_auth_token=script_args.use_auth_token,
)

ALPACA_PROMPT_FORMAT = (
	"You are a helpful, respectful and honest assistant. Below is an instruction that describes a task, paired with a document that provides further context. "
	"Write a response in complete sentences that appropriately answers the request.\n\n"
	"### Instruction:\n{instruction}\n\n### Document:\n{document}\n\n### Response:\n"
)

ALPACA_TEXT_FORMAT = (
	"You are a helpful, respectful and honest assistant. Below is an instruction that describes a task, paired with a document that provides further context. "
	"Write a response in complete sentences that appropriately answers the request.\n\n"
	"### Instruction:\n{instruction}\n\n### Document:\n{document}\n\n### Response:\n{summary}</s></s></s>"
)

d_domain_aspects = {"History" : "Education",
                    "Career" : "Life and Career",
                    "Background" : "Life and Career",
                    "Geography" : "Education",
                    "Life and career" : "Life and Career",
                    "Production" : "Music",
                    "Education" : "Education",
                    "Composition" : "Music",
                    "Soundtrack" : "Music" 
                }

# Step 2: Load the dataset
def preprocess_examples(examples, tokenizer, truncate_doc_to, threshold_max_tokens, target_domain):
	# encode the code-docstring pairs
	documents = examples['document']
	summaries = examples['summary']

	dataset = "webmd"
	try:
		persona = examples["perspective"]
		dataset = "webmd"
		INSTRUCTION_FORMAT = (
			"Summarize the medical document given below from the perspective of a {persona}"
		)
	except:
		print("Wrong data")

	texts = []
	prompts = []
	validity = []

	for idx in range(len(documents)):

		# truncate the document till last fullstop before truncate_doc_to
		documents[idx] = " ".join(documents[idx].split(" ")[:truncate_doc_to])
		documents[idx] = ".".join(documents[idx].split(".")[:-1]) + "."

		if dataset == "webmd":
			instruction = INSTRUCTION_FORMAT.format(persona=persona[idx])

		_text = ALPACA_TEXT_FORMAT.format(instruction=instruction, document=documents[idx],summary=summaries[idx])
		_prompt = ALPACA_PROMPT_FORMAT.format(instruction=instruction,document=documents[idx])

		texts.append(_text)
		prompts.append(_prompt)

		num_tokens = len(tokenizer(_text)["input_ids"])

		if num_tokens <= threshold_max_tokens:
			validity.append(1)
		else:
			validity.append(0)

	model_inputs = {}
	model_inputs["text"] = texts
	model_inputs["prompt"] = prompts
	model_inputs["validity"] = validity
	
	return model_inputs

def get_filter_dict(dataset):
	filter_dict = {}
	for split in dataset.keys():
		filter_dict[split] = []
		for idx in range(len(dataset[split])):
			if dataset[split][idx]["validity"] == 1:
				filter_dict[split].append(idx)
	return filter_dict

def formatting_prompts_func(example):
	output_texts = []
	for i in range(len(example['text'])):
		text = example['text'][i]
		output_texts.append(text)
	return output_texts


############ To finetune with different dataset, change filename here ############

dataset = load_dataset("json", data_files={"train": "<train jsonl file path>"})  #trian jsonl path


# dataset["train"] = dataset["train"].select(np.arange(30))
# dataset["test"] = dataset["test"].select(np.arange(4))
# dataset["validation"] = dataset["validation"].select(np.arange(4))

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
print("llama tokenizer pad token id: ", tokenizer.pad_token_id)
print("llama model pad token id: ", model.config.pad_token_id)

dataset = dataset.map(preprocess_examples,batched=True,fn_kwargs={"tokenizer":tokenizer, "truncate_doc_to":script_args.truncate_doc_to, "threshold_max_tokens":script_args.seq_length, "target_domain":script_args.target_domain})

response_template = "\n\n### Response:\n"
response_template_ids = [2277, 29937, 13291, 29901, 13] # contextual encoding of response template

collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

filter_dict = get_filter_dict(dataset)
for split in dataset.keys():
	dataset[split] = dataset[split].select(filter_dict[split])

# dataset.set_format(type="torch", columns=["text", "prompt"])

print("DATASET SUMMARY AFTER PROCESSING:\n",dataset)

print("TEXT SAMPLE:\n",dataset["train"][0]["text"])


###################### SAVING CHECKPOINTS ############################

# making checkpoint_after_num_epochs aggresively override all arguments related to saving checkpoints

# compute the number of epoch intervals between two checkpoints are saved, if checkpoint_after_num_epochs is specified that overrides the save_steps argument
if script_args.checkpoint_after_num_epochs is not None:
	script_args.save_steps = ((len(dataset['train']) // script_args.batch_size) // script_args.gradient_accumulation_steps) * script_args.checkpoint_after_num_epochs

	# if save_total_limits restricts from saving checkpoints at all epochs possible as specified by checkpoint_after_num_epochs, then save_total_limit is set to the number of epochs possible, i.e checkpoint_after_num_epochs overrides save_total_limit
	if (script_args.num_train_epochs // script_args.checkpoint_after_num_epochs) > script_args.save_total_limit:
		script_args.save_total_limit = (script_args.num_train_epochs // script_args.checkpoint_after_num_epochs) + 2

######################################################################

# Step 3: Define the training arguments
training_args = TrainingArguments(
	output_dir=script_args.output_dir,
	per_device_train_batch_size=script_args.batch_size,
	gradient_accumulation_steps=script_args.gradient_accumulation_steps,
	learning_rate=script_args.learning_rate,
	logging_steps=script_args.logging_steps,
	num_train_epochs=script_args.num_train_epochs,
	max_steps=script_args.max_steps,
	report_to=script_args.log_with,
	save_steps=script_args.save_steps,
	# save_total_limit=script_args.save_total_limit,
	push_to_hub=script_args.push_to_hub,
	hub_model_id=script_args.hub_model_id,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
	peft_config = LoraConfig(
		r=script_args.peft_lora_r,
		lora_alpha=script_args.peft_lora_alpha,
		bias="none",
		task_type="CAUSAL_LM",
	)
else:
	peft_config = None

# Step 5: Define the Trainer
trainer = SFTTrainer(
	model=model,
	args=training_args,
	max_seq_length=script_args.seq_length,
	train_dataset=dataset["train"],
	peft_config=peft_config,
	formatting_func=formatting_prompts_func,
    data_collator=collator,
	packing = False
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)

print("model saved and program ended")	

