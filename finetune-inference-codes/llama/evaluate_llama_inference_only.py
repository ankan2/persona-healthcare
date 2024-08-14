from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import transformers
import torch
import evaluate
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import pandas as pd
import nltk
from nltk.translate import meteor_score
import huggingface_hub
from argparse import ArgumentParser
from tqdm import tqdm
from peft import PeftModel, PeftConfig

'''
Features of the evaluation code:
1. To be used for evaluating a pre-trained model
2. Uses modified alpaca prompt
'''

huggingface_hub.login(token='<hugging face tokens>')  # hugging face tokens goes here
nltk.download('punkt')
nltk.download('wordnet')

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



# Step 2: Load the dataset
def preprocess_examples(examples, tokenizer, truncate_doc_to, target_domain):
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
		print("Wrong dataset")

	texts = []
	prompts = []
	validity = []
	instructions = []

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
		instructions.append(instruction)
		validity.append(1)

	model_inputs = {}
	model_inputs["text"] = texts
	model_inputs["prompt"] = prompts
	model_inputs["validity"] = validity
	model_inputs["instruction"] = instructions
	
	return model_inputs

def get_filter_dict(dataset):
	filter_dict = {}
	for split in dataset.keys():
		filter_dict[split] = []
		for idx in range(len(dataset[split])):
			if dataset[split][idx]["validity"] == 1:
				filter_dict[split].append(idx)
	return filter_dict

def compute_meteor(predictions, references, alpha=0.9, beta=3, gamma=0.5):
  scores = [
	  meteor_score.single_meteor_score(
		  word_tokenize(ref), word_tokenize(pred), alpha=alpha, beta=beta, gamma=gamma
	  )
	  for ref, pred in zip(references, predictions)
  ]
  return {"meteor": scores}

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument("--truncate_doc_to", type=int, default=1200, help="the number of words to truncate the document to")
	parser.add_argument("--model_dir", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="The checkpoint directory to use for inference")
	parser.add_argument("--output_dir", type=str, default="eval_output/", help="The output directory to store the results")
	parser.add_argument("--max_new_tokens", type=int, default=100, help="max summary token len")
	parser.add_argument("--target_domain", type=str, default="all", help="target domain")

	args = parser.parse_args()


	model_name = args.model_dir
	truncate_doc_to = args.truncate_doc_to

	if "7b" in model_name:
		model_param = "7b"
	else:
		model_param = "13b"

	if "checkpoint" not in model_name:
		checkpoint_name = "last"
	else:
		checkpoint_name = model_name.split("/")[-1]

	tokenizer = AutoTokenizer.from_pretrained(model_name)

	pipeline = transformers.pipeline(
					"text-generation",
					model=args.model_dir,
					torch_dtype=torch.float16,
					device_map="auto",
					tokenizer=tokenizer
				)

	
	############ To finetune with different dataset, change filename here ############

	dataset = load_dataset("json", data_files={"test": "<test jsonl file path>"}) #test jsonl file path goes here

	# dataset["test"] = dataset["test"].select(np.arange(4))

	tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
	dataset = dataset.map(preprocess_examples,batched=True,fn_kwargs={"tokenizer":tokenizer, "truncate_doc_to":args.truncate_doc_to, "target_domain":args.target_domain})

	filter_dict = get_filter_dict(dataset)
	for split in dataset.keys():
		dataset[split] = dataset[split].select(filter_dict[split])

	# dataset.set_format(type="torch", columns=["text", "prompt"])

	print("DATASET SUMMARY AFTER PROCESSING:\n",dataset)

	print("PROMPT SAMPLE:\n",dataset["test"][0]["prompt"])

	input_texts = [data['prompt'] for data in dataset["test"]]
	model_generated_summary = []
	label_summary = [data['summary'].replace("\n","") for data in dataset["test"]]

	store_intermediate_results = False
	intermediate_file_path = args.output_dir + "intermediate_results-" + checkpoint_name + ".csv"

	if store_intermediate_results:
		with open(intermediate_file_path, "w") as f:
			pass

	for i in tqdm(range(len(input_texts))):
		output = pipeline(
					input_texts[i],
					do_sample=True,
					top_k=10,
					# temperature = 0.01,
					# num_beams = 2,
					num_return_sequences=1,
					eos_token_id=tokenizer.eos_token_id,
					max_new_tokens=args.max_new_tokens, # 350 #256 #100 #40,
					# forced_eos_token_id=tokenizer.eos_token_id,
					# repetition_penalty=1.15,
				)
		
		output[0]['generated_text'] = output[0]['generated_text'].removeprefix(input_texts[i])
		output[0]['generated_text'] = output[0]['generated_text'].replace("\n", "")
		model_generated_summary.append(output)

		if store_intermediate_results:
			with open(intermediate_file_path, "a") as f:
				if output[0]['generated_text'] == "":
					print(f"Empty string generated in sample no. {i}")
				
				f.write(f"\'\'\'{output[0]['generated_text']}\'\'\':::\'\'\'{label_summary[i]}\'\'\'\n")

	# print(model_generated_summary)
	
	model_generated_summary = [model_generated_summary[i][0]['generated_text'] for i in range(len(model_generated_summary))]

	num_complete_sentences = 0
	for i in range(len(model_generated_summary)):
		if len(model_generated_summary[i].strip()) == 0 or model_generated_summary[i].strip()[-1] in ['.','!','?']:
			num_complete_sentences += 1
	frac_complete_sentences = num_complete_sentences / len(model_generated_summary)


	df_data = {
	"Instruction": [data['instruction'] for data in dataset["test"]],
	"Document": [data["document"] for data in dataset["test"]],
	"Model Generated Summary": model_generated_summary,
	"Label Summary": label_summary,
	}

	rouge = evaluate.load('rouge')
	bertscore = evaluate.load('bertscore')
	bleu = evaluate.load("bleu")

	results_rouge = rouge.compute(predictions=model_generated_summary, references=label_summary, use_aggregator=False)
	results_meteor = compute_meteor(predictions=model_generated_summary, references=label_summary)
	# results_bleu = bleu.compute(predictions=model_generated_summary, references=label_summary)
	results_bleu = []
	for i in range(len(model_generated_summary)):
		try:
			results_bleu.append(bleu.compute(predictions=[model_generated_summary[i]], references=[label_summary[i]])['bleu'])
		except:
			print(f"bleu computation throwing error in sample no. {i}")
			results_bleu.append(0)
	results_bleu = {'bleu': results_bleu}
	results_bertscore = bertscore.compute(predictions=model_generated_summary, references=label_summary, model_type="distilbert-base-uncased")

	eval_results = {
		"frac_complete_sentences": frac_complete_sentences
	}

	for metric in results_rouge.keys():
		df_data[metric] = results_rouge[metric]
		eval_results[metric] = np.mean(results_rouge[metric])

	for metric in results_meteor.keys():
		df_data[metric] = results_meteor[metric]
		eval_results[metric] = np.mean(results_meteor[metric])

	for metric in results_bleu.keys():
		df_data[metric] = results_bleu[metric]
		eval_results[metric] = np.mean(results_bleu[metric])

	for metric in results_bertscore.keys():
		if metric != 'hashcode':
			df_data[metric] = results_bertscore[metric]
			eval_results[metric] = np.mean(results_bertscore[metric])

	print(eval_results)

	# print("Debugging")
	# print(df_data)
	df = pd.DataFrame(df_data)
	df.to_csv(args.output_dir + f'LLAMA-{model_param}-webmd-test-inference-no-finetune-' + checkpoint_name + f'-{args.max_new_tokens}.csv')

