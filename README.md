# Persona-Healthcare

On The Persona-based Summarization of Domain-Specific Documents

    persona-codes/
    |
    |__ dataset/
    |   |__ sample_data/
    |   |   |__ ...                         -> contains two samples for reference 
    |   |        
    |   |__ full_data/
    |           |__ ...                     -> contains the train-val-test jsonl files
    |
    |__ finetune-inference-codes/           -> all code files
    |
    |
    |__ README.md
    |__ requirements.txt


Persona based Summary:-

    Description:-

    This is a sample README.md file for a project. The purpose of this document is to provide essential information about the project, including its description, installation instructions, usage guidelines, and contribution guidelines.

    Navigate to the project directory: cd your-project

    Install dependencies: pip install requirements.txt

    Sample commands for Llama2 :- 

        Finetune:-

        '''
        python3 finetune_llama.py \
            --model_name <hugging_face_model_dir> \    (example : meta-llama/Llama-2-7b-hf)
            --load_in_4bit \
            --use_peft \
            --batch_size 1 \
            --num_train_epochs 4 \
            --gradient_accumulation_steps 16 \
            --checkpoint_after_num_epochs 1 \
            --truncate_doc_to 800 \
            --seq_length 1600 \
            --learning_rate 0.00005 \
            --logging_step 10 \
            --target_domain all \
            --use_auth_token False \
            --output_dir <output directory path>
        '''

        Inference after Finetune:-
        '''
        python3 evaluate_llama_inference_after_FT.py \
            --model_dir <finetuned Llama2 model directory path> \
            --truncate_doc_to 800 \
            --output_dir <output directory path> \
            --max_new_tokens 100 \
            --target_domain all
        '''
        Only Inference:-

        '''
        python3 evaluate_llama_inference_only.py \
            --model_dir <hugging_face_model dir> \      (example : meta-llama/Llama-2-70b-hf)
            --truncate_doc_to 800 \
            --output_dir <output directory path> \
            --max_new_tokens 100 \
            --target_domain all
        '''
# Citation

    @article{mullick2024persona,
      title={On The Persona-based Summarization of Domain-Specific Documents},
      author={Mullick, Ankan and Bose, Sombit and Saha, Rounak and Bhowmick, Ayan Kumar and Goyal, Pawan and Ganguly, Niloy and Dey, Prasenjit and Kokku, Ravi},
      journal={arXiv preprint arXiv:2406.03986},
      year={2024}
    }
        
