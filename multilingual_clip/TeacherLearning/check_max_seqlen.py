#%%
# Estimate the max seqlen of the tokenizer by running the tokenizer over the dataset 
import transformers

tokenizerBase = 'aubmindlab/bert-large-arabertv2'

tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizerBase)

#%%

trainSamples = datasets.load_dataset('Arabic-Clip/Arabic_dataset_3M_translated_cleaned_v2_jsonl_format_ViT-B-16-plus-240', split='train[{}:]'.format(validationSize))

