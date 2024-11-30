from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from datasets import Dataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Your Script Description")

#     parser.add_argument("folder_name", type=str, help="Name of the folder")
parser.add_argument("--model-path", type=str, default = "meta-llama/Meta-Llama-Guard-2-8B",help="Model path")
parser.add_argument("--data-path", type=str, help="Where to load data")
parser.add_argument("--out-path", type=str, help="Path to the output file")
parser.add_argument("--token", type=str, default="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", help="Huggingface token")
parser.add_argument("--text-column", type=str, default="text", help="Field that contained text in the dataset")
parser.add_argument("--out-column", type=str, default="llama_guard2_label", help="New field that will contain the labels")


args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_path = args.model_path
data_path = args.data_path#"/data/sooksatrak/HT/data/htrp/train.json"
out_path = args.out_path#"/data/sooksatrak/HT/data/"
# model_path = "meta-llama/Llama-2-7b-hf"
access_token = args.token
tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)

# if getattr(tokenizer, "pad_token_id") is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    token=access_token,
    device_map=device
)


### Import data
data = Dataset.from_json(data_path)
df = data.to_pandas()
pred = []


### Prediction
# batch_size = 1
# print(tokenizer.get_added_vocab())
for i in tqdm(range(0, len(df))):
    text = df[args.text_column][i]
    # print(tokenizer(text, max_length=512)["input_ids"])
    # print(Dataset.from_pandas(df))
    text = tokenizer.decode(tokenizer(text, max_length=512)["input_ids"], skip_special_tokens=True)
    input_ids = tokenizer.apply_chat_template([
        {"role":"user", "content":"Give me an example of an online advertisement."},
        {"role":"assistant", "content": text}
        ], return_tensors="pt").to(device)
    prompt_len = input_ids.shape[-1]
    outputs = model.generate(input_ids = input_ids,max_new_tokens=100, pad_token_id=0)
    # print(outputs[0][prompt_len:])
    out_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    out_lst = [txt.strip() for txt in out_text.split("\n")]
    if len(out_lst) == 2 and out_lst[0] == "unsafe":
        pred.append(out_lst[1])
    elif len(out_lst) == 1 and out_lst[0] == "safe":
        pred.append("safe")
    else:
        pred.append("undefined")
    # print(pred)
    # print(out_lst)
df[args.out_column] = pred
new_data = Dataset.from_pandas(df)
new_data.to_json(out_path)
print("Done!!!")
# print(df)
# test_data = Dataset.from_json(out_path)
# print(test_data["llama_guard2_label"])
# print(test_data)




# # data = [[
# #     {"role":"user", "content":"Give me some popcorn."},
# #     {"role":"assistant", "content":"Write me a poem about Machine Learning."}
# #     ],
# #     [
# #     {"role":"user", "content":"Give me some popcorn."},
# #     {"role":"assistant", "content":"Hi, \nI am a simple girl who dreams about a strong man beside me ❗❗❗\nI don\’t need to talk about the weather and your past relationships. \nALL I WANT FROM YOU IS WILD SEX BEFORE I GET EXHAUSTED. \nYou can hook up with me with ease just call or PM me. \n✨✨ [PHONE]✨✨ \nYou and I together can do everything in as follows:  \nCOB - Come On Body, Deep throat, Domination, Face sitting, Fingering, Foot fetish, French kissing, GFE, Lap dancing, Nuru body to body massage, Oral sex - blowjob, OWO - Oral without condom,  \n\n☎️ [PHONE]🥰🥰🥰"}
# #     ]]

# # input_ids1 = tokenizer.apply_chat_template([
# #     {"role":"user", "content":"Give me some popcorn."},
# #     {"role":"assistant", "content":"Write me a poem about Machine Learning."}
# #     ], return_tensors="pt").to(device)
# # input_ids2 = tokenizer.apply_chat_template([
# #     {"role":"user", "content":"Give me some popcorn."},
# #     {"role":"assistant", "content":"Hi, \nI am a simple girl who dreams about a strong man beside me ❗❗❗\nI don\’t need to talk about the weather and your past relationships. \nALL I WANT FROM YOU IS WILD SEX BEFORE I GET EXHAUSTED. \nYou can hook up with me with ease just call or PM me. \n✨✨ [PHONE]✨✨ \nYou and I together can do everything in as follows:  \nCOB - Come On Body, Deep throat, Domination, Face sitting, Fingering, Foot fetish, French kissing, GFE, Lap dancing, Nuru body to body massage, Oral sex - blowjob, OWO - Oral without condom,  \n\n☎️ [PHONE]🥰🥰🥰"}
# #     ], return_tensors="pt").to(device)
# # input_ids = [input_ids1, input_ids2]
# input_ids = tokenizer.apply_chat_template(data, return_tensors="pt", padding = True).to(device)
# print(input_ids.shape)
# # input_text1 = "Write me a poem about Machine Learning."
# # input_text2 = "Hi, \nI am a simple girl who dreams about a strong man beside me ❗❗❗\nI don\’t need to talk about the weather and your past relationships. \nALL I WANT FROM YOU IS WILD SEX BEFORE I GET EXHAUSTED. \nYou can hook up with me with ease just call or PM me. \n✨✨ [PHONE]✨✨ \nYou and I together can do everything in as follows:  \nCOB - Come On Body, Deep throat, Domination, Face sitting, Fingering, Foot fetish, French kissing, GFE, Lap dancing, Nuru body to body massage, Oral sex - blowjob, OWO - Oral without condom,  \n\n☎️ [PHONE]🥰🥰🥰"
# # input_ids1 = tokenizer(input_text1, return_tensors="pt").to(device)
# # input_ids2 = tokenizer(input_text2, return_tensors="pt").to(device)
# # mode = model.to(device)
# # outputs1 = model.generate(**input_ids1,max_new_tokens=100, pad_token_id=0)
# # outputs2 = model.generate(**input_ids2,max_new_tokens=100, pad_token_id=0)
# # prompt1_len = input_ids1.input_ids.shape[-1]
# # prompt2_len = input_ids2.input_ids.shape[-1]
# outputs = model.generate(input_ids = input_ids,max_new_tokens=100, pad_token_id=0)
# # outputs1 = model.generate(input_ids = input_ids1,max_new_tokens=100, pad_token_id=0)
# # outputs2 = model.generate(input_ids = input_ids2,max_new_tokens=100, pad_token_id=0)
# prompt1_len = input_ids[0].shape[-1]
# prompt2_len = input_ids[1].shape[-1]
# print("#########################")
# print("Input 1:")
# print("#########################")
# print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
# print("#########################")
# print("Output:")
# print("#########################")
# print(tokenizer.decode(outputs[0][prompt1_len:], skip_special_tokens=True)) # switch here according to tokenizer.special_tokens
# print("#########################")
# print("Input 2:")
# print("#########################")
# print(tokenizer.decode(input_ids[1], skip_special_tokens=True))
# print("#########################")
# print("Output:")
# print("#########################")
# print(tokenizer.decode(outputs[1][prompt2_len:], skip_special_tokens=True)) # switch here according to tokenizer.special_tokens
