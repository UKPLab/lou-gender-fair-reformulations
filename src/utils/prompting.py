from random import random

from retry import retry
from tqdm import tqdm

@retry(Exception, tries=5, backoff=2, delay=1)
def run_icl_prompt(prompt, model_name, max_tokens=20, client=None):
    return client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=0,
        max_tokens=max_tokens,
    )

def get_icl_preds(instructions, token_mappings, german_instructions=True, model_name="gpt-3.5-turbo", client=None):

    unique_labels = list(set([ele["label"] for ele in instructions]))

    pred_tokens = []
    all_preds = []
    all_answers = []
    prediction_category = []


    for i, instruction in tqdm(enumerate(instructions)):
        label = instruction["label"]
        if "mistral" in model_name.lower() or "mixtral" in model_name.lower():
            first_instruction = {
                "role": "user",
                "content": f'{instruction["instructions"][0]["content"]}\n{instruction["instructions"][1]["content"]}'
            }

            instruction["instructions"] = [first_instruction] + instruction["instructions"][2:]

        response = run_icl_prompt(instruction["instructions"], model_name, 5, client)
        response_text = response.choices[0].message.content
        if "<|eot_id|>" in response_text:
            response_text = response_text.split("<|eot_id|>")[0]

        cleaned_response_text = response_text.replace("[", " ").replace("]", " ").replace("<|im_end|>", "").lower().strip().translate(str.maketrans('', '', string.punctuation)).replace("\n", " ").replace("imend", "")

        if german_instructions:
            cleaned_response_text = cleaned_response_text.replace("negative", "negativ").replace("positive", "positiv")

        response_tokens = cleaned_response_text.split(" ")

        occurred_label_tokens = [
            (token, label)
            for token, label in token_mappings.items()
            if token in response_tokens
        ]

        if len(occurred_label_tokens) == 1:
            pred_token, pred = occurred_label_tokens[0]
            prediction_category.append("one label token")
        elif len(occurred_label_tokens) > 1:
            pred_token, pred = random.choice(occurred_label_tokens)
            prediction_category.append("more label tokens")
        elif len (occurred_label_tokens) == 0:
            others = [ele for ele in unique_labels if ele != label]
            pred = others[0]
            pred_token = "other"
            print(cleaned_response_text)
            prediction_category.append("no label token")

        all_preds.append(pred)
        all_answers.append(cleaned_response_text)
        pred_tokens.append(pred_token)


    return all_preds, all_answers, pred_tokens, prediction_category
