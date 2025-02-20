import re
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from config import init, close, get_gsm8k_questions, Config


def sampler(
    model,
    input_string,
    temperature=0.0,
    top_p=1.0,
    max_prompt_length=None,
    max_completion_length=256,
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        truncate_prompt_tokens=max_prompt_length,
        max_tokens=max_completion_length,
    )
    output = model.generate([input_string], sampling_params, use_tqdm=False)
    return output[0].outputs[0].text


def extract_xml_answer(text, start_tag="<answer>", end_tag="</answer>"):
    """Extract the content within tags from a string using regex"""

    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    match = re.search(
        pattern, text, re.DOTALL
    )  # DOTALL allows matching across multiple lines

    if match:
        answer = match.group(1)
        answer = re.sub(r"[%$]", "", answer).strip()  # Remove '%' and '$'
        return answer

    return ""


def extract_last_xml_answer(text, start_tag="<answer>", end_tag="</answer>"):
    """Extract the content within the last occurrence of tags from a string using regex"""

    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, re.DOTALL)  # Find all matches

    if matches:
        answer = matches[-1]  # Get the last match
        answer = re.sub(r"[%$]", "", answer).strip()  # Remove '%' and '$'
        return answer

    return ""


def find_number(search_string):
    """Finds the last number to appear in a string"""

    # Use regular expression to find all numbers in the search string
    numbers = re.compile(
        r"-?[\d,]*\.?\d+",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(search_string)

    if numbers:
        return numbers[-1]  # Return the last number found
    else:
        return ""  # Return empty string if no number is found


def remove_symbols(x: str) -> str:
    # Example: 5,600 -> 5600 | 55% -> 55 | 30$ -> 30
    return x.replace(",", "").replace("%", "").replace("$", "").strip()


def get_num_tokens(text, tokenizer):
    encoding = tokenizer(text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    return len(input_ids[0])


if __name__ == "__main__":

    init()
    params = Config()

    # Load the model
    gsm8k_test = get_gsm8k_questions("test")

    MODEL_NAME = "Google/gemma-2-2b-it"
    #MODEL_NAME = "gemma-2-2b-it-grpo"

    llm = LLM(model=MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ground_truth = {}
    answers = {}
    input_tokens = []
    output_tokens = []
    idx = 0
    correct_format = 0
    plausibly_correct = 0
    correct = 0

    for task_id, item in tqdm(enumerate(gsm8k_test), total=len(gsm8k_test)):

        # Formulate and print the full prompt
        prompt = item["prompt"][0]["content"]
        ground_truth[task_id] = item["answer"]

        input_tokens.append(get_num_tokens(prompt, tokenizer))
        response = sampler(
            llm,
            input_string=prompt,
            temperature=0,
            max_prompt_length=params.max_prompt_length,
            max_completion_length=params.max_completion_length,
        )
        output_tokens.append(get_num_tokens(response, tokenizer))
        answers[task_id] = remove_symbols(find_number(response))

        pattern = r"^<reasoning>[\s\S]*?<\/reasoning>\s*<answer>[\s\S]*?<\/answer>$"
        if re.match(pattern, response.strip()):
            correct_format += 1

        if (
            answers[task_id] == ground_truth[task_id]
            or extract_last_xml_answer(response) == ground_truth[task_id]
        ):
            plausibly_correct += 1

        if extract_last_xml_answer(response) == ground_truth[task_id]:
            correct += 1

        idx += 1

print("-" * 40)

print(f"Input:  max tokens: {max(input_tokens)}")
print(f"        avg tokens: {sum(input_tokens) / (idx + 1):.1f}")

print(f"Output: max tokens: {max(output_tokens)}")
print(f"        avg tokens: {sum(output_tokens) / (idx + 1):.1f}")

print(
    f"Correct format:       {correct_format} out of {idx+1} "
    f"({correct_format / (idx + 1) * 100:.1f}%)"
)

print(
    f"Plausibly correct:    {plausibly_correct} out of {idx+1} "
    f"({plausibly_correct / (idx + 1) * 100:.1f}%)"
)

print(
    f"Correct:              {correct} out of {idx+1} "
    f"({correct / (idx + 1) * 100:.1f}%)"
)

print("=" * 40)

close(llm)
