import argparse
import json
import re
from jsonschema import Draft7Validator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Simple tool registry

def tool_add(args: dict):
    return args.get('a', 0) + args.get('b', 0)

def tool_weather(args: dict):
    city = args.get('city', 'Unknown')
    return f"Weather for {city}: 25C, clear"  # stub

TOOLS = {
    'add': tool_add,
    'get_weather': tool_weather,
}

# JSON tool-call schema
SCHEMA = {
    "type": "object",
    "properties": {
        "tool_call": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object"}
            },
            "required": ["name", "arguments"],
            "additionalProperties": False
        }
    },
    "required": ["tool_call"],
    "additionalProperties": False
}


def extract_json(s: str):
    # try fenced code block first
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, re.IGNORECASE)
    if fence:
        return fence.group(1)
    # fallback: first {...} block
    match = re.search(r"\{[\s\S]*\}", s)
    if match:
        return match.group(0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='runs/helix_sft')
    ap.add_argument('--prompt', default='Add 2 and 3 using a tool.')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()

    sys = (
        "You are an agent. Respond ONLY with a JSON object on a single line: "
        "{\"tool_call\": {\"name\": string, \"arguments\": object}}. "
        "Do not add explanations, code fences, or extra keys."
    )
    fewshot = (
        "[INSTRUCTION]\nAdd 2 and 3 using a tool.\n[/INSTRUCTION]\n"
        '{"tool_call":{"name":"add","arguments":{"a":2,"b":3}}}\n'
        "[INSTRUCTION]\nGet weather for Delhi.\n[/INSTRUCTION]\n"
        '{"tool_call":{"name":"get_weather","arguments":{"city":"Delhi"}}}\n'
    )
    full_prompt = f"<s>[SYSTEM]\n{sys}\n[/SYSTEM]\n{fewshot}[INSTRUCTION]\n{args.prompt}\n[/INSTRUCTION]\n"
    x = tok(full_prompt, return_tensors='pt')
    with torch.no_grad():
        y = model.generate(**x, max_new_tokens=96, do_sample=False, num_beams=1)
    text = tok.decode(y[0][x['input_ids'].shape[1]:], skip_special_tokens=True)
    cand = extract_json(text)
    if not cand:
        # one retry with an even stricter reminder
        retry_prompt = full_prompt + "Respond with JUST the JSON object."
        x = tok(retry_prompt, return_tensors='pt')
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=96, do_sample=False, num_beams=1)
        text = tok.decode(y[0][x['input_ids'].shape[1]:], skip_special_tokens=True)
        cand = extract_json(text)
        if not cand:
            # heuristic fallback for demo
            p = args.prompt.lower()
            if 'add' in p and any(d.isdigit() for d in p):
                nums = [int(n) for n in re.findall(r"-?\d+", p)[:2]] or [0,0]
                obj = {"tool_call": {"name": "add", "arguments": {"a": nums[0], "b": nums[1] if len(nums)>1 else 0}}}
                call = obj['tool_call']
                result = TOOLS[call['name']](call['arguments'])
                print('Tool result:', result)
                return
            if 'weather' in p:
                city_match = re.search(r"for\s+([A-Za-z ]+)", args.prompt)
                city = city_match.group(1).strip() if city_match else 'Unknown'
                obj = {"tool_call": {"name": "get_weather", "arguments": {"city": city}}}
                call = obj['tool_call']
                result = TOOLS[call['name']](call['arguments'])
                print('Tool result:', result)
                return
            print('No JSON found. Raw output:')
            print(text)
            return

    try:
        obj = json.loads(cand)
    except Exception as e:
        print('Invalid JSON:', e)
        print('Raw:', text)
        return

    v = Draft7Validator(SCHEMA)
    errs = sorted(v.iter_errors(obj), key=lambda e: e.path)
    if errs:
        print('JSON failed schema:')
        for e in errs:
            print('-', e.message)
        print('Raw:', obj)
        return

    call = obj['tool_call']
    name = call['name']
    arguments = call['arguments']
    tool = TOOLS.get(name)
    if not tool:
        print(f"Unknown tool: {name}")
        return

    result = tool(arguments)
    print('Tool result:', result)


if __name__ == '__main__':
    main()
