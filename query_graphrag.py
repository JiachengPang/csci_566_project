import subprocess
import json

MODE = 'clean' # 'clean' or 'poisoned' or 'poisoned_eandr'
METHOD = 'local' # 'global' or 'local'
QUESTIONS_PATH = './contexts/questions_500_contexts_V50_1.json'

if MODE == 'clean':
    RAG_DIR = './graphrag_clean'
    RESPONSES = f'graphrag_500_responses_clean_{METHOD}.json'
elif MODE == 'poisoned':
    RAG_DIR = './graphrag_V50'
    RESPONSES = f'graphrag_responses_500_poisoned_{METHOD}.json'
elif MODE == 'poisoned_eandr':
    RAG_DIR = './ragtest_poison_eandr'
    RESPONSES = f'graphrag_responses_poisoned_eandr_{METHOD}.json'

def generate_prompt(question):
    prompt = f"{question}\nYou should strictly limit your answer to less than 10 words."
    return prompt

def query_graphrag(prompt, method, rag_dir):
    command = f'python -m graphrag.query --root {rag_dir} --method {method} \"{prompt}\"' 
    print(f'Running with command: {command}')
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    out = result.stdout
    keyword = f'SUCCESS: {method.capitalize()} Search Response:'
    index = out.find(keyword)

    if index != -1:
        response = out[(index + len(keyword)):].strip()
    else:
        print(f'Failed to gather response, result is {result}')
        return None

    print(f'Response: {response}')
    return response

if __name__ == '__main__':
    rag_dir = RAG_DIR
    output_path = RESPONSES
    
    with open(QUESTIONS_PATH, 'r') as inputs:
        data = json.load(inputs)
    
    with open(RESPONSES, 'w', encoding='utf-8') as f:
        f.write("[\n")
        for i, item in enumerate(data):
            print(f'Processing question id: {item["question_id"]}')
            question = item['question']
            response = query_graphrag(generate_prompt(question), METHOD, rag_dir)
            item['response'] = response

            json.dump(item, f,indent=4)
            if i < len(data) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]")
    
    with open(output_path, 'w') as outputs:
        json.dump(data, outputs, indent=4)