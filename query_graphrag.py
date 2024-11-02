import subprocess
import json

MODE = 'clean' # 'clean' or 'poisoned'

if MODE == 'clean':
    RAG_DIR = './ragtest'
    RESPONSES = 'graphrag_responses_clean.json'
else:
    RAG_DIR = './ragtest_poison'
    RESPONSES = 'graphrag_responses_poisoned.json'

def generate_prompt(question):
    prompt = f"""{question}

You should strictly limit your answer to less than 10 words."""
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
    method = 'local'
    rag_dir = RAG_DIR
    question_path = './questions/questions_contexts_1.json'
    output_path = RESPONSES
    
    with open(question_path, 'r') as inputs:
        data = json.load(inputs)
        
    for item in data:
        print(f'Processing question id: {item["question_id"]}')
        question = item['question']
        response = query_graphrag(generate_prompt(question), method, rag_dir)
        item['response'] = response
    
    with open(output_path, 'w') as outputs:
        json.dump(data, outputs, indent=4)