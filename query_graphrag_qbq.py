import subprocess
import json
import shutil
import os

METHOD = 'local'
QUESTIONS_PATH = './contexts/questions_new_contexts_V50.json'
RAG_DIR = './ragtest_poison/'
POISON_TEXTS_DIR = './poison_texts/poison_texts_new_V50'
RESPONSES = f'graphrag_responses_qbq_V50_{METHOD}.json'
N = 5

poisons = []

def generate_prompt(question):
    prompt = f"{question}\nYou should strictly limit your answer to less than 10 words."
    return prompt

# def index_graphrag_for_q(q, rag_dir):
#     global poisons
#     # delete old poisons
#     for path in poisons:
#         if os.path.exists(path):
#             os.remove(path)
#     poisons = []

#     # insert new poisons
#     source = f'{POISON_TEXTS_DIR}{q}.txt'
#     for i in range(N):
#         dest = f'{rag_dir}input/{q}_copy{i}.txt'
#         poisons.append(dest)
#         shutil.copy(source, dest)
    
#     # index graphrag
#     command = f'python -m graphrag.index --root {rag_dir}'
#     print(f'Running with command: {command}')
#     with subprocess.Popen(command) as process:
#         process.wait()

def index_graphrag_for_q(q, rag_dir):
    global poisons
    # delete old poisons
    for path in poisons:
        if os.path.exists(path):
            os.remove(path)
    poisons = []

    # insert new poisons
    
    for i in range(N):
        source = f'{POISON_TEXTS_DIR}_{i+1}/{q}.txt'
        dest = f'{rag_dir}input/{q}_poison{i+1}.txt'
        poisons.append(dest)
        shutil.copy(source, dest)
    
    # index graphrag
    command = f'python -m graphrag.index --root {rag_dir}'
    print(f'Running with command: {command}')
    with subprocess.Popen(command) as process:
        process.wait()

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

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("[\n")
        for i, item in enumerate(data):
            print(f'Processing question id: {item["question_id"]}')
            index_graphrag_for_q(str(item['question_id']), rag_dir)
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