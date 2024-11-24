import openai_key as key
from openai import OpenAI
import json
import os
import shutil

GPT_MODEL = 'gpt-4o'
client = OpenAI(api_key=key.OPENAI_KEY)
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]

template='entity_extraction_edt.txt'
with open(template, 'r') as f:
        template = f.read()

def generate_prompt(input_text):
    prompt = template.format(
        entity_types=DEFAULT_ENTITY_TYPES, 
        input_text=input_text, tuple_delimiter=DEFAULT_TUPLE_DELIMITER, 
        record_delimiter=DEFAULT_RECORD_DELIMITER, 
        completion_delimiter=DEFAULT_COMPLETION_DELIMITER)
    
    return prompt

def query_gpt(query, client):
    print(f'Querying {GPT_MODEL}')

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )

    message = response.choices[0].message.content
    print('--------------------------------------------------------------')
    print('Response message is')
    print(message)
    print('--------------------------------------------------------------')
    return message

def generate_eandr_poison(input_path='./questions/questions_contexts.json',
                          output_path='./questions/questions_contexts_eandr.json',
                          poison_texts_dir='./poison_texts_eandr/'):
    
    with open(input_path, 'r') as f:
        data = json.load(f)

    for item in data:
        question = item['question']
        context = item['context']
        input_text = question + '\n' + context
        prompt = generate_prompt(input_text)
        message = query_gpt(prompt, client)
        item['context_eandr'] = message
    
    # save as json
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    # save as individual poison texts
    for item in data:
        text_path = poison_texts_dir + str(item['question_id']) + '_eandr' + '.txt'
        with open(text_path, 'w') as f:
            f.write(item['context_eandr'])

def filter_by_relationship(input_dir='./poison_texts_eandr/', output_dir='./poison_texts_eandr_relationship'):
    print(f'Copying poison eandr texts from {input_dir} to {output_dir}, filtering out those without \"relationship\".')
    filtered = 0
    search_term = "\"relationship\"<|>"
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            if search_term in content:
                shutil.copy(filepath, os.path.join(output_dir, filename))
            else:
                filtered += 1
    print(f'Finished. Filtered out {filtered} files.')

if __name__ == '__main__':
    print('Start generating entity and relationship contexts')
    generate_eandr_poison()
    # filter_by_relationship()
