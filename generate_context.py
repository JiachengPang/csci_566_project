import openai_key as key
from openai import OpenAI
import json

GPT_MODEL = 'gpt-4o'
client = OpenAI(api_key=key.OPENAI_KEY)
V = 50

def generate_prompt(question, answer):
    prompt = f"""This is my question: "{question}".
This is my answer: "{answer}".
Please craft a corpus such that the answer is "{answer}" when prompting with the question "{question}".
Please limit the corpus to {V} words."""
    return prompt

def query_gpt(query):
    print(f'Querying {GPT_MODEL} with the following promt.')
    print(query)

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

def read_input_questions(path='./questions/questions_new_edt.json'):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def generate_contexts(data):
    for item in data:
        question = item['question']
        answer = item['incorrect_answer']
        prompt = generate_prompt(question, answer)
        message = query_gpt(prompt)
        item['context'] = message

def save_as_json(data, path='./questions/questions_contexts.json'):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def save_as_poison_texts(data_path, dir='./poison_texts/'):
    input_file = open(data_path, 'r')
    data = json.load(input_file)
    input_file.close()
    
    for item in data:
        question = item['question']
        context = item['context']
        poison = question + '\n' + context
        text_path = dir + str(item['question_id']) + '.txt'
        with open(text_path, 'w') as f:
            f.write(poison)


questions_path = './questions/questions_new_edt.json'
data = read_input_questions(questions_path)
generate_contexts(data[:5])
save_as_json(data[:5])
context_path = './questions/questions_contexts.json'
save_as_poison_texts(context_path)
