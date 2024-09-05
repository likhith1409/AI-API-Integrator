from flask import Flask, render_template, request, jsonify
import cohere
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os

app = Flask(__name__)

api_keys = {
    'api1': None,
    'api3': None,
    'api4': None,
}

outputs = {
    'api1': '',
    'api3': '',
    'api4': '',
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit_api_key/<api>', methods=['POST'])
def submit_api_key(api):
    key = request.json.get('key')
    api_keys[api] = key
    return jsonify({'status': 'success'})

@app.route('/submit_prompt', methods=['POST'])
def submit_prompt():
    apiSelect = request.json.get('apiSelect')
    prompt = request.json.get('prompt')
    responses = {}

    if apiSelect == 'all' or apiSelect == 'api1':
        if api_keys['api1']:
            print(f"API Key for Llama 3: {api_keys['api1']}")
            print(f"Prompt: {prompt}")
            outputs['api1'] = call_llama3_api(api_keys['api1'], prompt)
            responses['api1'] = outputs['api1']
    if apiSelect == 'all' or apiSelect == 'api3':
        if api_keys['api3']:
            outputs['api3'] = call_gemini_api(api_keys['api3'], prompt)
            responses['api3'] = outputs['api3']
    if apiSelect == 'all' or apiSelect == 'api4':
        if api_keys['api4']:
            outputs['api4'] = call_cohere_api(api_keys['api4'], prompt)
            responses['api4'] = outputs['api4']

    return jsonify(responses)

def call_llama3_api(api_key, prompt):
    try:
        os.environ['GROQ_API_KEY'] = api_key  
        client = Groq()
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content.strip() 
    except Exception as e:
        print(f"Error calling Llama 3 API: {e}")
        return str(e)

def call_gemini_api(api_key, prompt):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def call_cohere_api(api_key, prompt):
    co = cohere.Client(api_key)
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=100
    )
    return response.generations[0].text

@app.route('/similarity_measure')
def similarity_measure():
    texts = [outputs['api1'], outputs['api3'], outputs['api4']]
    vectorizer = TfidfVectorizer().fit_transform(texts)
    pairwise_similarity = cosine_similarity(vectorizer)
    similarity_scores = pairwise_similarity.mean(axis=1)
    return jsonify({
        'api1': similarity_scores[0],
        'api3': similarity_scores[1],
        'api4': similarity_scores[2]
    })

if __name__ == '__main__':
    app.run(debug=True)