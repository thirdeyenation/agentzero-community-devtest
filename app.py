from flask import Flask, render_template, request, jsonify
from agent_zero import AgentZero

app = Flask(__name__)
agent_zero = AgentZero()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    user_input = request.json['input']
    result = agent_zero.process_input(user_input)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)