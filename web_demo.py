from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return {
        'project': 'CogniGuard Vision Agent',
        'status': 'Running',
        'description': 'Realtime AI Vision Monitoring using Vision Agents SDK and local Gemma3 model'
    }

app.run(host='0.0.0.0', port=10000)
