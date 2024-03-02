from flask import Flask
from flask_websockets import WebSockets

app = Flask(__name__)
sockets = WebSockets(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
