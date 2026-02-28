"""Start the CodeMind chat server."""
from codemind.chat.server import run

if __name__ == '__main__':
    run(host="0.0.0.0", port=8000)
