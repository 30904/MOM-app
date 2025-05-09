from app import create_app, socketio
from config import Config

app = create_app(Config)

if __name__ == '__main__':
    try:
        socketio.run(app,
                    debug=True,
                    host='0.0.0.0',
                    port=8080,
                    allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Error: {e}") 