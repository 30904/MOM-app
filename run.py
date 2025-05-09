import eventlet
eventlet.monkey_patch()

from app import create_app, socketio
from config import Config

app = create_app(Config)

if __name__ == '__main__':
    try:
        # Use eventlet's WSGI server
        socketio.run(app,
                    debug=True,
                    host='0.0.0.0',
                    port=5000,
                    use_reloader=True,
                    log_output=True)
    except Exception as e:
        print(f"Error: {e}") 