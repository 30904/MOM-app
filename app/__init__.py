from flask import Flask
from flask_socketio import SocketIO
from config import Config

socketio = SocketIO()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize SocketIO with specific configuration
    socketio.init_app(app, 
                     cors_allowed_origins="*",
                     async_mode='eventlet',
                     logger=True,
                     engineio_logger=True)
    
    # Initialize the upload folder
    config_class.init_app(app)
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    return app 