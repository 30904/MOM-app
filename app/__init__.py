from flask import Flask
from flask_socketio import SocketIO
from pymongo import MongoClient
from config import Config

socketio = SocketIO()
mongo = None

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize MongoDB
    global mongo
    mongo = MongoClient(app.config['MONGODB_URI'])
    
    # Initialize SocketIO
    socketio.init_app(app, async_mode=app.config['SOCKETIO_ASYNC_MODE'])
    
    # Initialize the upload folder
    config_class.init_app(app)
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    return app 