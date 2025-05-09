#!/usr/bin/env python3
"""
Main entry point for the MOM (Minutes of Meeting) application.
Handles application startup, shutdown, and signal management.
"""

import os
import sys
import signal
import logging
import threading
import time
from typing import Optional
from flask import Flask
from werkzeug.serving import is_running_from_reloader
from app import create_app, socketio
from services.audio_service import AudioService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mom_app.log')
    ]
)
logger = logging.getLogger(__name__)

class ApplicationManager:
    """Manages the lifecycle of the Flask application."""
    
    def __init__(self):
        self.app: Optional[Flask] = None
        self.shutdown_event = threading.Event()
        self.audio_service = AudioService()
        
        # Track active connections
        self.active_connections = 0
        self.connections_lock = threading.Lock()
        
    def initialize(self) -> None:
        """Initialize the application and its components."""
        try:
            # Create Flask app
            self.app = create_app()
            
            # Register socket.io handlers
            self._setup_socketio_handlers()
            
            # Register signal handlers
            self._setup_signal_handlers()
            
            # Verify upload directory
            self._verify_upload_directory()
            
            logger.info("Application initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize application: %s", str(e))
            raise
    
    def _setup_socketio_handlers(self) -> None:
        """Set up Socket.IO connection handlers."""
        @socketio.on('connect')
        def handle_connect():
            with self.connections_lock:
                self.active_connections += 1
            logger.info("Client connected. Active connections: %d", self.active_connections)
        
        @socketio.on('disconnect')
        def handle_disconnect():
            with self.connections_lock:
                self.active_connections -= 1
            logger.info("Client disconnected. Active connections: %d", self.active_connections)
    
    def _setup_signal_handlers(self) -> None:
        """Set up handlers for system signals."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self._signal_handler)
    
    def _verify_upload_directory(self) -> None:
        """Verify and create upload directory if needed."""
        upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        if not os.path.exists(upload_dir):
            try:
                os.makedirs(upload_dir)
                logger.info("Created upload directory: %s", upload_dir)
            except Exception as e:
                logger.error("Failed to create upload directory: %s", str(e))
                raise
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle system signals for graceful shutdown."""
        signal_name = signal.Signals(signum).name
        logger.info("Received signal %s. Initiating graceful shutdown...", signal_name)
        self.initiate_shutdown()
    
    def initiate_shutdown(self) -> None:
        """Initiate graceful shutdown procedure."""
        logger.info("Initiating application shutdown...")
        self.shutdown_event.set()
        
        # Wait for active connections to close
        shutdown_timeout = 10  # seconds
        start_time = time.time()
        
        while self.active_connections > 0 and time.time() - start_time < shutdown_timeout:
            logger.info("Waiting for %d active connections to close...", self.active_connections)
            time.sleep(1)
        
        # Cleanup resources
        try:
            self.cleanup_resources()
        except Exception as e:
            logger.error("Error during cleanup: %s", str(e))
        
        # Exit
        sys.exit(0)
    
    def cleanup_resources(self) -> None:
        """Clean up application resources."""
        logger.info("Cleaning up application resources...")
        
        # Cleanup audio service
        try:
            self.audio_service.cleanup_resources()
            logger.info("Audio service cleaned up successfully")
        except Exception as e:
            logger.error("Error cleaning up audio service: %s", str(e))
        
        # Additional cleanup can be added here
    
    def run(self) -> None:
        """Run the Flask application."""
        try:
            if not is_running_from_reloader() or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
                logger.info("Starting application server...")
            
            socketio.run(
                self.app,
                debug=True,
                host='0.0.0.0',
                port=8080,
                allow_unsafe_werkzeug=True,
                use_reloader=True
            )
            
        except Exception as e:
            logger.error("Error running application: %s", str(e))
            self.initiate_shutdown()

def main():
    """Main entry point of the application."""
    try:
        app_manager = ApplicationManager()
        app_manager.initialize()
        app_manager.run()
    except Exception as e:
        logger.critical("Fatal error: %s", str(e))
        sys.exit(1)

if __name__ == '__main__':
    main() 