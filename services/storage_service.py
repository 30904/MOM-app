from datetime import datetime
from bson import ObjectId
from config import Config

class StorageService:
    def __init__(self, mongo_client):
        self.db = mongo_client[Config.DB_NAME]
        self.meetings = self.db.meetings
        self.transcriptions = self.db.transcriptions
        self.action_items = self.db.action_items
    
    def create_meeting(self, initial_data=None):
        """Create a new meeting record."""
        meeting_data = {
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'status': 'created',
            'transcription': '',
            'summary': '',
            'action_items': []
        }
        
        if initial_data:
            meeting_data.update(initial_data)
        
        result = self.meetings.insert_one(meeting_data)
        return result.inserted_id
    
    def update_meeting(self, meeting_id, update_data):
        """Update a meeting record."""
        update_data['updated_at'] = datetime.utcnow()
        
        self.meetings.update_one(
            {'_id': ObjectId(meeting_id)},
            {'$set': update_data}
        )
    
    def update_transcription(self, meeting_id, text, action_items=None):
        """Update meeting transcription and action items."""
        update_data = {
            'transcription': text,
            'updated_at': datetime.utcnow()
        }
        
        if action_items:
            update_data['action_items'] = action_items
        
        self.meetings.update_one(
            {'_id': ObjectId(meeting_id)},
            {
                '$set': update_data,
                '$push': {'transcription_history': {
                    'text': text,
                    'timestamp': datetime.utcnow()
                }}
            }
        )
    
    def get_meeting(self, meeting_id):
        """Retrieve a meeting record."""
        return self.meetings.find_one({'_id': ObjectId(meeting_id)})
    
    def list_meetings(self, limit=10, skip=0):
        """List recent meetings."""
        return list(self.meetings
                   .find()
                   .sort('created_at', -1)
                   .skip(skip)
                   .limit(limit))
    
    def update_action_item(self, meeting_id, action_item_id, update_data):
        """Update a specific action item."""
        self.meetings.update_one(
            {
                '_id': ObjectId(meeting_id),
                'action_items._id': ObjectId(action_item_id)
            },
            {
                '$set': {
                    'action_items.$.status': update_data.get('status', 'pending'),
                    'action_items.$.updated_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
            }
        )
    
    def delete_meeting(self, meeting_id):
        """Delete a meeting and its associated data."""
        self.meetings.delete_one({'_id': ObjectId(meeting_id)}) 