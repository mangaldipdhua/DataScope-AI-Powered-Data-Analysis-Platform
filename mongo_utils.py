import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json

logger = logging.getLogger(__name__)

class MongoDatabase:
    def __init__(self):
        self.connected = False
        self.client = None
        self.db = None
        self.init_connection()
    
    def init_connection(self):
        """Initialize MongoDB connection with detailed logging"""
        try:
            import pymongo
            from pymongo import MongoClient
            
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/datascope')
            db_name = os.getenv('MONGODB_DB_NAME', 'datascope')
            
            logger.info(f"Attempting to connect to MongoDB...")
            logger.info(f"MongoDB URI: {mongodb_uri[:50]}...")
            logger.info(f"Database Name: {db_name}")
            
            self.client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            
            self.db = self.client[db_name]
            
            self.client.admin.command('ping')
            self.connected = True
            
            logger.info("✓ MongoDB connection successful!")
            logger.info(f"✓ Connected to database: {db_name}")
            
            self._init_collections()
            self._create_indexes()
            
        except Exception as e:
            self.connected = False
            logger.error(f"✗ MongoDB connection failed: {type(e).__name__}: {e}")
            logger.warning("MongoDB will be disabled. App will work without persistent storage.")
    
    def _init_collections(self):
        """Initialize MongoDB collections"""
        try:
            if not self.connected or self.db is None:
                return
            
            collections_to_create = [
                os.getenv('MONGODB_PLOTS_COLLECTION', 'plots'),
                os.getenv('MONGODB_ANALYSIS_COLLECTION', 'analysis'),
                os.getenv('MONGODB_UPLOADS_COLLECTION', 'uploads')
            ]
            
            existing_collections = self.db.list_collection_names()
            
            for collection_name in collections_to_create:
                if collection_name not in existing_collections:
                    self.db.create_collection(collection_name)
                    logger.info(f"✓ Created collection: {collection_name}")
                else:
                    logger.info(f"✓ Collection exists: {collection_name}")
                    
        except Exception as e:
            logger.warning(f"Error initializing collections: {e}")
    
    def _create_indexes(self):
        """Create TTL and other indexes"""
        try:
            if not self.connected or self.db is None:
                return
            
            uploads_collection = self.db[os.getenv('MONGODB_UPLOADS_COLLECTION', 'uploads')]
            
            cleanup_hours = int(os.getenv('CLEANUP_AGE_HOURS', '24'))
            
            uploads_collection.create_index('createdAt', expireAfterSeconds=cleanup_hours*3600)
            logger.info(f"✓ Created TTL index: {cleanup_hours} hours for uploads")
            
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def save_analysis(self, analysis_id: str, analysis_data: Dict):
        """Save analysis to MongoDB"""
        if not self.connected or self.db is None:
            logger.debug("MongoDB not connected. Skipping save_analysis.")
            return False
        
        try:
            collection = self.db[os.getenv('MONGODB_ANALYSIS_COLLECTION', 'analysis')]
            
            document = {
                '_id': analysis_id,
                'data': analysis_data,
                'createdAt': datetime.utcnow(),
                'updatedAt': datetime.utcnow()
            }
            
            result = collection.update_one(
                {'_id': analysis_id},
                {'$set': document},
                upsert=True
            )
            
            logger.info(f"✓ Saved analysis to MongoDB: {analysis_id}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save analysis: {e}")
            return False
    
    def save_plot(self, plot_id: str, plot_metadata: Dict):
        """Save plot metadata to MongoDB"""
        if not self.connected or self.db is None:
            logger.debug("MongoDB not connected. Skipping save_plot.")
            return False
        
        try:
            collection = self.db[os.getenv('MONGODB_PLOTS_COLLECTION', 'plots')]
            
            document = {
                '_id': plot_id,
                'metadata': plot_metadata,
                'createdAt': datetime.utcnow()
            }
            
            result = collection.update_one(
                {'_id': plot_id},
                {'$set': document},
                upsert=True
            )
            
            logger.info(f"✓ Saved plot to MongoDB: {plot_id}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save plot: {e}")
            return False
    
    def save_upload_info(self, upload_id: str, upload_data: Dict):
        """Save upload info to MongoDB"""
        if not self.connected or self.db is None:
            logger.debug("MongoDB not connected. Skipping save_upload_info.")
            return False
        
        try:
            collection = self.db[os.getenv('MONGODB_UPLOADS_COLLECTION', 'uploads')]
            
            upload_path = upload_data.get('upload_path')
            if upload_path is not None:
                upload_path = str(upload_path)
            
            document = {
                '_id': upload_id,
                'filename': upload_data.get('filename'),
                'file_size': upload_data.get('file_size'),
                'file_type': upload_data.get('file_type'),
                'upload_path': upload_path,
                'createdAt': datetime.utcnow()
            }
            
            result = collection.update_one(
                {'_id': upload_id},
                {'$set': document},
                upsert=True
            )
            
            logger.info(f"✓ Saved upload info to MongoDB: {upload_id}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save upload info: {e}")
            return False
    
    def get_analysis(self, analysis_id: str):
        """Retrieve analysis from MongoDB"""
        if not self.connected or self.db is None:
            return None
        
        try:
            collection = self.db[os.getenv('MONGODB_ANALYSIS_COLLECTION', 'analysis')]
            result = collection.find_one({'_id': analysis_id})
            if result:
                logger.info(f"✓ Retrieved analysis from MongoDB: {analysis_id}")
            return result
        except Exception as e:
            logger.error(f"✗ Failed to retrieve analysis: {e}")
            return None
    
    def get_latest_analysis(self):
        """Retrieve the latest analysis from MongoDB for chat context"""
        if not self.connected or self.db is None:
            return None
        
        try:
            collection = self.db[os.getenv('MONGODB_ANALYSIS_COLLECTION', 'analysis')]
            result = collection.find_one(
                sort=[('createdAt', -1)]
            )
            if result:
                logger.info(f"✓ Retrieved latest analysis from MongoDB")
                return result.get('data', result)
            return None
        except Exception as e:
            logger.error(f"✗ Failed to retrieve latest analysis: {e}")
            return None
    
    def get_analysis_by_filename(self, filename: str):
        """Retrieve analysis by filename for chat context"""
        if not self.connected or self.db is None:
            return None
        
        try:
            collection = self.db[os.getenv('MONGODB_ANALYSIS_COLLECTION', 'analysis')]
            result = collection.find_one(
                sort=[('createdAt', -1)]
            )
            if result and result.get('data', {}).get('filename') == filename:
                logger.info(f"✓ Retrieved analysis for file: {filename}")
                return result.get('data', result)
            return None
        except Exception as e:
            logger.error(f"✗ Failed to retrieve analysis by filename: {e}")
            return None
    
    def get_plot(self, plot_id: str):
        """Retrieve plot metadata from MongoDB"""
        if not self.connected or self.db is None:
            return None
        
        try:
            collection = self.db[os.getenv('MONGODB_PLOTS_COLLECTION', 'plots')]
            result = collection.find_one({'_id': plot_id})
            if result:
                logger.info(f"✓ Retrieved plot from MongoDB: {plot_id}")
            return result
        except Exception as e:
            logger.error(f"✗ Failed to retrieve plot: {e}")
            return None
    
    def cleanup_old_records(self):
        """Clean up old records based on TTL"""
        if not self.connected or self.db is None:
            logger.debug("MongoDB not connected. Skipping cleanup.")
            return
        
        try:
            cleanup_hours = int(os.getenv('CLEANUP_AGE_HOURS', '24'))
            cutoff_time = datetime.utcnow() - timedelta(hours=cleanup_hours)
            
            uploads_collection = self.db[os.getenv('MONGODB_UPLOADS_COLLECTION', 'uploads')]
            
            result = uploads_collection.delete_many({'createdAt': {'$lt': cutoff_time}})
            
            if result.deleted_count > 0:
                logger.info(f"✓ Cleaned up {result.deleted_count} old upload records from MongoDB")
            
        except Exception as e:
            logger.error(f"✗ Cleanup failed: {e}")
    
    def close(self):
        """Close MongoDB connection"""
        try:
            if self.client:
                self.client.close()
                self.connected = False
                logger.info("✓ MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")


db = MongoDatabase()
