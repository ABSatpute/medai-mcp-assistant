import uuid
import bcrypt
from datetime import datetime
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection
DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URI)

class User:
    def __init__(self, user_id=None, email=None, full_name=None, phone=None):
        self.user_id = user_id or str(uuid.uuid4())
        self.email = email
        self.full_name = full_name
        self.phone = phone
        self.password_hash = None
        self.email_verified = False
        self.is_active = True
    
    @staticmethod
    def hash_password(password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def check_password(password, hashed):
        """Check password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def save(self):
        """Save user to database"""
        try:
            with engine.connect() as conn:
                query = text("""
                    INSERT INTO users (user_id, email, password_hash, full_name, phone, email_verified, is_active)
                    VALUES (:user_id, :email, :password_hash, :full_name, :phone, :email_verified, :is_active)
                """)
                conn.execute(query, {
                    'user_id': self.user_id,
                    'email': self.email,
                    'password_hash': self.password_hash,
                    'full_name': self.full_name,
                    'phone': self.phone,
                    'email_verified': self.email_verified,
                    'is_active': self.is_active
                })
                conn.commit()
                return True
        except Exception as e:
            print(f"Error saving user: {e}")
            return False
    
    @staticmethod
    def find_by_email(email):
        """Find user by email"""
        try:
            with engine.connect() as conn:
                query = text("SELECT * FROM users WHERE email = :email AND is_active = TRUE")
                result = conn.execute(query, {'email': email}).fetchone()
                
                if result:
                    user = User()
                    user.user_id = result.user_id
                    user.email = result.email
                    user.password_hash = result.password_hash
                    user.full_name = result.full_name
                    user.phone = result.phone
                    user.email_verified = result.email_verified
                    user.is_active = result.is_active
                    return user
                return None
        except Exception as e:
            print(f"Error finding user: {e}")
            return None
    
    @staticmethod
    def find_by_id(user_id):
        """Find user by ID"""
        try:
            with engine.connect() as conn:
                query = text("SELECT * FROM users WHERE user_id = :user_id AND is_active = TRUE")
                result = conn.execute(query, {'user_id': user_id}).fetchone()
                
                if result:
                    user = User()
                    user.user_id = result.user_id
                    user.email = result.email
                    user.password_hash = result.password_hash
                    user.full_name = result.full_name
                    user.phone = result.phone
                    user.email_verified = result.email_verified
                    user.is_active = result.is_active
                    return user
                return None
        except Exception as e:
            print(f"Error finding user by ID: {e}")
            return None
    
    def to_dict(self):
        """Convert user to dictionary (excluding password)"""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'full_name': self.full_name,
            'phone': self.phone,
            'email_verified': self.email_verified,
            'is_active': self.is_active
        }
