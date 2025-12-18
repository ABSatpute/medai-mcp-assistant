#!/usr/bin/env python3
"""
Database Setup Script for MedAI E-Commerce
Run this script to create the e-commerce tables in your MySQL database
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_ecommerce_database():
    """Create e-commerce tables in the existing MySQL database"""
    
    # Database connection
    DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    
    try:
        engine = create_engine(DB_URI)
        
        # Read SQL schema
        with open('database/ecommerce_schema.sql', 'r') as file:
            sql_content = file.read()
        
        # Split SQL statements (remove comments and empty lines)
        statements = []
        for statement in sql_content.split(';'):
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                statements.append(statement)
        
        print("🚀 Setting up e-commerce database tables...")
        
        with engine.begin() as conn:  # Use begin() for auto-commit
            for i, statement in enumerate(statements):
                if statement.strip():
                    try:
                        conn.execute(text(statement))
                        print(f"✅ Executed statement {i+1}/{len(statements)}")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            print(f"⚠️  Table already exists (skipping): {e}")
                        else:
                            print(f"❌ Error in statement {i+1}: {e}")
                            return False
        
        print("\n🎉 E-commerce database setup completed successfully!")
        print("\n📋 Created tables:")
        print("   - users (authentication)")
        print("   - user_addresses (delivery addresses)")
        print("   - shopping_cart & cart_items")
        print("   - orders & order_items")
        print("   - order_status_history")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def verify_tables():
    """Verify that all tables were created successfully"""
    DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    
    expected_tables = [
        'users', 'user_addresses', 'shopping_cart', 
        'cart_items', 'orders', 'order_items', 'order_status_history'
    ]
    
    try:
        engine = create_engine(DB_URI)
        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES")).fetchall()
            existing_tables = [row[0] for row in result]
            
            print("\n🔍 Verifying tables...")
            all_created = True
            for table in expected_tables:
                if table in existing_tables:
                    print(f"✅ {table}")
                else:
                    print(f"❌ {table} - NOT FOUND")
                    all_created = False
            
            return all_created
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🏥 MedAI E-Commerce Database Setup")
    print("=" * 40)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please ensure database credentials are set.")
        sys.exit(1)
    
    # Check if schema file exists
    if not os.path.exists('database/ecommerce_schema.sql'):
        print("❌ Schema file not found: database/ecommerce_schema.sql")
        sys.exit(1)
    
    # Setup database
    if setup_ecommerce_database():
        if verify_tables():
            print("\n🎉 All tables created successfully!")
            print("\n📝 Next steps:")
            print("   1. Install new dependencies: pip install flask-jwt-extended bcrypt")
            print("   2. Restart your Flask app")
            print("   3. E-commerce features will be available!")
        else:
            print("\n⚠️  Some tables may not have been created properly.")
    else:
        print("\n❌ Database setup failed. Please check your database connection and try again.")
