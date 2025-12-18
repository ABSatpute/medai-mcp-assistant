from sqlalchemy import create_engine, text
from decimal import Decimal
import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URI)

class Cart:
    def __init__(self, user_id):
        self.user_id = user_id
        self.cart_id = None
        self.items = []
        self.total_amount = 0.0
    
    def get_or_create_cart(self):
        """Get existing cart or create new one"""
        try:
            with engine.connect() as conn:
                # Check if cart exists
                query = text("SELECT cart_id FROM shopping_cart WHERE user_id = :user_id")
                result = conn.execute(query, {'user_id': self.user_id}).fetchone()
                
                if result:
                    self.cart_id = result.cart_id
                else:
                    # Create new cart
                    insert_query = text("INSERT INTO shopping_cart (user_id) VALUES (:user_id)")
                    conn.execute(insert_query, {'user_id': self.user_id})
                    conn.commit()
                    
                    # Get the new cart_id
                    result = conn.execute(query, {'user_id': self.user_id}).fetchone()
                    self.cart_id = result.cart_id
                
                return self.cart_id
        except Exception as e:
            print(f"Error getting/creating cart: {e}")
            return None
    
    def add_item(self, medicine_id, store_id, quantity, unit_price):
        """Add item to cart"""
        try:
            cart_id = self.get_or_create_cart()
            if not cart_id:
                return False
            
            total_price = float(unit_price) * quantity
            
            with engine.connect() as conn:
                # Check if item already exists
                check_query = text("""
                    SELECT item_id, quantity FROM cart_items 
                    WHERE cart_id = :cart_id AND medicine_id = :medicine_id AND store_id = :store_id
                """)
                existing = conn.execute(check_query, {
                    'cart_id': cart_id,
                    'medicine_id': medicine_id,
                    'store_id': store_id
                }).fetchone()
                
                if existing:
                    # Update quantity
                    new_quantity = existing.quantity + quantity
                    new_total = float(unit_price) * new_quantity
                    update_query = text("""
                        UPDATE cart_items 
                        SET quantity = :quantity, total_price = :total_price 
                        WHERE item_id = :item_id
                    """)
                    conn.execute(update_query, {
                        'quantity': new_quantity,
                        'total_price': new_total,
                        'item_id': existing.item_id
                    })
                else:
                    # Insert new item
                    insert_query = text("""
                        INSERT INTO cart_items (cart_id, medicine_id, store_id, quantity, unit_price, total_price)
                        VALUES (:cart_id, :medicine_id, :store_id, :quantity, :unit_price, :total_price)
                    """)
                    conn.execute(insert_query, {
                        'cart_id': cart_id,
                        'medicine_id': medicine_id,
                        'store_id': store_id,
                        'quantity': quantity,
                        'unit_price': unit_price,
                        'total_price': total_price
                    })
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error adding item to cart: {e}")
            return False
    
    def get_items(self):
        """Get all items in cart with medicine and store details"""
        try:
            cart_id = self.get_or_create_cart()
            if not cart_id:
                return []
            
            with engine.connect() as conn:
                query = text("""
                    SELECT ci.item_id, ci.quantity, ci.unit_price, ci.total_price,
                           m.medicine_name, m.brand_name,
                           ms.store_name, ms.address
                    FROM cart_items ci
                    JOIN medicines m ON ci.medicine_id = m.medicine_id
                    JOIN medical_stores ms ON ci.store_id = ms.store_id
                    WHERE ci.cart_id = :cart_id
                """)
                results = conn.execute(query, {'cart_id': cart_id}).fetchall()
                
                items = []
                total = 0.0
                for row in results:
                    item = {
                        'item_id': row.item_id,
                        'medicine_name': row.medicine_name,
                        'brand_name': row.brand_name,
                        'store_name': row.store_name,
                        'store_address': row.address,
                        'quantity': row.quantity,
                        'unit_price': float(row.unit_price),
                        'total_price': float(row.total_price)
                    }
                    items.append(item)
                    total += float(row.total_price)
                
                self.items = items
                self.total_amount = total
                return items
        except Exception as e:
            print(f"Error getting cart items: {e}")
            return []
    
    def update_quantity(self, item_id, quantity):
        """Update item quantity"""
        try:
            with engine.connect() as conn:
                # Get unit price
                price_query = text("SELECT unit_price FROM cart_items WHERE item_id = :item_id")
                result = conn.execute(price_query, {'item_id': item_id}).fetchone()
                
                if result:
                    new_total = float(result.unit_price) * quantity
                    update_query = text("""
                        UPDATE cart_items 
                        SET quantity = :quantity, total_price = :total_price 
                        WHERE item_id = :item_id
                    """)
                    conn.execute(update_query, {
                        'quantity': quantity,
                        'total_price': new_total,
                        'item_id': item_id
                    })
                    conn.commit()
                    return True
                return False
        except Exception as e:
            print(f"Error updating quantity: {e}")
            return False
    
    def remove_item(self, item_id):
        """Remove item from cart"""
        try:
            with engine.connect() as conn:
                query = text("DELETE FROM cart_items WHERE item_id = :item_id")
                conn.execute(query, {'item_id': item_id})
                conn.commit()
                return True
        except Exception as e:
            print(f"Error removing item: {e}")
            return False
    
    def clear_cart(self):
        """Clear all items from cart"""
        try:
            cart_id = self.get_or_create_cart()
            if not cart_id:
                return False
            
            with engine.connect() as conn:
                query = text("DELETE FROM cart_items WHERE cart_id = :cart_id")
                conn.execute(query, {'cart_id': cart_id})
                conn.commit()
                return True
        except Exception as e:
            print(f"Error clearing cart: {e}")
            return False
    
    def get_item_count(self):
        """Get total number of items in cart"""
        try:
            cart_id = self.get_or_create_cart()
            if not cart_id:
                return 0
            
            with engine.connect() as conn:
                query = text("SELECT COUNT(*) as count FROM cart_items WHERE cart_id = :cart_id")
                result = conn.execute(query, {'cart_id': cart_id}).fetchone()
                return result.count if result else 0
        except Exception as e:
            print(f"Error getting item count: {e}")
            return 0
