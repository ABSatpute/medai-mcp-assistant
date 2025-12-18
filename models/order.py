import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import json

load_dotenv()

DB_URI = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
engine = create_engine(DB_URI)

class Order:
    def __init__(self, user_id):
        self.user_id = user_id
        self.order_id = str(uuid.uuid4())
        self.status = "PENDING"
        self.total_amount = 0.0
        self.delivery_fee = 0.0
        self.final_amount = 0.0
    
    def create_order(self, cart_items, address_data, payment_data):
        """Create order from cart items"""
        try:
            with engine.connect() as conn:
                # Calculate totals
                self.total_amount = sum(float(item['total_price']) for item in cart_items)
                self.delivery_fee = 50.0 if self.total_amount < 500 else 0.0  # Free delivery above ₹500
                self.final_amount = self.total_amount + self.delivery_fee
                
                # Save address first
                address_query = text("""
                    INSERT INTO user_addresses (user_id, address_type, full_name, phone, street_address, city, state, postal_code, landmark)
                    VALUES (:user_id, 'HOME', :full_name, :phone, :street_address, :city, :state, :postal_code, :landmark)
                """)
                result = conn.execute(address_query, {
                    'user_id': self.user_id,
                    'full_name': address_data['full_name'],
                    'phone': address_data['phone'],
                    'street_address': address_data['street_address'],
                    'city': address_data['city'],
                    'state': address_data['state'],
                    'postal_code': address_data['postal_code'],
                    'landmark': address_data.get('landmark', '')
                })
                address_id = result.lastrowid
                
                # Estimate delivery (2-3 days from now)
                estimated_delivery = datetime.now() + timedelta(days=2)
                
                # Create order
                order_query = text("""
                    INSERT INTO orders (order_id, user_id, store_id, status, total_amount, delivery_fee, final_amount, 
                                      payment_method, payment_status, razorpay_order_id, razorpay_payment_id, 
                                      delivery_address_id, estimated_delivery)
                    VALUES (:order_id, :user_id, :store_id, 'CONFIRMED', :total_amount, :delivery_fee, :final_amount,
                            'Razorpay', 'COMPLETED', :razorpay_order_id, :razorpay_payment_id, :address_id, :estimated_delivery)
                """)
                
                # Use first item's store_id (in real app, you'd handle multiple stores)
                store_id = 1  # Default store for now
                
                conn.execute(order_query, {
                    'order_id': self.order_id,
                    'user_id': self.user_id,
                    'store_id': store_id,
                    'total_amount': self.total_amount,
                    'delivery_fee': self.delivery_fee,
                    'final_amount': self.final_amount,
                    'razorpay_order_id': payment_data.get('razorpay_order_id'),
                    'razorpay_payment_id': payment_data.get('razorpay_payment_id'),
                    'address_id': address_id,
                    'estimated_delivery': estimated_delivery
                })
                
                # Create order items
                for item in cart_items:
                    item_query = text("""
                        INSERT INTO order_items (order_id, medicine_id, quantity, unit_price, total_price)
                        VALUES (:order_id, :medicine_id, :quantity, :unit_price, :total_price)
                    """)
                    
                    # Get medicine_id from medicine_name (simplified)
                    med_query = text("SELECT medicine_id FROM medicines WHERE medicine_name LIKE :name LIMIT 1")
                    med_result = conn.execute(med_query, {'name': f"%{item['medicine_name']}%"}).fetchone()
                    medicine_id = med_result.medicine_id if med_result else 1
                    
                    conn.execute(item_query, {
                        'order_id': self.order_id,
                        'medicine_id': medicine_id,
                        'quantity': item['quantity'],
                        'unit_price': item['unit_price'],
                        'total_price': item['total_price']
                    })
                
                # Add initial status history
                status_query = text("""
                    INSERT INTO order_status_history (order_id, status, notes)
                    VALUES (:order_id, 'CONFIRMED', 'Order confirmed and payment received')
                """)
                conn.execute(status_query, {
                    'order_id': self.order_id
                })
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error creating order: {e}")
            return False
    
    @staticmethod
    def get_user_orders(user_id, limit=10):
        """Get user's order history"""
        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT o.order_id, o.status, o.total_amount, o.delivery_fee, o.final_amount,
                           o.payment_method, o.estimated_delivery, o.delivered_at, o.created_at,
                           ua.full_name, ua.city, ua.state
                    FROM orders o
                    LEFT JOIN user_addresses ua ON o.delivery_address_id = ua.address_id
                    WHERE o.user_id = :user_id
                    ORDER BY o.created_at DESC
                    LIMIT :limit
                """)
                results = conn.execute(query, {'user_id': user_id, 'limit': limit}).fetchall()
                
                orders = []
                for row in results:
                    order = {
                        'order_id': row.order_id,
                        'status': row.status,
                        'total_amount': float(row.total_amount),
                        'delivery_fee': float(row.delivery_fee),
                        'final_amount': float(row.final_amount),
                        'payment_method': row.payment_method,
                        'estimated_delivery': row.estimated_delivery.isoformat() if row.estimated_delivery else None,
                        'delivered_at': row.delivered_at.isoformat() if row.delivered_at else None,
                        'created_at': row.created_at.isoformat() if row.created_at else None,
                        'delivery_address': {
                            'full_name': row.full_name,
                            'city': row.city,
                            'state': row.state
                        }
                    }
                    orders.append(order)
                
                return orders
                
        except Exception as e:
            print(f"Error getting orders: {e}")
            return []
    
    @staticmethod
    def get_order_details(order_id, user_id):
        """Get detailed order information"""
        try:
            with engine.connect() as conn:
                # Get order info
                order_query = text("""
                    SELECT o.*, ua.full_name, ua.phone, ua.street_address, ua.city, ua.state, ua.postal_code, ua.landmark
                    FROM orders o
                    LEFT JOIN user_addresses ua ON o.delivery_address_id = ua.address_id
                    WHERE o.order_id = :order_id AND o.user_id = :user_id
                """)
                order_result = conn.execute(order_query, {'order_id': order_id, 'user_id': user_id}).fetchone()
                
                if not order_result:
                    return None
                
                # Get order items
                items_query = text("""
                    SELECT oi.*, m.medicine_name, m.brand_name
                    FROM order_items oi
                    JOIN medicines m ON oi.medicine_id = m.medicine_id
                    WHERE oi.order_id = :order_id
                """)
                items_results = conn.execute(items_query, {'order_id': order_id}).fetchall()
                
                # Get status history
                status_query = text("""
                    SELECT status, notes, created_at
                    FROM order_status_history
                    WHERE order_id = :order_id
                    ORDER BY created_at ASC
                """)
                status_results = conn.execute(status_query, {'order_id': order_id}).fetchall()
                
                order_details = {
                    'order_id': order_result.order_id,
                    'status': order_result.status,
                    'total_amount': float(order_result.total_amount),
                    'delivery_fee': float(order_result.delivery_fee),
                    'final_amount': float(order_result.final_amount),
                    'payment_method': order_result.payment_method,
                    'payment_status': order_result.payment_status,
                    'estimated_delivery': order_result.estimated_delivery.isoformat() if order_result.estimated_delivery else None,
                    'created_at': order_result.created_at.isoformat() if order_result.created_at else None,
                    'delivery_address': {
                        'full_name': order_result.full_name,
                        'phone': order_result.phone,
                        'street_address': order_result.street_address,
                        'city': order_result.city,
                        'state': order_result.state,
                        'postal_code': order_result.postal_code,
                        'landmark': order_result.landmark
                    },
                    'items': [
                        {
                            'medicine_name': item.medicine_name,
                            'brand_name': item.brand_name,
                            'quantity': item.quantity,
                            'unit_price': float(item.unit_price),
                            'total_price': float(item.total_price)
                        }
                        for item in items_results
                    ],
                    'status_history': [
                        {
                            'status': status.status,
                            'notes': status.notes,
                            'created_at': status.created_at.isoformat() if status.created_at else None
                        }
                        for status in status_results
                    ]
                }
                
                return order_details
                
        except Exception as e:
            print(f"Error getting order details: {e}")
            return None
