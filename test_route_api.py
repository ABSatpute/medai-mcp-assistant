import requests
import json

def test_route_api():
    print("=== TESTING ROUTE API ===")
    
    # Test with sample coordinates (Pune area)
    user_lat, user_lon = 18.566039, 73.766370
    store_lat, store_lon = 18.570000, 73.770000
    
    # Test 1: Direct OSRM API call
    print("\n1. Testing OSRM API directly:")
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{user_lon},{user_lat};{store_lon},{store_lat}?overview=full&geometries=geojson"
    
    try:
        response = requests.get(osrm_url, timeout=10)
        print(f"OSRM Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('routes'):
                route = data['routes'][0]
                print(f"Route found: {route['distance']/1000:.1f}km, {route['duration']/60:.0f}min")
                print(f"Coordinates: {len(route['geometry']['coordinates'])} points")
            else:
                print("No routes in response")
        else:
            print(f"OSRM API failed: {response.text}")
    except Exception as e:
        print(f"OSRM API error: {e}")
    
    # Test 2: Your Flask route API
    print("\n2. Testing your Flask route API:")
    flask_url = f"http://localhost:5000/api/route/{user_lat}/{user_lon}/{store_lat}/{store_lon}"
    
    try:
        response = requests.get(flask_url, timeout=10)
        print(f"Flask API Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"Flask route works: {data['distance']}km, {data['duration']}min")
                print(f"Route coordinates: {len(data['route'])} points")
            else:
                print(f"Flask route failed: {data.get('error')}")
        else:
            print(f"Flask API failed: {response.text}")
    except Exception as e:
        print(f"Flask API error: {e}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    test_route_api()
