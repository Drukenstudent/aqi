import requests
import json
import sys  

# CONFIG
API_TOKEN = "c95bd6ad019cee15267a1a4dda6c9e322792598a"
OUTPUT_FILE = 'latest_data.py'
DEFAULT_CITY = 'hanoi'

def fetch_and_convert(city_name):
    # Sanitize input slightly (strip quotes if user added them)
    city_name = city_name.strip("'\"")
    
    url = f"https://api.waqi.info/feed/{city_name}/?token={API_TOKEN}"
    
    print(f"Fetching live data for: {city_name} ...")
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return

    # Check if API returned an error (e.g., Unknown city)
    if data.get('status') != 'ok':
        print(f"❌ API Error: {data.get('data')}")
        print("   (Check your city name spelling!)")
        return

    # Write to a Python file
    file_content = f"""# Auto-generated live data
# City: {city_name}
# Source: WAQI API

# Raw dictionary
aqi_snapshot = {json.dumps(data, indent=4)}

def get_pm25():
    '''Returns current PM2.5 or None if missing'''
    try:
        return aqi_snapshot['data']['iaqi']['pm25']['v']
    except (KeyError, TypeError):
        return None

def get_all_pollutants():
    '''Returns a dictionary of all available pollutants'''
    try:
        iaqi = aqi_snapshot['data']['iaqi']
        return {{k: v['v'] for k, v in iaqi.items()}}
    except (KeyError, TypeError):
        return {{}}
"""

    with open(OUTPUT_FILE, 'w') as f:
        f.write(file_content)
    
    print(f"✅ Generated '{OUTPUT_FILE}' with live data for {city_name}.")

if __name__ == "__main__":
    # Logic: Check if user provided an argument in the command line
    if len(sys.argv) > 1:
        # User typed: python convert_to_py.py "singapore/south"
        # sys.argv[0] is the script name, sys.argv[1] is the first argument
        target_city = sys.argv[1]
    else:
        # User typed: python convert_to_py.py
        print(f"ℹ️  No city specified. Using default: {DEFAULT_CITY}")
        print("   Tip: Run 'python convert_to_py.py [city_name]' to choose.")
        target_city = DEFAULT_CITY
        
    fetch_and_convert(target_city)
