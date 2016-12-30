import requests
import time
import schedule

def request_for_data():
    params = {
        'range': '100',
        'interval': '1m',
        'includePrePost': False,
        'includeTimestamps': True,
        'indicators': "quote"
    }
    while True:
        print("\n\n\n\n Let\'s get the party started!", time.time())
        r = requests.get('https://query2.finance.yahoo.com/v7/finance/chart/^GSPC', params)
        print("\n\n\n\n This stuff works, dude.", time.time())
        
