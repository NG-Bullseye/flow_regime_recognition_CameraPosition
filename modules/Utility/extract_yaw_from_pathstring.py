import re

def extract_value(folder_name):
    match = re.search(r'yaw_(-?\d+(\.\d+)?)', folder_name)
    return float(match.group(1)) if match else None