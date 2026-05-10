
import json
import re

log_path = "/Users/goodlad/.gemini/antigravity/brain/b2258097-9733-46c0-8098-a6737f81d22e/.system_generated/logs/overview.txt"

with open(log_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("source") == "STDOUT":
                content = data.get("content", "")
                if "Win rate vs" in content:
                    print(content.strip())
        except:
            pass
