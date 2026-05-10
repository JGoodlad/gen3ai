import requests
from bs4 import BeautifulSoup
import os
import re
import time

SMOGON_URL = "https://www.smogon.com/forums/threads/adv-ou-sample-teams.3687813/"
OUTPUT_DIR = "data/teams"

def sync_teams():
    print(f"Fetching Smogon thread: {SMOGON_URL}")
    response = requests.get(SMOGON_URL)
    if response.status_code != 200:
        print(f"Failed to fetch Smogon thread: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'lxml')
    
    # The first post is typically the first message-inner or has id="js-post-8918736"
    # But usually just taking the first .bbWrapper inside the first .message-inner works.
    first_post = soup.select_one('.message-inner .bbWrapper')
    if not first_post:
        print("Could not find the first post content.")
        return

    # Find all links to pokepast.es
    links = first_post.find_all('a', href=re.compile(r'pokepast\.es/'))
    print(f"Found {len(links)} potential team links.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    synced_count = 0
    teams_metadata = []

    for link in links:
        url = link.get('href')
        if not url: continue
        
        if not url.endswith('/raw'):
            raw_url = f"{url.rstrip('/')}/raw"
        else:
            raw_url = url
            url = url.replace('/raw', '')

        # Try to determine a name for the team
        name = link.get_text().strip()
        if "pokepast.es" in name.lower() or not name:
            bold_tag = link.find_previous('b') or link.find_previous('strong')
            if bold_tag:
                name = bold_tag.get_text().strip()
            else:
                prev_text = link.previous_sibling
                if prev_text and isinstance(prev_text, str):
                    name = prev_text.strip().rstrip(':').strip()
        
        if not name or "pokepast.es" in name.lower():
            parent_text = link.parent.get_text().split('\n')[0].strip()
            if len(parent_text) < 100:
                name = parent_text.split(': http')[0].strip()

        # Use the PokePaste ID as the filename to ensure uniqueness
        paste_id = url.split('/')[-1]
        filename = f"{paste_id}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"Syncing: {name} ({url}) -> {filename}")
        
        try:
            team_res = requests.get(raw_url)
            if team_res.status_code == 200:
                with open(filepath, 'w') as f:
                    f.write(team_res.text)
                
                teams_metadata.append({
                    "id": paste_id,
                    "name": name,
                    "url": url,
                    "file": f"teams/{filename}"
                })
                synced_count += 1
            else:
                print(f"  Failed to fetch raw team from {raw_url}: {team_res.status_code}")
        except Exception as e:
            print(f"  Error syncing {url}: {e}")
        
        time.sleep(1)

    # Save the index metadata
    import json
    metadata_path = "data/teams/teams.json"
    with open(metadata_path, 'w') as f:
        json.dump(teams_metadata, f, indent=2)

    print(f"Successfully synced {synced_count} teams to {OUTPUT_DIR}")
    print(f"Metadata index saved to {metadata_path}")

if __name__ == "__main__":
    sync_teams()
