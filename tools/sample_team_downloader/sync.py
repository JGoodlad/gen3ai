import requests
from bs4 import BeautifulSoup
import os
import re
import time

SMOGON_URL = "https://www.smogon.com/forums/threads/adv-ou-sample-teams.3687813/"
OUTPUT_DIR = "data/teams/sample"
METADATA_PATH = os.path.join(OUTPUT_DIR, "teams.json")

def extract_metadata(first_post):
    """Extracts team metadata from the Smogon post using an ordered zip strategy."""
    # Find all unique pokepast.es links in order
    links = []
    seen = set()
    for a in first_post.find_all('a', href=re.compile(r'pokepast\.es/')):
        url = a.get('href').rstrip('/')
        if url not in seen:
            links.append(url)
            seen.add(url)
    
    # Find all team title strings in order (lines containing " - by " or " – by ")
    title_nodes = [s for s in first_post.find_all(string=re.compile(r' [–-] by ')) if len(s.strip()) < 100]
    
    if len(links) != len(title_nodes):
        print(f"Warning: Link count ({len(links)}) does not match title count ({len(title_nodes)})")
    
    teams = []
    for i in range(min(len(links), len(title_nodes))):
        url = links[i]
        node = title_nodes[i]
        full_text = node.strip()
        
        # Category: Find the last bold header before this node
        category = ""
        prev_bold = node.find_previous(['b', 'strong'])
        if prev_bold: category = prev_bold.get_text().strip()
        
        # Name & Author
        parts = re.split(r' [–-] by | by ', full_text, flags=re.IGNORECASE)
        name = parts[0].strip()
        name = re.sub(r'[ –-]+(by)?$', '', name).strip()
        author = parts[1].strip() if len(parts) > 1 else ""
        
        # If author is empty, check next sibling (might be a link)
        if not author:
            next_s = node.next_sibling
            if next_s: author = next_s.get_text().strip()

        teams.append({
            "url": url,
            "name": name,
            "author": author,
            "category": category if len(category) < 50 else ""
        })
    return teams

def fetch_and_save_team(meta):
    """Downloads a single team and saves it to a file."""
    url = meta["url"]
    raw_url = f"{url}/raw"
    paste_id = url.split('/')[-1]
    name = meta["name"] or f"Team {paste_id}"
    filename = f"{paste_id}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    print(f"Syncing: [{meta['category']}] {name} by {meta['author']} ({url})")
    try:
        res = requests.get(raw_url, timeout=10)
        if res.status_code == 200:
            with open(filepath, 'w') as f: f.write(res.text)
            return {
                **meta, 
                "id": paste_id, 
                "file": f"teams/sample/{filename}",
                "source": url
            }
    except Exception as e:
        print(f" Error {url}: {e}")
    return None

def sync_teams():
    """Main entry point for syncing Smogon sample teams."""
    print(f"Fetching Smogon thread: {SMOGON_URL}")
    response = requests.get(SMOGON_URL)
    if response.status_code != 200:
        print(f"Failed to fetch Smogon thread: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'lxml')
    first_post = soup.select_one('.message-inner .bbWrapper')
    if not first_post:
        print("Could not find the first post content.")
        return

    teams_to_sync = extract_metadata(first_post)
    print(f"Found {len(teams_to_sync)} unique teams to sync.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        teams_metadata = [r for r in executor.map(fetch_and_save_team, teams_to_sync) if r]

    # Save the index metadata
    import json
    with open(METADATA_PATH, 'w') as f:
        json.dump(teams_metadata, f, indent=2)

    print(f"Successfully synced {len(teams_metadata)} teams to {OUTPUT_DIR}")
    print(f"Metadata index saved to {METADATA_PATH}")

if __name__ == "__main__":
    sync_teams()
