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

        # Try to determine a name and category for the team
        name = ""
        category = ""
        author = ""
        ignored_words = ["description", "click", "paste", "link", "hide", "spikes", "here"]
        
        # Strategy 1: Iterate backwards to find name and category
        # We look for text nodes that look like names (often contain "by" or are longer than 5 chars)
        for s in link.previous_siblings:
            text = ""
            is_bold = False
            if isinstance(s, str):
                text = s.strip()
            elif s.name in ['b', 'strong']:
                text = s.get_text().strip()
                is_bold = True
            
            # Clean up the text
            text = text.rstrip(':').rstrip('-').rstrip('—').strip()
            if not text or len(text) <= 2 or any(w in text.lower() for w in ignored_words):
                continue
                
            if is_bold and not category:
                category = text
            elif not name:
                # Check if this text contains an author "by [Name]"
                if " by" in text.lower() or " – by" in text.lower():
                    # Extract name and author if possible
                    parts = re.split(r' [–-] by ', text, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        name = parts[0].strip()
                        author = parts[1].strip()
                    else:
                        name = text.rstrip('by').rstrip('–').strip()
                        # Look at the next sibling for the author (might be a link)
                        next_s = s.next_sibling
                        if next_s:
                            author = next_s.get_text().strip()
                else:
                    name = text
            
            # If we found both, we can stop
            if name and category:
                break

        # Strategy 2: If name is still just the category or empty, check link text
        if not name or name.lower() == category.lower():
            link_text = link.get_text().strip()
            if link_text and "pokepast.es" not in link_text.lower():
                name = link_text

        # Final fallback to unique identifier if everything fails
        paste_id = url.split('/')[-1]
        if not name or name.lower() == category.lower():
            name = f"Team {paste_id}"
        
        # Final name cleanup
        name = name.strip().rstrip('–').rstrip('-').strip()
        
        filename = f"{paste_id}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"Syncing: [{category}] {name} ({url}) -> {filename}")
        
        try:
            team_res = requests.get(raw_url)
            if team_res.status_code == 200:
                with open(filepath, 'w') as f:
                    f.write(team_res.text)
                
                teams_metadata.append({
                    "id": paste_id,
                    "name": name,
                    "author": author,
                    "category": category,
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
