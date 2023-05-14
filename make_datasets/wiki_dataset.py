import json
import bz2
import re
from tqdm import tqdm
import urllib.parse
import urllib.request

# Set the path to the Wikipedia dump file
dump_path = "data/enwiki-latest-pages-articles.xml.bz2"

# Set the path to the output file
output_path = "data/wiki.jsonl"

# Define a regular expression to match English WikiLinks
wiki_link_regex = re.compile(r"\[\[en:([^\[\]]+)\]\]")


# Define a function to fetch the definition from Wikidata
def get_definition(entity_id):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            definition = data["entities"][entity_id]["descriptions"]["en"]["value"]
            return definition
    except:
        return ""


# Open the output file for writing
with open(output_path, "w", encoding="utf-8") as output_file:
    # Open the Wikipedia dump file for reading
    with bz2.open(dump_path, "rt", encoding="utf-8") as input_file:
        for line in tqdm(input_file):
            # Parse the line as a JSON object
            try:
                article = json.loads(line)
            except ValueError:
                continue
            # Extract the title and text of the article
            title = article["title"]
            text = article["text"]
            # Find all English WikiLinks in the text
            links = wiki_link_regex.findall(text)
            # Loop over the links
            for link in links:
                # Split the link into the page title and the anchor text
                parts = link.split("|")
                if len(parts) == 1:
                    anchor_text = parts[0].replace("_", " ")
                    page_title = anchor_text
                else:
                    anchor_text = parts[0].replace("_", " ")
                    page_title = parts[1].replace("_", " ")
                # Check if the page title is a valid entity ID
                if page_title.startswith("Q"):
                    # Get the definition from Wikidata
                    definition = get_definition(page_title)
                    # Find 5 contexts in which the entity is used
                    contexts = []
                    for match in re.finditer(re.escape(anchor_text), text):
                        start_pos = match.start()
                        end_pos = match.end()
                        context = text[max(0, start_pos - 50):min(len(text), end_pos + 50)]
                        context = context.replace("\n", " ").strip()
                        if context:
                            contexts.append(context)
                            if len(contexts) == 5:
                                break
                    # Write the data to the output file
                    data = {"entity": page_title, "definition": definition, "anchor_text": anchor_text,
                            "contexts": contexts}
                    output_file.write(json.dumps(data, ensure_ascii=False) + "\n")
