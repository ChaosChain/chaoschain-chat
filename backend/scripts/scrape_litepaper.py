import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path
import time
import re

def clean_text(text):
    """Clean the scraped text by removing extra whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def scrape_page(url):
    """Scrape content from a single page."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the main content area
        main_content = soup.find('main')
        if not main_content:
            return None
        
        # Extract title
        title = soup.find('h1')
        title_text = clean_text(title.text) if title else "Untitled"
        
        # Extract content
        content = []
        for element in main_content.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'li']):
            if element.name in ['h1', 'h2', 'h3']:
                content.append(f"\n{'#' * int(element.name[1])} {clean_text(element.text)}\n")
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    content.append(f"- {clean_text(li.text)}")
            else:
                content.append(clean_text(element.text))
        
        return {
            'title': title_text,
            'content': '\n\n'.join(content)
        }
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

def scrape_litepaper():
    """Scrape the entire litepaper and save as markdown files."""
    base_url = "https://chaoschain-lite-paper.vercel.app"
    
    # Define the sections to scrape
    sections = [
        "/introduction/vision",
        "/introduction/overview",
        "/introduction/compute-governance-thesis",
        "/introduction/self-evolving-chain",
        "/technical-architecture/overview",
        "/technical-architecture/node-architecture",
        "/technical-architecture/chain-identity-system",
        "/l1-integration/overview",
        "/l1-integration/chain-registry-contract",
        "/l1-integration/bridge-contract",
        "/node-modifications/block-proposal-and-voting",
        "/node-modifications/horizontal-scalability",
        "/node-modifications/identity-module",
        "/node-modifications/mempool-flexibility",
        "/chain-management/overview",
        "/chain-management/minimal-bootstrap-requirements",
        "/chain-management/chain-creation",
        "/agent-ecosystem/overview",
        "/agent-ecosystem/agent-registry",
        "/agent-ecosystem/agent-development",
        "/academic-research/overview",
        "/governance-revolution/overview",
        "/governance-revolution/current-problems",
        "/governance-revolution/agentic-governance",
        "/governance-revolution/governance-as-compute",
        "/governance-revolution/proof-of-humanity",
        "/compute-wars-thesis/security-research",
        "/compute-wars-thesis/societal-implications",
        "/compute-wars-thesis/technical-research",
        "/roadmap/overview",
        "/developer-resources/overview",
        "/developer-resources/agent-development",
        "/developer-resources/chain-creation",
        "/api-reference/overview",
        "/api-reference/common-api-details",
        "/api-reference/agent-registry-api",
        "/api-reference/identity-api",
        "/api-reference/chains-api",
        "/api-reference/agent-consensus-api"
    ]
    
    # Create output directory
    output_dir = Path("../../src")
    output_dir.mkdir(exist_ok=True)
    
    # Scrape each section
    for section in sections:
        url = f"{base_url}{section}"
        print(f"Scraping {url}...")
        
        content = scrape_page(url)
        if content:
            # Create filename from section path
            filename = section.strip('/').replace('/', '_') + '.md'
            filepath = output_dir / filename
            
            # Write content to markdown file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {content['title']}\n\n")
                f.write(content['content'])
            
            print(f"Saved {filename}")
        
        # Be nice to the server
        time.sleep(1)

if __name__ == "__main__":
    scrape_litepaper() 