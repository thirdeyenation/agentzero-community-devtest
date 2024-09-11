import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from agent_zero.config import DATA_SOURCES

class DataCollector:
    def __init__(self):
        self.sources = DATA_SOURCES

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch content from a given URL."""
        async with session.get(url) as response:
            return await response.text()

    async def parse(self, html: str) -> List[Dict[str, str]]:
        """Parse HTML content and extract relevant information."""
        soup = BeautifulSoup(html, 'html.parser')
        data = []
        for article in soup.find_all('article'):
            title = article.find('h2')
            content = article.find('div', class_='content')
            if title and content:
                data.append({
                    "title": title.text.strip(),
                    "content": content.text.strip()
                })
        return data

    async def collect(self) -> List[Dict[str, Any]]:
        """Collect data from all sources."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch(session, source) for source in self.sources]
            html_contents = await asyncio.gather(*tasks)
        
        all_data = []
        for html in html_contents:
            all_data.extend(await self.parse(html))
        return all_data

    async def run(self) -> List[Dict[str, Any]]:
        """Main method to run the data collection process."""
        try:
            return await self.collect()
        except Exception as e:
            print(f"Error in data collection: {str(e)}")
            return []