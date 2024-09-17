# AI Web Scraper

This project is a web scraping tool with a user-friendly interface built using [Streamlit](https://streamlit.io/), which helps scrape data from web pages without getting blocked by IP bans. It utilizes the [Tor](https://www.torproject.org/) network for anonymous requests, and [Ollama Model 3.1](https://ollama.com/) to process and generate outputs based on the scraped content.

## Features
- **Web Scraping with Tor**: Scrapes data through Tor to avoid IP bans.
- **Streamlit Interface**: Simple and interactive interface to initiate scraping.
- **Ollama Model Integration**: Uses the AI model to generate outcomes based on the user's prompts.
- **Rotating User Agents**: Uses `fake-useragent` to add randomness in requests to avoid bot detection.
- **Flexible Scraping Methods**: Combines requests, Selenium, and Tor-based scraping for better results.

## Installation

### Prerequisites
- **Python**: Ensure Python is installed (version 3.8 or higher recommended).
- **Tor**: Tor should be installed and running for anonymous scraping.
- **Ollama Model**: Ollama's Llama 3.1 model must be installed for AI-based data parsing.

### Installing Requirements
Install the necessary dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt


