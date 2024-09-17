# AI Web Scraper

This project is an AI-powered web scraper that utilizes Tor for anonymous scraping and Ollama (LLaMA 3.1) for generating insights from the scraped data. The user interface is built with Streamlit for ease of use.

## Table of Contents

- [AI Web Scraper](#ai-web-scraper)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Technologies Used](#technologies-used)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- Web scraping with IP rotation using Tor
- AI-powered analysis of scraped data using Ollama (LLaMA 3.1)
- User-friendly interface built with Streamlit
- Multiple scraping methods (requests, Selenium, Tor)
- User agent rotation for enhanced anonymity

## Technologies Used

- Python
- Streamlit
- Langchain
- Selenium
- BeautifulSoup4
- Tor
- Ollama (LLaMA 3.1)
- WSL (Windows Subsystem for Linux)

## Prerequisites

- Python 3.7+
- WSL (for running Tor and Ollama)
- Tor
- Ollama

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/AI.git
   cd AI/AI_WebScraper
   ```

2. Install the required Python packages:

   ```
   pip install -r requirements.txt
   ```

3. Install Tor (in WSL):

   ```
   sudo apt install tor
   ```

4. Install Ollama:

   ```
   curl -fsSL https://ollama.com/install.sh | sh
   ```

## Usage

1. Start the Tor service:

   ```
   sudo service tor start
   ```

2. Verify Tor is running:

   ```
   curl --socks5-hostname localhost:9050 http://check.torproject.org
   ```

3. Start the Ollama server:

   ```
   ollama serve
   ```

4. Run the Streamlit app:

   ```
   streamlit run main.py
   ```

5. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Project Structure

- `main.py`: The main Streamlit application
- `scrape.py`: Contains the web scraping logic
- `parse.py`: Handles parsing and AI analysis of scraped data
- `requirements.txt`: List of Python dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
