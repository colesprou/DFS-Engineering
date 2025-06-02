# DFS-Engineering
# Daily Fantasy Sports and Sportsbook Data Engineering
## Overview
- This project is a robust Python Flask application designed for a high stakes DFS player. It is tailored to adeptly handle CSV file uploads, perform data processing and cleaning, and prepare model-ready data for immediate download to the user's device. All major sports are supported on this app.

- Also, I have added an enpoint /MLBOptimizer that gives the user a optimized MLB DFS lineup directly down to your device when you upload DFS lineups from DraftKings.

## Key Features
- CSV File Handling: Upload, process, and clean CSV files to prepare them for data modeling.
- Instant Download: Model-ready data is made available for immediate download to the user's device, ensuring efficiency in data handling.
- Docker Integration: Utilizes Docker for efficient containerization, which guarantees smooth deployment on Azure's cloud platform.
- Data Extraction: Implements APIs and web scraping techniques to extract and refine data from RotoGrinders, FanDuel, and DraftKings. This is essential for generating accurate projections in the context of sports analytics.
## Technologies Used
- Python Flask: For building the web application.
- Docker: To containerize the application, ensuring consistency across various development and deployment environments.
- Microsoft Azure: Cloud platform used for deploying the containerized application.
- Data Extraction Tools: APIs and web scraping methods for data collection from specified sources.
- Data Storing: Storing historical expected fantasy points in a json file in Azure Blob Storage.
