#!/usr/bin/env python
# coding: utf-8

# In[48]:


from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup


import numpy as np
import requests
from datetime import datetime
from OddsJamClient import OddsJamClient


import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the environment variable
api_key = os.environ.get('ODDSJAM_API_KEY')


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
API_KEY = os.environ.get('ODDSJAM_API_KEY')


def transform_odds(odds):
    """Transforms DraftKings odds to a custom format."""
    if pd.isna(odds):  # Handle NaN values by returning NaN
        return odds
    
    if odds == 0:  # Return 0 as is
        return pd.NA

    if odds >= 100:
        return odds
    elif -195 <= odds  <= -105:
        return 200 + odds  # This will convert, for example, -105 to 4895
    elif odds <= -200:
        return 10200 + (-odds - 200)
    else:
        return odds

def get_todays_game_ids():
    # Initialize the client (assuming you have a client setup)
    Client = OddsJamClient(os.environ.get('ODDSJAM_API_KEY'))
    Client.UseV2()
    
    # Get games for the league
    GamesResponse = Client.GetGames(league='mlb')
    Games = GamesResponse.Games
    
    # Filter games based on today's date
    games_data = [{'game_id': game.id, 'start_date': game.start_date} for game in Games]
    games_df = pd.DataFrame(games_data)
    games_df['start_date'] = pd.to_datetime(games_df['start_date'])
    today = datetime.now().date()
    todays_games = games_df[games_df['start_date'].dt.date == today]
    
    return todays_games['game_id'].tolist()

def fetch_game_data(game_ids):
    desired_markets = [
        "Player Home Runs",
        "Player RBIs",
        "Player Runs",
        "Player Stolen Bases",
        "Player Singles",
        "Player Doubles",
        "Player Triples",
        "Player Batting Walks"
    ]

    URL = "https://api-external.oddsjam.com/api/v2/game-odds"
    API_KEY = os.environ.get('ODDSJAM_API_KEY')  # Fetching the API key from environment variables

    # Break game IDs into chunks of 5
    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]

    all_data = []

    # For each chunk, fetch odds
    for chunk in game_id_chunks:
        params = {
            'key': API_KEY,
            'sportsbook': ['DraftKings'],
            'game_id': chunk,
            'market_name': desired_markets
        }

        response = requests.get(URL, params=params)

        if response.status_code == 200:
            all_data.extend(response.json()['data'])
        else:
            print(f"Error {response.status_code}: {response.text}")

    rows = []
    #print(all_data)
    # Iterate through all games in the aggregated response data
    for game_data in all_data:
        odds_list = game_data['odds']
        home_team = game_data['home_team']
        away_team = game_data['away_team']
        
        for item in odds_list:
            
            market_name = item['market_name']
            
            if market_name not in desired_markets:
                continue
            
            player_name = item['selection']
            column_name = f"{item['market_name']} {item['selection_line']} {item['selection_points']}"
            #print(player_name)
            if player_name == home_team:
                team = home_team
            elif player_name == away_team:
                team = away_team
            else:
                team = 'Unknown'  # Default value

            row = {
                'Player Name': player_name,
                'Team': team,
                column_name: item['price']
            }
            rows.append(row)
            
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    #print(df)
    # Aggregate rows by player name
    df = df.groupby(['Player Name', 'Team']).sum().reset_index()

    df['Player Name'] = df['Player Name'].str.replace('[^A-Za-z\s]', '', regex=True)
    df['Player Name'] = df['Player Name'].apply(lambda x: ' '.join(x.split(' ')[:2]))

    return df
### Roto Grinders

def roto_grinders():
    URL = 'https://rotogrinders.com/lineups/mlb?site=fanduel'

    # Make a request to the webpage
    fresponse = requests.get(URL)
    fresponse.raise_for_status()  # Raise exception if the request wasn't successful

    fsoup = BeautifulSoup(fresponse.content, 'html.parser')

    # Extract data for each game
    fall_data = []

    fgames = fsoup.find_all('div', class_='blk game')
    for game in fgames:
        fplayers = game.find_all('li', class_='player')
        for player in fplayers:
            fpname = player.find('span', class_='pname').get_text(strip=True)
            fsalary = player.find('span', class_='salary').get_text(strip=True)
            fall_data.append((fpname, fsalary))

    # Convert data to dataframe
    fdf = pd.DataFrame(fall_data, columns=['Player', 'Salary'])

    fdf['Sportsbook'] = 'FanDuel'
    return fdf

def projected_df(file_path):
    todays_ids = get_todays_game_ids()

    df = fetch_game_data(todays_ids)
    fdf = roto_grinders()
    ### Projected DF
    fdDF = pd.read_csv(file_path)
    fdDF['Nickname'] = fdDF['Nickname'].str.replace('[^A-Za-z\s]', '', regex=True)
    fdDF['Nickname'] = fdDF['Nickname'].apply(lambda x: ' '.join(x.split(' ')[:2]))
    projDF = fdDF.merge(fdf,left_on = 'Nickname',right_on= 'Player',how = 'left') 

    projDF = projDF[projDF['Player'].notna()]

    projDF = projDF[['Nickname','Team','Salary_x','Game','Roster Position','Injury Indicator','Batting Order',]]
    projDF = pd.merge(projDF,df,left_on='Nickname',right_on='Player Name',how='left')
    #print(projDF)
    filtered_cols = [col for col in projDF.columns if "under" not in col]

    projDF = projDF[filtered_cols]
    projDF.rename({'Nickname':'name','Team_x':'team','Salary_x':'Cost','Player Batting Walks over 0.5':'walk','Player Doubles over 0.5':'double',
                   'Player Singles over 0.5':'single','Player Stolen Bases over 0.5':'sb','Player Runs over 0.5':'run',
                  'Player Home Runs over 0.5':'hr','Player RBIs over 0.5':'rbi','Player Triples over 0.5':'triple'},axis=1,inplace=True)
    columns_to_transform = ['walk', 'double', 'single', 'sb', 'hr', 'run', 'rbi', 'triple']

    for col in columns_to_transform:
        for idx, value in projDF[col].items():
            projDF.at[idx, col] = transform_odds(value)


    projDF.rename({'name':'Name','Cost':'cost'},axis=1,inplace=True)
    finalProj = projDF[['team','Name','Roster Position','cost','hr','rbi','run','sb','single','double','triple','walk',
                                  'Injury Indicator','Batting Order']]
    return finalProj
def roto_grinders_draftkings():
    URL = 'https://rotogrinders.com/lineups/mlb?site=draftkings'

    # Make a request to the webpage
    dresponse = requests.get(URL)
    dresponse.raise_for_status()# Raise exception if the request wasn't successful

    dsoup = BeautifulSoup(dresponse.content, 'html.parser')

    # Extract data for each game
    dall_data = []

    dgames = dsoup.find_all('div', class_='blk game')
    for game in dgames:
        dplayers = game.find_all('li', class_='player')
        for player in dplayers:
            dpname = player.find('span', class_='pname').get_text(strip=True)
            dsalary = player.find('span', class_='salary').get_text(strip=True)
            dall_data.append((dpname, dsalary))

    # Convert data to dataframe
    ddf = pd.DataFrame(dall_data, columns=['Player', 'Salary'])
    ddf['Sportsbook'] = 'DraftKings'
    return ddf

def draftkings_projected_df(file_path):
    todays_ids = get_todays_game_ids()

    df = fetch_game_data(todays_ids)
    ddf = roto_grinders_draftkings()
    dkDF = pd.read_csv(file_path)
    dkDF['Name'] = dkDF['Name'].str.replace('[^A-Za-z\s]', '', regex=True)
    dkDF['Name'] = dkDF['Name'].apply(lambda x: ' '.join(x.split(' ')[:2]))


    projDF_dk = dkDF.merge(ddf,left_on = 'Name',right_on= 'Player',how = 'left')


    projDF_dk = projDF_dk[projDF_dk['Player'].notna()]


    projDF_dk = projDF_dk[['Name','Roster Position','Salary_x','TeamAbbrev']]

    projDF_dk.rename({'Salary_x':'cost','TeamAbbrev':'Team'},axis=1,inplace=True)
    #print(df.head())
    DKDF = pd.merge(projDF_dk,df,left_on='Name',right_on='Player Name',how='left')
    #print(DKDF.columns)
    #filtered_cols = [col for col in DKDF.columns if "under" not in col]
    #DKDF = DKDF[filtered_cols]
    DKDF = DKDF[['Player Name', 'Roster Position', 'cost', 'Team_x', 'Player Name',
       'Player Batting Walks over 0.5','Player Triples over 0.5','Player Doubles over 0.5','Player Runs over 0.5',
        'Player Singles over 0.5','Player Home Runs over 0.5','Player RBIs over 0.5','Player Stolen Bases over 0.5']]
    

    DKDF
    DKDF.rename({'Player Name':'name','Team_x':'team','Salary_x':'Cost','Player Batting Walks over 0.5':'walk','Player Doubles over 0.5':'double',
                   'Player Singles over 0.5':'single','Player Stolen Bases over 0.5':'sb','Player Runs over 0.5':'run',
                  'Player Home Runs over 0.5':'hr','Player RBIs over 0.5':'rbi','Player Triples over 0.5':'triple'},axis=1,inplace=True)
    columns_to_transform = ['walk', 'double', 'single', 'sb', 'hr', 'run', 'rbi', 'triple']
    for col in columns_to_transform:
        for idx, value in DKDF[col].items():
            DKDF.at[idx, col] = transform_odds(value)


    finalProj = DKDF[['team','name','Roster Position','cost','hr','rbi','run','sb','single','double','walk','triple']]
    finalProj =finalProj.loc[:, ~finalProj.columns.duplicated(keep='first')]

    return finalProj
#finalProj = draftkings_projected_df('/Users/colesprouse/')

@app.route('/')
def index():
    return render_template('updated_upload.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return 'No file uploaded!'
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = projected_df(filepath)

    output_filename = "processed_output.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/process_draftkings', methods=['POST'])
def process_file_draftkings():
    if 'file' not in request.files:
        return 'No file uploaded!'
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = draftkings_projected_df(filepath)  # Assuming you're processing the DraftKings file the same way

    output_filename = "processed_output_draftkings.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# In[30]:


#app.run(port=5001)  # Use a different port if 5001 is occupied
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


# In[ ]:




