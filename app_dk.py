#!/usr/bin/env python
# coding: utf-8

# In[48]:


from flask import Flask, render_template, request, send_from_directory
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import logging 
from bs4 import BeautifulSoup



import numpy as np
import requests
from datetime import datetime, timedelta, timezone, time
from OddsJamClient import OddsJamClient
import string

import os
from dotenv import load_dotenv
import json
from fuzzywuzzy import fuzz



# Load environment variables from .env file
load_dotenv()

# Access the environment variable
api_key = os.environ.get('ODDSJAM_API_KEY')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
API_KEY = os.environ.get('ODDSJAM_API_KEY')


def transform_odds(odds):
    if pd.isna(odds):  
        return odds
    
    if odds == 0:  
        return pd.NA

    if odds >= 100:
        return odds
    elif -195 <= odds  <= -105:
        return 200 + odds  
    elif odds <= -200:
        return 10200 + (-odds - 200)
    else:
        return odds
#os.environ.get('ODDSJAM_API_KEY')
def get_pitcher_data(game_ids):
    desired_markets = ['Player Strikeouts','Player Outs','Player Walks','Player Earned Runs',
                       'Player To Record Win','Player Hits Allowed']

    URL = "https://api-external.oddsjam.com/api/v2/game-odds"
    API_KEY = os.environ.get('ODDSJAM_API_KEY')

        # Break game IDs into chunks of 5
    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]

    all_data = []

        # For each chunk, fetch odds
    for chunk in game_id_chunks:
        params = {
                'key': API_KEY,
                'sportsbook': ['DraftKings'],
                'game_id': chunk,
                'market_name': desired_markets,
                'is_main':'true'
            }

        response = requests.get(URL, params=params)

        if response.status_code == 200:
            all_data.extend(response.json()['data'])
        else:
            print(f"Error {response.status_code}: {response.text}")

    # Clear previous data
    # Initialize a new dictionary to store the revised player data
    revised_player_data = {}

    # Loop through each game's data
    for game in all_data:
        if 'odds' in game and game['odds']:
            for odd in game['odds']:
                # Extract player name and handle "Yes" or "No" in win odds
                player_name = odd['selection']
                if odd['market'] == 'player_to_record_win':
                    player_name = player_name.replace(" Yes", "").replace(" No", "")

                # Initialize player entry if not present
                if player_name not in revised_player_data:
                    revised_player_data[player_name] = {
                        "Player Name": player_name,
                        "Hits Line": None,
                        "Hits Odds":None,
                        "Strikeout Line": None,
                        "Strikeout Odds": None,
                        "Outs Line": None,
                        "Outs Odds": None,
                        "To record a win odds": None,
                        "Walks Line": None,
                        "Walks Odds": None,
                        "Earned Run Line": None,
                        "Earned Run Odds": None
                    }

                # Update player data based on the odd's market
                if 'over' in odd['name'].lower():
                    if odd['market'] == 'player_hits_allowed':
                        revised_player_data[player_name]["Hits Line"] = odd.get('bet_points', None)
                        revised_player_data[player_name]["Hits Odds"] = odd.get('price', None)
                    if odd['market'] == 'player_strikeouts':
                        revised_player_data[player_name]["Strikeout Line"] = odd.get('bet_points', None)
                        revised_player_data[player_name]["Strikeout Odds"] = odd.get('price', None)
                    elif odd['market'] == 'player_outs':
                        revised_player_data[player_name]["Outs Line"] = odd.get('bet_points', None)
                        revised_player_data[player_name]["Outs Odds"] = odd.get('price', None)
                    elif odd['market'] == 'player_walks':
                        revised_player_data[player_name]["Walks Line"] = odd.get('bet_points', None)
                        revised_player_data[player_name]["Walks Odds"] = odd.get('price', None)
                    elif odd['market'] == 'player_earned_runs':
                        revised_player_data[player_name]["Earned Run Line"] = odd.get('bet_points', None)
                        revised_player_data[player_name]["Earned Run Odds"] = odd.get('price', None)
                elif odd['market'] == 'player_to_record_win' and "yes" in odd['name'].lower():
                    revised_player_data[player_name]["To record a win odds"] = odd.get('price', None)

    # Convert the revised player data into a DataFrame
    revised_df = pd.DataFrame.from_dict(revised_player_data, orient='index').reset_index(drop=True)
    return revised_df
def get_todays_game_ids():
    
    Client = OddsJamClient(os.environ.get('ODDSJAM_API_KEY'))
    Client.UseV2()
    
    # Get games for the league
    GamesResponse = Client.GetGames(league='mlb')
    Games = GamesResponse.Games
    
    # Filter games based on today's date
    games_data = [{'game_id': game.id, 'start_date': game.start_date} for game in Games]
    games_df = pd.DataFrame(games_data)
    games_df['start_date'] = pd.to_datetime(games_df['start_date'])
    #today = datetime.now().date()
    #todays_games = games_df[games_df['start_date'].dt.date == today]
    
    return games_df['game_id'].tolist()

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
    API_KEY = os.environ.get('ODDSJAM_API_KEY')  

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
            
            if player_name == home_team:
                team = home_team
            elif player_name == away_team:
                team = away_team
            else:
                team = 'Unknown'  

            row = {
                'Player Name': player_name,
                'Team': team,
                column_name: item['price']
            }
            rows.append(row)
            
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    logging.info("Columns in the DataFrame: %s", df.columns)
    print(df.columns)
    # Aggregate rows by player name
    df = df.groupby(['Player Name', 'Team']).sum().reset_index()

    df['Player Name'] = df['Player Name'].str.replace('[^A-Za-z\s]', '', regex=True)
    df['Player Name'] = df['Player Name'].apply(lambda x: ' '.join(x.split(' ')[:2]))
    df.loc[df['Player Name'].str.contains('JiHwan Bae'), 'Player Name'] = df['Player Name'].str.replace('JiHwan Bae', 'Ji Hwan')
    return df

def projected_df(file_path):
    todays_ids = get_todays_game_ids()
    pdf = get_pitcher_data(todays_ids)
    df = fetch_game_data(todays_ids)
    df = pd.concat([df.set_index('Player Name'), pdf.set_index('Player Name')], axis=1, join='outer').reset_index()

    ### Projected DF
    fdDF = pd.read_csv(file_path)
    fdDF['Nickname'] = fdDF['Nickname'].str.replace('[^A-Za-z\s]', '', regex=True)
    fdDF['Nickname'] = fdDF['Nickname'].apply(lambda x: ' '.join(x.split(' ')[:2]))
    fdDF.loc[fdDF['Nickname'].str.contains('Palacios'), 'Nickname'] = fdDF['Nickname'].str.replace('Joshua', 'Josh')
    projDF = fdDF 
    #print(projDF)
    #projDF = projDF[projDF['Player'].notna()]

    projDF = projDF[['Nickname','Team','Salary','Game','Roster Position','Injury Indicator','Batting Order',]]
    projDF = pd.merge(projDF,df,left_on='Nickname',right_on='Player Name',how='left')
    #filtered_cols = [col for col in projDF.columns if "under" not in col]
    columns_to_include = ['Nickname','Team_x','Salary','Game','Roster Position','Injury Indicator','Batting Order',
                      'Player Batting Walks over 0.5','Player Doubles over 0.5','Player Runs over 0.5',
                      'Player Singles over 0.5','Player Home Runs over 0.5','Player RBIs over 0.5','Player Stolen Bases over 0.5',
                      'Hits Line','Hits Odds', 'Strikeout Line', 'Strikeout Odds', 'Outs Line', 'Outs Odds',
                      'To record a win odds', 'Walks Line', 'Walks Odds', 'Earned Run Line', 'Earned Run Odds']

    if 'Player Triples over 0.5' in df.columns:
        columns_to_include.append('Player Triples over 0.5')

    projDF = projDF[columns_to_include]
    projDF.rename({'Nickname':'name','Team_x':'team','Salary':'Cost','Player Batting Walks over 0.5':'walk','Player Doubles over 0.5':'double',
                   'Player Singles over 0.5':'single','Player Stolen Bases over 0.5':'sb','Player Runs over 0.5':'run',
                  'Player Home Runs over 0.5':'hr','Player RBIs over 0.5':'rbi','Player Triples over 0.5':'triple','Strikeout Line': 'so',
    'Strikeout Odds': 'so odds', 'Hits Line':'hits','Hits Odds':'hits odds',
    'Outs Line': 'outs',
    'Outs Odds': 'outs odds',
    'To record a win odds': 'win',
    'Walks Line': 'walks (p)',
    'Walks Odds': 'walks (p) odds',
    'Earned Run Line': 'er',
    'Earned Run Odds': 'er odds'},axis=1,inplace=True)
    columns_to_transform = ['walk', 'double', 'single', 'sb', 'hr', 'run', 'rbi', 'triple','hits odds',
 'so odds',
 'outs odds',
 'win',
 'walks (p) odds',
 'er odds']

    for col in columns_to_transform:
        if col in projDF.columns:
            for idx, value in projDF[col].items():
                projDF.at[idx, col] = transform_odds(value)


    projDF.rename({'name':'Name','Cost':'cost'},axis=1,inplace=True)
    final_columns = [col for col in ['team','Name','Roster Position','cost','hr','rbi',
                                     'run','sb','single','double','triple','walk',
                                 'Injury Indicator','Batting Order','hits','hits odds','so',
 'so odds',
 'outs',
 'outs odds',
 'win',
 'walks (p)',
 'walks (p) odds',
 'er',
 'er odds'] if col in projDF.columns]
    finalProj = projDF[final_columns]
    cols_to_check = ['hr', 'rbi', 'run', 'sb',
       'single', 'double', 'triple', 'walk','so','hits','hits odds',
 'so odds',
 'outs',
 'outs odds',
 'win',
 'walks (p)',
 'walks (p) odds',
 'er',
 'er odds']
    cols_to_check = [col for col in cols_to_check if col in finalProj.columns]
    finalProj.dropna(subset=cols_to_check,how='all',inplace=True)
    return finalProj
def mlb_fanduel_optimized(df):
    positions = ["P", "C-1B", "2B", "3B", "SS", "OF"]

    # Replace "C/1B" with "C-1B" to avoid issues with splitting
    df['Roster Position'] = df['Roster Position'].str.replace('C/1B', 'C-1B')

    for position in positions:
        df[position] = ''

    for index, row in df.iterrows():
        roster_positions = str(row['Roster Position']).split('/')
        for pos in roster_positions:
            pos = pos.strip()
            if pos in positions:
                df.at[index, pos] = pos

    df = df.drop(columns=['Roster Position'])

    df['blank1'] = np.nan
    df['blank2'] = np.nan
    df['blank3'] = np.nan
    df['blank4'] = np.nan
    df['blank5'] = np.nan
    df['blank6'] = np.nan

    new_column_order = [
        'team', 'Name', 'blank1', 'cost', 'P', 'C-1B', '2B', '3B', 'SS', 'OF', 'blank2', 'blank3', 
        'blank4', 'hr', 'rbi', 'run', 'sb', 'single', 'double', 'triple', 'walk', 'blank5', 'blank6', 
        'so', 'so odds', 'outs', 'outs odds', 'win', 'hits', 'hits odds', 'walks (p)', 'walks (p) odds', 
        'er', 'er odds'
    ]
    df = df[new_column_order]
    df.rename({'C-1B':'C/1B'},axis=1,inplace=True)
    df['C/1B'] = df['C/1B'].str.replace('C-1B', 'C/1B')
    return df
def draftkings_projected_df(file_path):
    todays_ids = get_todays_game_ids()
    pdf = get_pitcher_data(todays_ids)
    df = fetch_game_data(todays_ids)
    dkDF = pd.read_csv(file_path)
    df = pd.concat([df.set_index('Player Name'), pdf.set_index('Player Name')], axis=1, join='outer').reset_index()

    dkDF.loc[dkDF['Name'].str.contains('Palacios'), 'Name'] = dkDF['Name'].str.replace('Joshua', 'Josh')
    dkDF['Name'] = dkDF['Name'].str.replace('[^A-Za-z\s]', '', regex=True)
    dkDF['Name'] = dkDF['Name'].apply(lambda x: ' '.join(x.split(' ')[:2]))


    projDF_dk = dkDF[['Name','Roster Position','Salary','TeamAbbrev']]

    projDF_dk.rename({'Salary':'cost','TeamAbbrev':'Team'},axis=1,inplace=True)
    DKDF = pd.merge(projDF_dk,df,left_on='Name',right_on='Player Name',how='left')
    columns_to_include = ['Name', 'Roster Position', 'cost', 'Team_x',
                        'Player Batting Walks over 0.5','Player Doubles over 0.5','Player Runs over 0.5',
                        'Player Singles over 0.5','Player Home Runs over 0.5','Player RBIs over 0.5','Player Stolen Bases over 0.5',  "Hits Line","Hits Odds",  "Strikeout Line", 
    "Strikeout Odds", 
    "Outs Line", 
    "Outs Odds", 
    "To record a win odds", 
    "Walks Line", 
    "Walks Odds", 
    "Earned Run Line", 
    "Earned Run Odds"
]
    if 'Player Triples over 0.5' in DKDF.columns:
        columns_to_include.append('Player Triples over 0.5')
    DKDF = DKDF[columns_to_include]
    
    DKDF.rename({'Name':'name','Team_x':'team','cost':'Cost','Player Batting Walks over 0.5':'walk','Player Doubles over 0.5':'double',
                       'Player Singles over 0.5':'single','Player Stolen Bases over 0.5':'sb','Player Runs over 0.5':'run',
                      'Player Home Runs over 0.5':'hr','Player RBIs over 0.5':'rbi','Player Triples over 0.5':'triple','Hits Line':'hits','Hits Odds':'hits odds','Strikeout Line': 'so',
    'Strikeout Odds': 'so odds',
    'Outs Line': 'outs',
    'Outs Odds': 'outs odds',
    'To record a win odds': 'win',
    'Walks Line': 'walks (p)',
    'Walks Odds': 'walks (p) odds',
    'Earned Run Line': 'er',
    'Earned Run Odds': 'er odds'},axis=1,inplace=True)
    #print(DKDF)

    columns_to_transform = ['walk', 'double', 'single', 'sb', 'hr', 'run', 'rbi', 'triple','hits odds',
 'so odds',
 'outs odds',
 'win',
 'walks (p) odds',
 'er odds']

    for col in columns_to_transform:
        if col in DKDF.columns:
            for idx, value in DKDF[col].items():
                DKDF.at[idx, col] = transform_odds(value)


    DKDF.rename({'name':'Name','Cost':'cost'},axis=1,inplace=True)
    final_columns = [col for col in ['team','Name','Roster Position','cost','hr','rbi',
                                         'run','sb','single','double','triple','walk',
                                     'Injury Indicator','Batting Order','hits','hits odds','so','so odds',
 'outs',
 'outs odds',
 'win',
 'walks (p)',
 'walks (p) odds',
 'er',
 'er odds'] if col in DKDF.columns]
    finalProj = DKDF[final_columns]
    cols_to_check = ['hr', 'rbi', 'run', 'sb',
       'single', 'double', 'triple', 'walk','hits','hits odds','so','so odds',
 'outs',
 'outs odds',
 'win',
 'walks (p)',
 'walks (p) odds',
 'er',
 'er odds']
    cols_to_check = [col for col in cols_to_check if col in finalProj.columns]
    finalProj.dropna(subset=cols_to_check,how='all',inplace=True)
    return finalProj

def mlb_draftkings_optimized(df):
    positions = ["P", "C", "1B", "2B", "3B", "SS", "OF"]
    df = df[~df['Roster Position'].str.contains('CPT')]
    # Initialize the position columns with empty strings
    for position in positions:
        df[position] = ''

    # Fill the position columns based on the "Roster Position" values
    for index, row in df.iterrows():
        roster_positions = str(row['Roster Position']).split('/')
        if all(pos.strip() in ["UTIL", "CPT"] for pos in roster_positions):
            df.at[index, 'UTIL/CPT'] = '/'.join(roster_positions)
        else:
            for pos in roster_positions:
                pos = pos.strip()
                if pos in positions:
                    df.at[index, pos] = pos

    # Drop the original "Roster Position" column
    df = df.drop(columns=['Roster Position'])

    # Initialize the blank columns
    df['UTIL/CPT'] = np.nan
    df['blank2'] = np.nan
    df['blank3'] = np.nan
    df['blank4'] = np.nan
    df['blank5'] = np.nan
    df['blank6'] = np.nan

    # Define the new column order
    new_column_order = [
        'team', 'Name', 'UTIL/CPT', 'cost', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'blank2', 'blank3', 
        'blank4', 'hr', 'rbi', 'run', 'sb', 'single', 'double', 'triple', 'walk', 'blank5', 'blank6', 
        'so', 'so odds', 'outs', 'outs odds', 'win', 'hits', 'hits odds', 'walks (p)', 'walks (p) odds', 
        'er', 'er odds'
    ]

    # Reorder the columns
    df = df[new_column_order]

    return df
    

def get_nfl_game_ids():
    # Endpoint
    endpoint = "https://api-external.oddsjam.com/api/v2/games"

    # Current week date range
    today = datetime.now(timezone.utc)
    start_date = today.isoformat()


    # Calculate the upcoming Monday or today if it's Monday
    days_until_monday = (7 - today.weekday()) % 7
    monday = today + timedelta(days=days_until_monday)
    
    # Set time to 11:59 pm and adjust for CST (UTC-6)
    end_date_cst = (monday + timedelta(days=1)).replace(hour=5, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)
    # Parameters
    params = {
        "key": os.environ.get('ODDSJAM_API_KEY'),
        "sport": "football",
        "league": "NFL",
        "start_date_after": start_date,
        "start_date_before": end_date_cst.isoformat()
    }

    # Make the request
    response = requests.get(endpoint, params=params)

    # Check response
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    data = response.json()
    game_ids = [game['id'] for game in data['data']]
    return game_ids
def nfl_fetch_game_data(game_ids):
    desired_markets = [
        "Player Passing Touchdowns",
        "Player Interceptions",
        "Player Touchdowns",
        "Player Receiving Yards",
        "Player Passing Yards",
        "Player Rushing Yards",
        "Player Receptions",
        "Player Kicking Points"
    ]

    URL = "https://api-external.oddsjam.com/api/v2/game-odds"
    API_KEY = os.environ.get('ODDSJAM_API_KEY')  

    # Break game IDs into chunks of 5
    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]

    all_data = []

    # For each chunk, fetch odds
    for chunk in game_id_chunks:
        params = {
            'key': API_KEY,
            'sportsbook': ['DraftKings'],
            'game_id': chunk,
            'market_name': desired_markets,
            'is_main':'true'
        }

        response = requests.get(URL, params=params)

        if response.status_code == 200:
            all_data.extend(response.json()['data'])
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    json_data = all_data
    desired_columns = [
    "name",
    "# of pass tds",
    "odds on over pass tds",
    "odds on over interception",
    "odds on td scored",
    "receiving yds line",
    "passing yds line",
    "rush yds line",
    "number of pass receptions",
    "odds on pass receptions",
    "kicker points line",
    "odds on kicker points"
    ]

    df = pd.DataFrame(columns=desired_columns)
    df = pd.DataFrame(columns=desired_columns)

# Map values from the updated JSON data to the DataFrame using the corrected conditions
    for entry in json_data:
        for bet in entry['odds']:
            player_name = bet['selection']
            market_name = bet['market_name']
            selection_line = bet['selection_line']
            selection_points = bet['selection_points']
            price = bet['price']

            # Determine the appropriate column based on market name and selection line
            if "Player Passing Touchdowns" in market_name and selection_line == "over":

                column_name = "# of pass tds"
                value = selection_points
                # Check if player exists in the DataFrame
                if player_name in df['name'].values:
                    df.loc[df['name'] == player_name, column_name] = value
                    df.loc[df['name'] == player_name,"odds on over pass tds"]= price
                else:
                    # If player doesn't exist, add new player data
                    new_row = pd.DataFrame([{'name': player_name, column_name: value, "odds on over pass tds": price}])
                    df = pd.concat([df,new_row], ignore_index=True)
            
            elif "Player Interceptions" in market_name and selection_line == "over":
                print(f"Found Player Interceptions for {player_name} with selection line: {selection_line}")
                column_name = "odds on over interception"
                value = price
            elif market_name == "Player Passing Yards" and selection_line == "over":
                column_name = "passing yds line"
                value = selection_points
            elif market_name == "Player Touchdowns" and selection_line == "over" and selection_points == 0.5:
                column_name = 'odds on td scored' 
                value = price
            elif market_name == "Player Receiving Yards" and selection_line == "over":
                column_name = "receiving yds line"
                value = selection_points
            elif market_name == "Player Rushing Yards" and selection_line == "over":
                column_name = "rush yds line"
                value = selection_points 
            elif market_name == "Player Receptions" and selection_line == "over":
                column_name = "number of pass receptions"
                value = selection_points
                # Check if player exists in the DataFrame
                if player_name in df['name'].values:
                    df.loc[df['name'] == player_name, column_name] = value
                    df.loc[df['name'] == player_name, "odds on pass receptions"] = price
                else:
                    new_row = pd.DataFrame([{'name': player_name, column_name: value, "odds on pass receptions": price}])
                    df = pd.concat([df,new_row], ignore_index=True)
            elif market_name == "Player Kicking Points" and selection_line == "over":
                column_name = "kicker points line"
                value = selection_points
                # Check if player exists in the DataFrame
                if player_name in df['name'].values:
                    df.loc[df['name'] == player_name, column_name] = value
                    df.loc[df['name'] == player_name, "odds on kicker points"] = price
                else:
                    new_row = pd.DataFrame([{'name': player_name, column_name: value, "odds on kicker points": price}])
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                continue

            # Update the DataFrame with the extracted values (excluding the special cases)
            if market_name not in ["Player Passing Touchdowns", "Player Receptions", "Player Kicking Points"]:
                if player_name not in df['name'].values:
                    
                    new_row = pd.DataFrame({'name': [player_name], column_name: [value]})
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    df.loc[df['name'] == player_name, column_name] = value
    return df
def process_nfl_data(file_path):
    todays_ids = get_nfl_game_ids()
    Oddsdf = nfl_fetch_game_data(todays_ids)
    FDdf = pd.read_csv(file_path)
 
    FDdf = FDdf[['Nickname','Salary','Team','Roster Position']]
    
    Oddsdf['name'] = Oddsdf['name'].apply(lambda x: ''.join(ch for ch in x if ch not in string.punctuation))

    Oddsdf['name'] = Oddsdf['name'].apply(lambda x: x.split(' Defense')[0] if 'Defense' in x else x)


    Oddsdf['name'] = Oddsdf['name'].replace({
        'Chigoziem Okonkwo': 'Chig Okonkwo',
        'Scott Miller': 'Scotty Miller',
        'Gabriel Davis': 'Gabe Davis'
    })


    FDdf['Nickname'] = FDdf['Nickname'].apply(lambda x: ''.join(ch for ch in x if ch not in string.punctuation))

    Fdf = FDdf.merge(Oddsdf,left_on='Nickname',right_on='name',how='left')

    Fdf.columns

    columns_to_transform = ['odds on over pass tds','odds on over interception','odds on td scored','odds on pass receptions','odds on kicker points']
    for col in columns_to_transform:
            for idx, value in Fdf[col].items():
                Fdf.at[idx, col] = transform_odds(value)

    Fdf = Fdf[['Team','Nickname','Roster Position','Salary','# of pass tds', 'odds on over pass tds','passing yds line',
           'odds on over interception', 'odds on td scored', 'receiving yds line',
           'rush yds line', 'number of pass receptions', 'odds on pass receptions',
           'kicker points line', 'odds on kicker points']]

    Fdf.rename({'Team':'team','Nickname':'name','Roster Position':'position','Salary':'cost'},axis=1,inplace=True)
    
    return Fdf
def nfl_draftkings_process(file_path):
    todays_ids = get_nfl_game_ids()
    Oddsdf = nfl_fetch_game_data(todays_ids)
    DKdf = pd.read_csv(file_path)

    DKdf = DKdf[['TeamAbbrev','Name','Roster Position','Salary']]

    Oddsdf['name'] = Oddsdf['name'].apply(lambda x: ''.join(ch for ch in x if ch not in string.punctuation))

    Oddsdf['name'] = Oddsdf['name'].apply(lambda x: x.split(' Defense')[0] if 'Defense' in x else x)


    DKdf['Name'] = DKdf['Name'].apply(lambda x: ''.join(ch for ch in x if ch not in string.punctuation))

    nfl_team_map = {
        "Bears": "Chicago Bears", "Packers": "Green Bay Packers", "Patriots": "New England Patriots", "Rams": "Los Angeles Rams", "Cardinals": "Arizona Cardinals", "Falcons": "Atlanta Falcons", "Ravens": "Baltimore Ravens", "Bills": "Buffalo Bills", "Panthers": "Carolina Panthers", "Bengals": "Cincinnati Bengals", "Browns": "Cleveland Browns", "Cowboys": "Dallas Cowboys", "Broncos": "Denver Broncos", "Lions": "Detroit Lions", "Texans": "Houston Texans", "Colts": "Indianapolis Colts", "Jaguars": "Jacksonville Jaguars", "Chiefs": "Kansas City Chiefs", "Chargers": "Los Angeles Chargers", "Dolphins": "Miami Dolphins", "Vikings": "Minnesota Vikings", "Saints": "New Orleans Saints", "Giants": "New York Giants", "Jets": "New York Jets", "Raiders": "Las Vegas Raiders", "Eagles": "Philadelphia Eagles", "Steelers": "Pittsburgh Steelers", "49ers": "San Francisco 49ers","Seahawks": "Seattle Seahawks", "Buccaneers": "Tampa Bay Buccaneers", "Titans": "Tennessee Titans", "Commanders": "Washington Commanders"  # Note: As of my last update, the Washington team was still named the "Washington Football Team".
    }
    inverse_nfl_team_map = {v: k for k, v in nfl_team_map.items()}
    Oddsdf['name'] = Oddsdf['name'].replace(inverse_nfl_team_map)
    Oddsdf['name'] = Oddsdf['name'].replace({
            'Chigoziem Okonkwo': 'Chig Okonkwo',
            'Scott Miller': 'Scotty Miller',
            'Gabriel Davis': 'Gabe Davis'
        })

    df = DKdf.merge(Oddsdf,left_on='Name',right_on='name',how='left')


    columns_to_transform = ['odds on over pass tds','odds on over interception','odds on td scored','odds on pass receptions','odds on kicker points']
    for col in columns_to_transform:
        for idx, value in df[col].items():
            df.at[idx, col] = transform_odds(value)

    df = df[['TeamAbbrev','Name','Roster Position','Salary','# of pass tds', 'odds on over pass tds','passing yds line',
               'odds on over interception', 'odds on td scored', 'receiving yds line',
               'rush yds line', 'number of pass receptions', 'odds on pass receptions',
               'kicker points line', 'odds on kicker points']]

    df.rename({'TeamAbbrev':'team','Name':'name','Roster Position':'position','Salary':'cost'},axis=1,inplace=True)
    
    return df

## NCAA Football
def get_ncaa_game_ids():
    # Endpoint
    endpoint = "https://api-external.oddsjam.com/api/v2/games"

    # Current week date range
    today = datetime.now(timezone.utc)
    start_date = today.isoformat()


    # Calculate the upcoming Monday or today if it's Monday
    days_until_monday = (7 - today.weekday()) % 7
    monday = today + timedelta(days=days_until_monday)
    
    # Set time to 11:59 pm and adjust for CST (UTC-6)
    end_date_cst = (monday + timedelta(days=1)).replace(hour=5, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)
    # Parameters
    params = {
        "key": os.environ.get('ODDSJAM_API_KEY'),
        "sport": "football",
        "league": "NCAAF",
        "start_date_after": start_date,
        "start_date_before": end_date_cst.isoformat()
    }

    # Make the request
    response = requests.get(endpoint, params=params)

    # Check response
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    data = response.json()
    game_ids = [game['id'] for game in data['data']]
    return game_ids
def ncaa_fetch_game_data(game_ids):
    desired_markets = [
        "Player Passing Touchdowns",
        "Player Interceptions",
        "Player Touchdowns",
        "Player Receiving Yards",
        "Player Passing Yards",
        "Player Rushing Yards",
        "Player Receptions",
        "Player Kicking Points"
    ]

    URL = "https://api-external.oddsjam.com/api/v2/game-odds"
    API_KEY = os.environ.get('ODDSJAM_API_KEY')  

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
    print(response)
    json_data = all_data

    desired_columns = [
    "name",
    "# of pass tds",
    "odds on over pass tds",
    "odds on over interception",
    "odds on td scored",
    "receiving yds line",
    "passing yds line",
    "rush yds line",
    "number of pass receptions",
    "odds on pass receptions",
    "kicker points line",
    "odds on kicker points"
    ]

    df = pd.DataFrame(columns=desired_columns)

# Map values from the updated JSON data to the DataFrame using the corrected conditions
    for entry in json_data:
        for bet in entry['odds']:
            player_name = bet['selection']
            market_name = bet['market_name']
            selection_line = bet['selection_line']
            selection_points = bet['selection_points']
            price = bet['price']

            # Determine the appropriate column based on market name and selection line
            if "Player Passing Touchdowns" in market_name and selection_line == "over":

                column_name = "# of pass tds"
                value = selection_points
                # Check if player exists in the DataFrame
                if player_name in df['name'].values:
                    df.loc[df['name'] == player_name, column_name] = value
                    df.loc[df['name'] == player_name,"odds on over pass tds"]= price
                else:
                    # If player doesn't exist, add new player data
                    new_row = pd.DataFrame([{'name': player_name, column_name: value, "odds on over pass tds": price}])
                    df = pd.concat([df,new_row], ignore_index=True)

            elif "Player Interceptions" in market_name and selection_line == "over":
                print(f"Found Player Interceptions for {player_name} with selection line: {selection_line}")

                column_name = "odds on over interception"
                value = price
            elif market_name == "Player Passing Yards" and selection_line == "over":
                column_name = "passing yds line"
                value = selection_points
            elif market_name == "Player Touchdowns" and selection_line == "over" and selection_points == 0.5:
                column_name = 'odds on td scored'
                value = price
            elif market_name == "Player Receiving Yards" and selection_line == "over":
                column_name = "receiving yds line"
                value = selection_points
            elif market_name == "Player Rushing Yards" and selection_line == "over":
                column_name = "rush yds line"
                value = selection_points 
            elif market_name == "Player Receptions" and selection_line == "over":
                column_name = "number of pass receptions"
                value = selection_points
                # Check if player exists in the DataFrame
                if player_name in df['name'].values:
                    df.loc[df['name'] == player_name, column_name] = value
                    df.loc[df['name'] == player_name, "odds on pass receptions"] = price
                else:
                    new_row = pd.DataFrame([{'name': player_name, column_name: value, "odds on pass receptions": price}])
                    df = pd.concat([df,new_row], ignore_index=True)
            elif market_name == "Player Kicking Points" and selection_line == "over":
                column_name = "kicker points line"
                value = selection_points
                # Check if player exists in the DataFrame
                if player_name in df['name'].values:
                    df.loc[df['name'] == player_name, column_name] = value
                    df.loc[df['name'] == player_name, "odds on kicker points"] = price
                else:
                    new_row = pd.DataFrame([{'name': player_name, column_name: value, "odds on kicker points": price}])
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                continue

            # Update the DataFrame with the extracted values (excluding the special cases)
            if market_name not in ["Player Passing Touchdowns", "Player Receptions", "Player Kicking Points"]:
                if player_name not in df['name'].values:
                    
                    new_row = pd.DataFrame({'name': [player_name], column_name: [value]})
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    df.loc[df['name'] == player_name, column_name] = value
    return df
def ncaa_process_data_fd(file_path):
    
    todays_ids = get_ncaa_game_ids()
    Oddsdf = ncaa_fetch_game_data(todays_ids)
    FDdf = pd.read_csv(file_path)
    FDdf = FDdf[['Nickname','Salary','Team','Roster Position']]

    Oddsdf['name'] = Oddsdf['name'].apply(lambda x: ''.join(ch for ch in x if ch not in string.punctuation))

    Oddsdf['name'] = Oddsdf['name'].apply(lambda x: x.split(' Defense')[0] if 'Defense' in x else x)
    
    # Naming convention normalizing

    FDdf['Nickname'] = FDdf['Nickname'].apply(lambda x: ''.join(ch for ch in x if ch not in string.punctuation))

    Fdf = FDdf.merge(Oddsdf,left_on='Nickname',right_on='name',how='left')



    columns_to_transform = ['odds on over pass tds','odds on over interception','odds on td scored','odds on pass receptions','odds on kicker points']
    for col in columns_to_transform:
            for idx, value in Fdf[col].items():
                Fdf.at[idx, col] = transform_odds(value)

    Fdf = Fdf[['Team','Nickname','Roster Position','Salary','# of pass tds', 'odds on over pass tds',
               'odds on over interception', 'odds on td scored', 'receiving yds line','passing yds line',
               'rush yds line', 'number of pass receptions', 'odds on pass receptions',
               'kicker points line', 'odds on kicker points']]

    Fdf.rename({'Team':'team','Nickname':'name','Roster Position':'position','Salary':'cost'},axis=1,inplace=True)
    return Fdf

## PGA
def get_to_make_the_cut_odds(MakeCutID):
    API_ENDPOINT = "https://api-external.oddsjam.com/api/v2/future-odds"
    PARAMS = {
                "key":os.environ.get('ODDSJAM_API_KEY'),
                "sport": "golf",
                "league": "PGA",
                "sportsbook": "DraftKings",
                "future_id": MakeCutID
            }

    response = requests.get(API_ENDPOINT, params=PARAMS)
    if response.status_code == 200:
        odds_data = response.json()
    else:
        x=0
    odds_data = odds_data['data'][0]['odds']

    # Reformatting the data into two columns: 'To Make the Cut' and 'To Miss the Cut'
    formatted_data = {}
    for item in odds_data:
        # Extract player name and whether it's for making or missing the cut
        player_name = ' '.join(item['selection'].split()[:-1])
        make_or_miss = item['selection'].split()[-1]

        # Initialize the player entry if not already present
        if player_name not in formatted_data:
            formatted_data[player_name] = {'To Make the Cut': None, 'To Miss the Cut': None}

        # Assign the odds to the appropriate category
        if make_or_miss == 'Yes':
            formatted_data[player_name]['To Make the Cut'] = item['price']
        elif make_or_miss == 'No':
            formatted_data[player_name]['To Miss the Cut'] = item['price']

    # Displaying a portion of the formatted data for brevity
    df = pd.DataFrame.from_dict(formatted_data, orient='index')

    # Setting the golfer's name as the index
    df.index.name = 'Golfer Name'
    df.reset_index(inplace=True)
    df.rename({'Golfer Name':'name'},axis=1,inplace=True)

    return df

def get_pga_futures(tournament_name):
    API_ENDPOINT = "https://api-external.oddsjam.com/api/v2/futures"
    PARAMS = {
        "key": os.environ.get('ODDSJAM_API_KEY'),
        "sport": "golf",
        "league": "PGA"
    }
    
    response = requests.get(API_ENDPOINT, params=PARAMS)
    if response.status_code == 200:
        futures_data = response.json()["data"]
        # Filter for likely men's PGA futures
        pga_futures = [future for future in futures_data if "LPGA" not in future["name"]]
    else:
        return []
    filtered_data = [d for d in pga_futures if tournament_name in d['name']]
    return filtered_data

def get_draftkings_odds(future_ids):
    API_ENDPOINT = "https://api-external.oddsjam.com/api/v2/future-odds"
    PARAMS = {
            "key": os.environ.get('ODDSJAM_API_KEY'),
            "sport": "golf",
            "league": "PGA",
            "sportsbook": "DraftKings",
            "future_id": future_ids
        }

    response = requests.get(API_ENDPOINT, params=PARAMS)
    if response.status_code == 200:
        odds_data = response.json()
    else:
        x=0
    df = odds_data


    df = pd.DataFrame(odds_data['data'])
    df['bet_type'] = df['id'].str.split('-').apply(lambda x: x[2] if len(x) > 2 else None)
    flattened_df = df.explode('odds')


    flattened_df['id'].unique()

    flattened_df.reset_index(inplace=True)

    flattened_df.drop('index',axis=1,inplace=True)

    flattened_df['Golfer_name'] = flattened_df['odds'].apply(lambda x: x['selection'] if 'selection' in x else None)
    flattened_df['price'] = flattened_df['odds'].apply(lambda x: x['price'] if 'price' in x else None)


    pivot_df = flattened_df.pivot(index='Golfer_name',columns='name', values='price')

    return pivot_df


def process_golf_data(file_path,tournament):
    pga_futures = get_pga_futures(tournament_name=tournament)

    future_ids = [future["id"] for future in pga_futures]
    future_names = [future["name"] for future in pga_futures]

    keywords = ['winner',"top_10", "top_5", "top_20", "top_30", "top_40",'make_the_cut']

    filtered_futures = [s for s in future_ids if any(keyword in s for keyword in keywords) and "first_round" not in s]
    make_cut_ids = [s for s in filtered_futures if "make_the_cut" in s]
    if make_cut_ids:
        make_cut_id = make_cut_ids[0]  # assuming you want the first match
    else:
        make_cut_id = None  # or handle the case where there is no match
    odds_data = get_draftkings_odds(future_ids=filtered_futures)
    odds_data.reset_index(inplace=True)
    odds_data.rename({'Golfer_name':'name'},axis=1,inplace=True)
    cut_data = get_to_make_the_cut_odds(MakeCutID=make_cut_id)
    odds_data = odds_data.merge(cut_data,how='left',left_on='name',right_on='name')
    odds_data.rename({'bet_name':'name',tournament+' Top 10 Finish' :'top 10',tournament+' Top 20 Finish':'top 20',
                     tournament+' Top 30 Finish':'top 30',tournament+' Top 40 Finish':'top 40',
                      tournament+' Top 5 Finish':'top 5',
                     tournament+' Winner':'winner'},axis=1,inplace=True)
    fdDF = pd.read_csv(file_path)
    fdDF = fdDF[['Nickname','Salary']]

    fdDF.rename({'Nickname':'name','Salary':'cost'},axis=1,inplace=True)
    fdDF['name'] = fdDF['name'].replace('Seonghyeon Kim', 'S.H. Kim')
    fdDF['name'] = fdDF['name'].replace('Kyoung-Hoon Lee', 'K.H. Lee')
    s = {}

        # For each key in fdDF
    for k1 in fdDF['name']:
    # Find the closest match in odds_data
        matches = [(k2, fuzz.ratio(k1, k2)) for k2 in odds_data['name']]
        # Sort matches by score in descending order
        matches = sorted(matches, key=lambda x: x[1], reverse=True)

        # If the highest score is above threshold
        if matches[0][1] >= 80:
            s[k1] = matches[0][0]

        # Map the matches in fdDF
    fdDF['matched_key'] = fdDF['name'].map(s)

        # Merge dataframes on the keys
    merged = fdDF.merge(odds_data.rename(columns={'name': 'matched_key'}), on='matched_key', how='left')
    merged.rename({'bet_name':'name',tournament+' Top 10 Finish' :'top 10',tournament+' Top 20 Finish':'top 20',
                     tournament+' Top 30 Finish':'top 30',tournament+' Top 40 Finish':'top 40',
                      tournament+' Top 5 Finish':'top 5',
                     tournament+' Winner':'winner'},axis=1,inplace=True)

        # Drop the matched_key column for cleaner output
    merged.drop(columns=['matched_key'], inplace=True)

    columns_to_transform = ['To Make the Cut','top 5', 'top 10', 'top 20', 'top 30', 'top 40','winner']

    for col in columns_to_transform:
        if col in merged.columns:
            for idx, value in merged[col].items():
                merged.at[idx, col] = transform_odds(value)
    final_columns = ['name', 'cost','To Make the Cut', 'winner', 'top 5', 'top 10', 'top 20', 'top 30', 'top 40']
    existing_columns = [col for col in final_columns if col in merged.columns]
    final = merged[existing_columns]
    return final

def dk_process_golf_data(file_path,tournament):
    pga_futures = get_pga_futures(tournament_name=tournament)

    future_ids = [future["id"] for future in pga_futures]
    future_names = [future["name"] for future in pga_futures]

    keywords = ['championship_winner',"top_10", "top_5", "top_20", "top_30", "top_40",'make_the_cut']

    filtered_futures = [s for s in future_ids if any(keyword in s for keyword in keywords) and "first_round" not in s]
    make_cut_ids = [s for s in filtered_futures if "make_the_cut" in s]
    if make_cut_ids:
        make_cut_id = make_cut_ids[0]  # assuming you want the first match
    else:
        make_cut_id = None  # or handle the case where there is no match
    odds_data = get_draftkings_odds(future_ids=filtered_futures)
    odds_data.reset_index(inplace=True)
    odds_data.rename({'Golfer_name':'name'},axis=1,inplace=True)
    cut_data = get_to_make_the_cut_odds(MakeCutID=make_cut_ids)
    odds_data = odds_data.merge(cut_data,how='left',left_on='name',right_on='name')
    odds_data.rename({'bet_name':'name',tournament+' Top 10 Finish' :'top 10',tournament+' Top 20 Finish':'top 20',
                     tournament+' Top 30 Finish':'top 30',tournament+' Top 40 Finish':'top 40',
                      tournament+' Top 5 Finish':'top 5',
                     tournament+' Winner':'winner'},axis=1,inplace=True)
    dkDF = pd.read_csv(file_path)
    dkDF = dkDF[['Name','Salary']]

    dkDF.rename({'Name':'name','Salary':'cost'},axis=1,inplace=True)
    dkDF['name'] = dkDF['name'].replace('Seonghyeon Kim', 'S.H. Kim')
    dkDF['name'] = dkDF['name'].replace('Kyoung-Hoon Lee', 'K.H. Lee')
    s = {}

        # For each key in fdDF
    for k1 in dkDF['name']:
    # Find the closest match in odds_data
        matches = [(k2, fuzz.ratio(k1, k2)) for k2 in odds_data['name']]
        # Sort matches by score in descending order
        matches = sorted(matches, key=lambda x: x[1], reverse=True)

        # If the highest score is above threshold
        if matches[0][1] >= 80:
            s[k1] = matches[0][0]

        # Map the matches in fdDF
    dkDF['matched_key'] = dkDF['name'].map(s)

        # Merge dataframes on the keys
    merged = dkDF.merge(odds_data.rename(columns={'name': 'matched_key'}), on='matched_key', how='left')

        # Drop the matched_key column for cleaner output
    merged.drop(columns=['matched_key'], inplace=True)
    merged.rename({'bet_name':'name',tournament+' Top 10 Finish' :'top 10',tournament+' Top 20 Finish':'top 20',
                     tournament+' Top 30 Finish':'top 30',tournament+' Top 40 Finish':'top 40',
                      tournament+' Top 5 Finish':'top 5',
                     tournament+' Winner':'winner'},axis=1,inplace=True)

    columns_to_transform = ['To Make the Cut','top 5', 'top 10', 'top 20', 'top 30', 'top 40','winner']

    for col in columns_to_transform:
        if col in merged.columns:
            for idx, value in merged[col].items():
                merged.at[idx, col] = transform_odds(value)
    final_columns = ['name', 'cost','To Make the Cut', 'winner', 'top 5', 'top 10', 'top 20', 'top 30', 'top 40']
    existing_columns = [col for col in final_columns if col in merged.columns]
    final = merged[existing_columns]
    return final

## NBA
def nba_get_todays_game_ids():
    # Initialize the client (assuming you have a client setup)
    Client = OddsJamClient(os.environ.get('ODDSJAM_API_KEY'))
    Client.UseV2()
    # Get games for the league
    GamesResponse = Client.GetGames(league='nba')
    Games = GamesResponse.Games
    
    # Filter games based on today's and tomorrow's dates
    games_data = [{'game_id': game.id, 'start_date': game.start_date} for game in Games]
    games_df = pd.DataFrame(games_data)
    
    # Convert the start_date column to strings, remove the timezone information, 
    # and then convert to datetime dtype
    games_df['start_date'] = games_df['start_date'].astype(str).str[:-6]
    games_df['start_date'] = pd.to_datetime(games_df['start_date'])
    
    today = datetime.now().date()
    #tomorrow = today + timedelta(days=1)
    
    todays_games = games_df[(games_df['start_date'].dt.date == today)]
    
    return todays_games['game_id'].tolist()


def get_nba_odds(game_ids):
    
    desired_markets = ['Player Points','Player Rebounds',
                   'Player Assists','Player Steals + Blocks',
                   'Player Turnovers',
                  'Player Double Double','Player Triple Double']
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
                'market_name': desired_markets,
                'is_main':'true'
            }

        response = requests.get(URL, params=params)

        if response.status_code == 200:
            all_data.extend(response.json()['data'])
        else:
            print(f"Error {response.status_code}: {response.text}")
    desired_columns = ["name", "Player Points", "Player Points odds", "Player Rebounds", 
                           "Player Rebounds odds", "Player Assists", "Player Assists odds", 
                           "Player Steals + Blocks", "Player Steals + Blocks odds", 
                           "Player Turnovers", "Player Turnovers odds", 
                           "Player Double Double odds", "Player Triple Double odds"]
    df = pd.DataFrame(columns=desired_columns)
    json_data = all_data
    # Iterate through the response data and populate the DataFrame
    for entry in json_data:
        for odds_entry in entry['odds']:
            player_name = odds_entry['selection']
            market_name = odds_entry['market_name']
            selection_points = odds_entry['selection_points']
            price = odds_entry['price']
            selection_line = odds_entry['selection_line']

            # Check if the player already exists in the DataFrame
            player_index = df.index[df['name'] == player_name]
            if not player_index.empty:
                player_index = player_index[0]
            else:
                # If player doesn't exist, add new player data using concat
                new_row = pd.DataFrame({'name': [player_name]})
                df = pd.concat([df, new_row], ignore_index=True)
                player_index = df.index[df['name'] == player_name][0]

            # Map the data to the DataFrame based on the market name and selection line
            if market_name in ["Player Double Double", "Player Triple Double"]:
                df.loc[player_index, f'{market_name} odds'] = price
            elif selection_line == 'over':
                df.loc[player_index, f'{market_name}'] = selection_points
                df.loc[player_index, f'{market_name} odds'] = price

    # Display the DataFrame
    return df 


def fd_nba_process_data(file_path,file_path2):
    ids = nba_get_todays_game_ids()
    df = get_nba_odds(ids)
    df['name'] = df['name'].str.replace(' Jr.', '').str.replace(' III', '').str.replace(' Jr','').str.replace(' Sr.','').str.replace(' II','').str.replace(' Sr','')



    fdDF = pd.read_csv(file_path)
    fdDF['Nickname'] = fdDF['Nickname'].str.replace(' Jr.', '').str.replace(' III', '').str.replace(' Jr','').str.replace(' Sr.','').str.replace(' II','').str.replace(' Sr','')


    fdDF = fdDF[['Nickname','Roster Position','Salary','Team']]

    Fdf = fdDF.merge(df,left_on='Nickname',right_on='name',how='left')

    columns_to_transform = ['Player Points odds','Player Rebounds odds',
                            'Player Assists odds','Player Steals + Blocks odds',
                            'Player Turnovers odds',"Player Double Double odds",
                           "Player Triple Double odds"]
    for col in columns_to_transform:
            for idx, value in Fdf[col].items():
                Fdf.at[idx, col] = transform_odds(value)


    Fdf.rename(columns={
        'Team':'team',
        'Nickname':'player',
        'Roster Position':'position',
        'Salary':'cost',
            'Player Points': 'pts',
            'Player Points odds': 'pts odds',
            'Player Rebounds': 'rebounds',
            'Player Rebounds odds': 'rebounds odds',
            'Player Assists': 'assists',
            'Player Assists odds': 'assists odds',
            'Player Steals + Blocks': 'steals and blocks',
            'Player Steals + Blocks odds': 'steals and blocks odds',
            'Player Turnovers': 'turnovers',
            'Player Turnovers odds': 'turnovers odds',
            'Player Double Double odds': 'DD odds',
            'Player Triple Double odds': 'TD odds'
        },inplace=True)

    final = Fdf[['team','player','position','cost','pts','pts odds','rebounds','rebounds odds','assists','assists odds',
        'steals and blocks','steals and blocks odds','turnovers','turnovers odds','DD odds','TD odds']]
    
    rgdf = pd.read_csv(file_path2)
    rgdf = rgdf[['PLAYER','FPTS']]
    rgdf = rgdf[rgdf['FPTS'].notna()]
    merged_df = final.merge(rgdf, left_on='player', right_on='PLAYER', how='left')

    # Define the columns to check for NaN
    numerical_columns = ['pts','pts odds','rebounds','rebounds odds','assists','assists odds',
        'steals and blocks','steals and blocks odds','turnovers','turnovers odds','DD odds','TD odds']
    # Create a mask where all numerical columns are NaN
    all_na_mask = merged_df[numerical_columns].isna().all(axis=1)
    # Replace 'pts' only where all numerical columns are NaN
    merged_df.loc[all_na_mask, 'pts'] = merged_df.loc[all_na_mask, 'FPTS']
    # Drop the unnecessary columns
    FdDF = merged_df.drop(columns=['PLAYER', 'FPTS'])
    return FdDF



    return FdDF

def dk_nba_process_data(file_path,file_path2):
    ids = nba_get_todays_game_ids()
    df = get_nba_odds(ids)

    dkDF = pd.read_csv(file_path)

    dkDF = dkDF[['Name','Roster Position','Salary','TeamAbbrev']]

    Fdf = dkDF.merge(df,left_on='Name',right_on='name',how='left')

    columns_to_transform = ['Player Points odds','Player Rebounds odds',
                                'Player Assists odds','Player Steals + Blocks odds',
                                'Player Turnovers odds',"Player Double Double odds",
                               "Player Triple Double odds"]
    for col in columns_to_transform:
            for idx, value in Fdf[col].items():
                Fdf.at[idx, col] = transform_odds(value)

    Fdf.rename(columns={
            'TeamAbbrev':'team',
            'Name':'player',
            'Roster Position':'position',
            'Salary':'cost',
                'Player Points': 'pts',
                'Player Points odds': 'pts odds',
                'Player Rebounds': 'rebounds',
                'Player Rebounds odds': 'rebounds odds',
                'Player Assists': 'assists',
                'Player Assists odds': 'assists odds',
                'Player Steals + Blocks': 'steals and blocks',
                'Player Steals + Blocks odds': 'steals and blocks odds',
                'Player Turnovers': 'turnovers',
                'Player Turnovers odds': 'turnovers odds',
                'Player Double Double odds': 'DD odds',
                'Player Triple Double odds': 'TD odds'
            },inplace=True)

    final = Fdf[['team','player','position','cost','pts','pts odds','rebounds','rebounds odds','assists','assists odds',
            'steals and blocks','steals and blocks odds','turnovers','turnovers odds','DD odds','TD odds']]
    rgdf = pd.read_csv(file_path2)
    rgdf = rgdf[['PLAYER','FPTS']]
    rgdf = rgdf[rgdf['FPTS'].notna()]
    merged_df = final.merge(rgdf, left_on='player', right_on='PLAYER', how='left')
    numerical_columns = ['pts','pts odds','rebounds','rebounds odds','assists','assists odds',
        'steals and blocks','steals and blocks odds','turnovers','turnovers odds','DD odds','TD odds']
    # Create a mask where all numerical columns are NaN
    all_na_mask = merged_df[numerical_columns].isna().all(axis=1)
    # Replace 'pts' only where all numerical columns are NaN
    merged_df.loc[all_na_mask, 'pts'] = merged_df.loc[all_na_mask, 'FPTS']
    # Drop the unnecessary columns
    dkDF = merged_df.drop(columns=['PLAYER', 'FPTS'])
    return dkDF



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
@app.route('/process_mlb_fd_optimized', methods=['POST'])
def process_file_mlb_optimized_fd():
    if 'file' not in request.files:
        return 'No file uploaded!'
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = projected_df(filepath)
    final_proj = mlb_fanduel_optimized(df = final_proj)

    output_filename = "fanduel_optomized_mlb.csv"
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
@app.route('/process_mlb_dk_optimized', methods=['POST'])
def process_file_mlb_optimized_dk():
    if 'file' not in request.files:
        return 'No file uploaded!'
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = draftkings_projected_df(filepath)
    final_proj = mlb_draftkings_optimized(df = final_proj)

    output_filename = "draftkings_optimized_mlb.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)


@app.route('/process_nfl_fd', methods=['POST'])
def process_file_nfl_fd():
    if 'file' not in request.files:
        return 'No file uploaded!'
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = process_nfl_data(filepath)  # Assuming you're processing the DraftKings file the same way

    output_filename = "processed_output_nfl.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/process_nfl_dk', methods=['POST'])
def process_file_nfl_dk():
    if 'file' not in request.files:
        return 'No file uploaded!'
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = nfl_draftkings_process(filepath)  # Assuming you're processing the DraftKings file the same way

    output_filename = "dk_processed_output_nfl.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/process_ncaaf_fd', methods=['POST'])
def process_file_ncaaf_fd():
    if 'file' not in request.files:
        return 'No file uploaded!'
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = ncaa_process_data_fd(file_path=filepath)  # Assuming you're processing the DraftKings file the same way

    output_filename = "fd_processed_output_ncaaf.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/process_golf_fd', methods=['POST'])
def process_file_pga_fd():
    if 'file' not in request.files:
        return 'No file uploaded!'
    tournament_name = request.form.get('tournament_name')
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = process_golf_data(filepath,tournament=tournament_name)  # Assuming you're processing the DraftKings file the same way

    output_filename = "fd_processed_output_pga.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/process_golf_dk', methods=['POST'])
def process_file_pga_dk():
    if 'file' not in request.files:
        return 'No file uploaded!'
    tournament_name = request.form.get('tournament_name', 'Sony Open')
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = dk_process_golf_data(filepath,tournament=tournament_name) 

    output_filename = "dk_processed_output_pga.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/process_nba_fd', methods=['POST'])
def process_file_nba_fd():
    if 'fd_salaries' not in request.files or 'nba_projections' not in request.files:
        return 'Both files (FD Salaries and NBA Projections) need to be uploaded!'

    fd_salaries_file = request.files['fd_salaries']
    fd_salaries_filename = fd_salaries_file.filename
    fd_salaries_filepath = os.path.join(app.config['UPLOAD_FOLDER'], fd_salaries_filename)
    fd_salaries_file.save(fd_salaries_filepath)

    nba_projections_file = request.files['nba_projections']
    nba_projections_filename = nba_projections_file.filename
    nba_projections_filepath = os.path.join(app.config['UPLOAD_FOLDER'], nba_projections_filename)
    nba_projections_file.save(nba_projections_filepath)

    final_proj = fd_nba_process_data(fd_salaries_filepath, nba_projections_filepath)  # Assuming you have a similar function for FanDuel

    output_filename = "fd_processed_output_nba.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)

    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/process_nba_dk', methods=['POST'])
def process_file_nba_dk():
    if 'dk_salaries' not in request.files or 'nba_projections' not in request.files:
        return 'Both files (DK Salaries and NBA Projections) need to be uploaded!'

    dk_salaries_file = request.files['dk_salaries']
    dk_salaries_filename = dk_salaries_file.filename
    dk_salaries_filepath = os.path.join(app.config['UPLOAD_FOLDER'], dk_salaries_filename)
    dk_salaries_file.save(dk_salaries_filepath)

    nba_projections_file = request.files['nba_projections']
    nba_projections_filename = nba_projections_file.filename
    nba_projections_filepath = os.path.join(app.config['UPLOAD_FOLDER'], nba_projections_filename)
    nba_projections_file.save(nba_projections_filepath)

    final_proj = dk_nba_process_data(dk_salaries_filepath, nba_projections_filepath)

    output_filename = "dk_processed_output_nba.csv"
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




