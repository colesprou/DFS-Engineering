from flask import Flask, render_template, request, send_from_directory, send_file, url_for, redirect
import os
from cole_fantasy_proj import pitcher_data_dfs, batter_data_dfs, generate_fantasy_projections, generate_multiple_lineups,extract_opponent

from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary
import ortools
from ortools.linear_solver import pywraplp

import pandas as pd
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
import logging 
from bs4 import BeautifulSoup
import xlsxwriter
import numpy as np
import requests
from datetime import datetime, timedelta, timezone, time
from OddsJamClient import OddsJamClient
import string

import os
from dotenv import load_dotenv
import json
from fuzzywuzzy import fuzz, process



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
# Pick 6
def fetch_game_data_pick_6(game_ids, api_key, market_type='player', league='MLB', sportsbooks=['DraftKings (Pick 6)', 'DraftKings'], include_player_name=True):
    markets = []
    sport = ''
    
    # Set sport and markets for each league
    if league == 'MLB':
        sport = 'baseball'
        markets = ['Player Hits + Runs + RBIs', 'Player Strikeouts', 'Player Hits', 'Player Hits Allowed', 'Player Singles']
    elif league == 'NFL':
        sport = 'football'
        url = "https://api.opticodds.com/api/v3/markets"
        params = {
            "key": api_key,
            "sport": sport,
            "sportsbook": 'DraftKings (Pick 6)',
            "league": 'NFL'
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            markets_data = response.json()
            markets = [market['name'] for market in markets_data['data'] if 'Player' in market['name']]
        else:
            print(f"Failed to retrieve NFL data, status code: {response.status_code}")
    elif league == 'NBA':
        sport = 'basketball'
        url = "https://api.opticodds.com/api/v3/markets"
        params = {
            "key": api_key,
            "sport": sport,
            "sportsbook": 'DraftKings (Pick 6)',
            "league": 'NBA'
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            markets_data = response.json()
            markets = [market['name'] for market in markets_data['data'] if 'Player' in market['name']]
        else:
            print(f"Failed to retrieve NBA data, status code: {response.status_code}")
    elif league == 'NHL':
        sport = 'hockey'
        url = "https://api.opticodds.com/api/v3/markets"
        params = {
            "key": api_key,
            "sport": sport,
            "sportsbook": 'DraftKings (Pick 6)',
            "league": 'NHL'
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            markets_data = response.json()
            markets = [market['name'] for market in markets_data['data'] if 'Player' in market['name']]
        else:
            print(f"Failed to retrieve NHL data, status code: {response.status_code}")

    # Main data fetching URL
    url = "https://api-external.oddsjam.com/api/v2/game-odds"
    all_data = []

    # Fetch data in chunks for each sportsbook and game ID
    for chunk in [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]:
        for sportsbook in sportsbooks:
            params = {
                'key': api_key,
                'sportsbook': sportsbook,
                'game_id': chunk,
                'market_name': markets
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json().get('data', [])
                all_data.extend(data)
            else:
                print(f"Error fetching data for {sportsbook}: {response.status_code} - {response.text}")

    # Convert the fetched data into a DataFrame
    rows = []
    for game_data in all_data:
        home_team = game_data.get('home_team', 'Unknown')
        away_team = game_data.get('away_team', 'Unknown')
        start_date = game_data.get('start_date', 'Unknown')
        odds_list = game_data.get('odds', [])
        
        for item in odds_list:
            row = {
                'Game ID': game_data.get('id', 'Unknown'),
                "Start Date": start_date,
                "Game Name": f"{home_team} vs {away_team}",
                "Bet Name": item.get('name', None),
                'Market Name': item.get('market_name', ''),
                'Sportsbook': item.get('sports_book_name', sportsbook),
                'line': item.get('bet_points', None),
                'Odds': item.get('price', None),
            }
            if include_player_name and market_type == 'player':
                row['Player Name'] = item.get('selection', 'Unknown')
            rows.append(row)

    df = pd.DataFrame(rows)
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce').dt.tz_convert('US/Eastern')
    return df
def pick_6_data(league):
    if league == 'MLB':
        game_ids = get_todays_game_ids()
    if league == 'NFL':
        game_ids = get_nfl_game_ids()
    if league == 'NBA':
        game_ids = nba_get_todays_game_ids()
    if league == "NHL":
        game_ids = nhl_get_todays_game_ids()
    df = fetch_game_data_pick_6(game_ids, api_key=os.environ.get('ODDSJAM_API_KEY'), market_type='player', league=league, sportsbooks=['DraftKings (Pick 6)','DraftKings'], include_player_name=True)
    df_draftkings = df[df['Sportsbook'] == 'DraftKings']

    df_pick6 = df[df['Sportsbook'] == 'DraftKings (Pick 6)']

    merged_df = pd.merge(df_draftkings, df_pick6, 
                         on=['Game ID', 'Market Name', 'Bet Name', 'line', 'Player Name'], 
                         suffixes=('_DraftKings', '_Pick6'))
    def determine_more_or_less(row):
        if 'Over' in row['Bet Name']:
            return 'More'
        elif 'Under' in row['Bet Name']:
            return 'Less'
        return None

    merged_df['More_or_Less'] = merged_df.apply(determine_more_or_less, axis=1)

    final_df = merged_df[['Player Name',"Game Name_DraftKings" ,'Market Name', 'line', 'More_or_Less', 'Odds_DraftKings']]
    final_df.columns = ['Player',"Game Name", 'Category', 'Number', 'More or Less', 'Odds']
    final_df_filtered = merged_df.loc[merged_df.groupby(['Player Name', 'Market Name'])['Odds_DraftKings'].idxmin()]

    final_df_filtered_output = final_df_filtered[['Player Name','Game Name_DraftKings', 'Market Name', 'line', 'More_or_Less', 'Odds_DraftKings']]
    final_df_filtered_output.columns = ['Player', "Game",'Category', 'Number', 'More or Less', 'Odds']
    return final_df_filtered_output

def get_pitcher_data(game_ids):
    desired_markets = ['Player Strikeouts','Player Outs','Player Walks','Player Earned Runs',
                       'Player To Record Win','Player Hits Allowed']

    URL = "https://api.opticodds.com/api/v3/fixtures/odds"
    API_KEY =os.environ.get('ODDSJAM_API_KEY')

        # Break game IDs into chunks of 5
    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]

    all_data = []

        # For each chunk, fetch odds
    for chunk in game_id_chunks:
        params = {
                'key': API_KEY,
                'sportsbook': ['DraftKings'],
                'fixture_id': chunk,
                'market': desired_markets,
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
                    if odd['market_id'] == 'player_hits_allowed':
                        revised_player_data[player_name]["Hits Line"] = odd.get('points', None)
                        revised_player_data[player_name]["Hits Odds"] = odd.get('price', None)
                    if odd['market_id'] == 'player_strikeouts':
                        revised_player_data[player_name]["Strikeout Line"] = odd.get('points', None)
                        revised_player_data[player_name]["Strikeout Odds"] = odd.get('price', None)
                    elif odd['market_id'] == 'player_outs':
                        revised_player_data[player_name]["Outs Line"] = odd.get('points', None)
                        revised_player_data[player_name]["Outs Odds"] = odd.get('price', None)
                    elif odd['market_id'] == 'player_walks':
                        revised_player_data[player_name]["Walks Line"] = odd.get('points', None)
                        revised_player_data[player_name]["Walks Odds"] = odd.get('price', None)
                    elif odd['market_id'] == 'player_earned_runs':
                        revised_player_data[player_name]["Earned Run Line"] = odd.get('points', None)
                        revised_player_data[player_name]["Earned Run Odds"] = odd.get('price', None)
                elif odd['market_id'] == 'player_to_record_win' and "yes" in odd['name'].lower():
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
def fetch_game_data2(game_ids):
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

    URL = "https://api.opticodds.com/api/v3/fixtures/odds"
    API_KEY = os.environ.get('ODDSJAM_API_KEY')

    # Break game IDs into chunks of 5
    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]

    all_data = []

    # For each chunk, fetch odds
    for chunk in game_id_chunks:
        params = {
            'key': API_KEY,
            'sportsbook': ['DraftKings'],
            'fixture_id': chunk,
            'market': desired_markets
        }

        response = requests.get(URL, params=params)
        print("Request params:", params)
        print("Response status code:", response.status_code)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                all_data.extend(data['data'])
            else:
                print("No 'data' key in response")
        else:
            print(f"Error {response.status_code}: {response.text}")

    if not all_data:
        print("No data retrieved from API")
        
    rows = []

    # Iterate through all games in the aggregated response data
    for game_data in all_data:
        odds_list = game_data.get('odds', [])
        home_team = game_data.get('home_team', 'Unknown')
        away_team = game_data.get('away_team', 'Unknown')
        
        for item in odds_list:
            market_name = item.get('market', '')
            if market_name not in desired_markets:
                continue
            
            player_name = item.get('selection', 'Unknown')
            column_name = f"{item.get('market', '')} {item.get('selection_line', '')} {item.get('points', '')}"
            
            if player_name == home_team:
                team = home_team
            elif player_name == away_team:
                team = away_team
            else:
                team = 'Unknown'  

            row = {
                'Player Name': player_name,
                'Team': team,
                column_name: item.get('price', None)
            }
            rows.append(row)
            
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    print("DataFrame columns:", df.columns)
    # Aggregate rows by player name
    df = df.groupby(['Player Name', 'Team']).sum().reset_index()

    df['Player Name'] = df['Player Name'].str.replace('[^A-Za-z\s]', '', regex=True)
    df['Player Name'] = df['Player Name'].apply(lambda x: ' '.join(x.split(' ')[:2]))
    df.loc[df['Player Name'].str.contains('JiHwan Bae'), 'Player Name'] = df['Player Name'].str.replace('JiHwan Bae', 'Ji Hwan')
    
    return df
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
    print(API_KEY)

    # Break game IDs into chunks of 5
    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]
    print(game_id_chunks)
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
        print(all_data)
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
    todays_ids = get_todays_game_ids_v3(api_key,league='MLB')
    todays_ids = list(todays_ids.keys())
    pdf = get_pitcher_data(todays_ids)
    df = fetch_game_data2(todays_ids)
    print(df.head())
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

    # Ensure 'triple' column is present in the DataFrame
    if 'triple' not in df.columns:
        df['triple'] = np.nan  # or df['triple'] = 0 if you prefer a default value of 0

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
    todays_ids = get_todays_game_ids_v3(api_key=os.environ.get('ODDSJAM_API_KEY'),league='MLB')
    todays_ids = list(todays_ids.keys())

    pdf = get_pitcher_data(todays_ids)
    df = fetch_game_data2(todays_ids)
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

    # Ensure 'triple' column is present in the DataFrame
    if 'triple' not in df.columns:
        df['triple'] = np.nan  # or df['triple'] = 0 if you prefer a default value of 0

    new_column_order = [
        'team', 'Name', 'UTIL/CPT', 'cost', 'P','C','1B', '2B', '3B', 'SS', 'OF', 'blank2', 'blank3', 
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

    # Current date and time
    today = datetime.now(timezone.utc)

    # Define the start of the NFL season
    nfl_season_start = datetime(2024, 9, 5, tzinfo=timezone.utc)  # Adjust for actual season start

    # Determine start date based on current day and NFL season start
    if today < nfl_season_start:
        # If today is before the NFL season starts, start date should be the NFL season start
        start_date = nfl_season_start
    else:
        # If today is on or after the NFL season start, start date is today
        start_date = today

    # Calculate the end date for the current NFL week (Tuesday morning after Monday night)
    days_until_next_monday = (7 - start_date.weekday() + 0) % 7  # Calculate days until next Monday
    next_monday = start_date + timedelta(days=days_until_next_monday)  # Get the date of the next Monday
    
    # Set end time to 5:59 AM UTC on Tuesday morning (after Monday night)
    end_time_utc = (next_monday + timedelta(days=1)).replace(hour=5, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)

    # Parameters
    params = {
        "key": os.environ.get('ODDSJAM_API_KEY'),
        "sport": "football",
        "league": "NFL",
        "start_date_after": start_date.isoformat(),
        "start_date_before": end_time_utc.isoformat()
    }
    print(start_date.isoformat(), end_time_utc.isoformat())

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
def NFL_fetch_player_data(game_ids):
    # List of desired markets for players
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
            'is_main': 'true'
        }

        response = requests.get(URL, params=params)

        if response.status_code == 200:
            all_data.extend(response.json()['data'])
        else:
            print(f"Error {response.status_code}: {response.text}")

    json_data = all_data
    # Define DataFrame columns
    desired_columns = [
        "game_id",  # Add game_id for joining
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

    # Map values from JSON data to DataFrame
    for entry in json_data:
        game_id = entry['id']  # Extract game_id for each entry
        for bet in entry['odds']:
            player_name = bet['selection']
            market_name = bet['market_name']
            selection_line = bet.get('selection_line', '')
            selection_points = bet.get('selection_points', '')
            price = bet['price']

            # Determine the appropriate column based on market name and selection line
            if "Player Passing Touchdowns" in market_name and selection_line == "over":
                column_name = "# of pass tds"
                value = selection_points
                if player_name in df['name'].values:
                    df.loc[df['name'] == player_name, column_name] = value
                    df.loc[df['name'] == player_name, "odds on over pass tds"] = price
                else:
                    new_row = pd.DataFrame([{'game_id': game_id, 'name': player_name, column_name: value, "odds on over pass tds": price}])
                    df = pd.concat([df, new_row], ignore_index=True)
            
            elif "Player Interceptions" in market_name and selection_line == "over":
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
                if player_name in df['name'].values:
                    df.loc[df['name'] == player_name, column_name] = value
                    df.loc[df['name'] == player_name, "odds on pass receptions"] = price
                else:
                    new_row = pd.DataFrame([{'game_id': game_id, 'name': player_name, column_name: value, "odds on pass receptions": price}])
                    df = pd.concat([df, new_row], ignore_index=True)
            elif market_name == "Player Kicking Points" and selection_line == "over":
                column_name = "kicker points line"
                value = selection_points
                if player_name in df['name'].values:
                    df.loc[df['name'] == player_name, column_name] = value
                    df.loc[df['name'] == player_name, "odds on kicker points"] = price
                else:
                    new_row = pd.DataFrame([{'game_id': game_id, 'name': player_name, column_name: value, "odds on kicker points": price}])
                    df = pd.concat([df, new_row], ignore_index=True)
            else:
                continue

            # Update the DataFrame with the extracted values (excluding special cases)
            if market_name not in ["Player Passing Touchdowns", "Player Receptions", "Player Kicking Points"]:
                if player_name not in df['name'].values:
                    new_row = pd.DataFrame({'game_id': [game_id], 'name': [player_name], column_name: [value]})
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    df.loc[df['name'] == player_name, column_name] = value
    

    return df
def fetch_game_data(game_ids):
    URL = "https://api-external.oddsjam.com/api/v2/game-odds"
    API_KEY = os.environ.get('ODDSJAM_API_KEY')

    # Break game IDs into chunks of 5
    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]

    all_data = []

    # Fetch game-level "Total Points" market data
    for chunk in game_id_chunks:
        params = {
            'key': API_KEY,
            'sportsbook': ['DraftKings'],
            'game_id': chunk,
            'market_name': ['Total Points'],
            'is_main': 'true'
        }

        response = requests.get(URL, params=params)

        if response.status_code == 200:
            all_data.extend(response.json()['data'])
        else:
            print(f"Error {response.status_code}: {response.text}")

    # Create DataFrame for game-level data
    game_data = []
    for entry in all_data:
        game_id = entry['id']
        for bet in entry['odds']:
            if bet['market_name'] == "Total Points" and bet['is_main']:
                game_data.append({
                    'game_id': game_id,
                    'total points over line': bet['bet_points']
                })

    game_df = pd.DataFrame(game_data)
    game_df = game_df.drop_duplicates()
    
    return game_df

def nfl_fetch_combined_data(game_ids):
    player_df = NFL_fetch_player_data(game_ids)
    game_df = fetch_game_data(game_ids)
    
    # Merge both DataFrames on 'game_id'
    combined_df = pd.merge(player_df, game_df, on='game_id', how='left')
        # Sort by 'game_id' to ensure correct filling
    combined_df.sort_values(by='game_id', inplace=True)
    
    
    # Forward fill to ensure game-level data is populated across all rows with the same game_id
    #combined_df[['total points over line']] = combined_df[['total points over line']].fillna(method='ffill')
    combined_df.rename({'total points over line':'o/u score'},axis=1,inplace=True)
    # Forward fill and then backward fill the 'o/u score' column
    combined_df['o/u score'] = combined_df['o/u score'].fillna(method='ffill').fillna(method='bfill')

    return combined_df

def optomized_nfl_pulls(file_path):
    game_ids = get_nfl_game_ids()  # Replace with actual game IDs
    Oddsdf = nfl_fetch_combined_data(game_ids)
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
           'kicker points line', 'odds on kicker points',"o/u score"]]

    Fdf.rename({'Team':'team','Nickname':'name','Roster Position':'position','Salary':'cost'},axis=1,inplace=True)
    Fdf['blank'] = np.nan
    Fdf = Fdf[['# of pass tds', 'odds on over pass tds','passing yds line',
           'odds on over interception', 'odds on td scored', 'receiving yds line',
           'rush yds line', 'number of pass receptions', 'odds on pass receptions',
           'kicker points line', 'odds on kicker points',"o/u score","blank",'team','name','position',
           'cost']]
    return Fdf

def optimized_nfl_pulls_dk(file_path):
    todays_ids = get_nfl_game_ids()  # Replace with actual game IDs
    Oddsdf = nfl_fetch_combined_data(todays_ids)
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
               'kicker points line', 'odds on kicker points','o/u score']]

    df.rename({'TeamAbbrev':'team','Name':'name','Roster Position':'position','Salary':'cost'},axis=1,inplace=True)
    df['blank'] = np.nan
    df = df[['# of pass tds', 'odds on over pass tds','passing yds line',
           'odds on over interception', 'odds on td scored', 'receiving yds line',
           'rush yds line', 'number of pass receptions', 'odds on pass receptions',
           'kicker points line', 'odds on kicker points',"o/u score","blank",'team','name','position',
           'cost']]

    return df


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
def get_current_week_pga_tournament_name():
    API_ENDPOINT = "https://api.opticodds.com/api/v3/futures"
    PARAMS = {
            "key": os.environ.get('ODDSJAM_API_KEY'),
            "sport": "golf",
            "league": "PGA"
        }
    response = requests.get(API_ENDPOINT, params=PARAMS)
    if response.status_code == 200:
        futures_data = response.json()["data"]
        # Filter for likely men's PGA futures
    else:
        futures_data = []

    today = datetime.utcnow()
    start_of_week = today - timedelta(days=today.weekday())  # Monday
    end_of_week = start_of_week + timedelta(days=6)          # Sunday

    for item in futures_data:
        if item.get('league', {}).get('id') != 'pga':
            continue

        start_date = datetime.fromisoformat(item['start_date'].replace('Z', '+00:00'))
        if start_of_week.date() <= start_date.date() <= end_of_week.date():
            return item['tournament']['name']  # or item['name'] if you want "Top 10 Finish" etc.

    return None

def get_to_make_the_cut_odds(MakeCutID):
    API_ENDPOINT = "https://api.opticodds.com/api/v3/futures/odds"
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
        #print(odds_data)
    else:
        x=0
    odds_data = odds_data['data'][0]['odds']

    # Reformatting the data into two columns: 'To Make the Cut' and 'To Miss the Cut'
    formatted_data = {}
    for item in odds_data:
        # Extract player name and whether it's for making or missing the cut
        player_name = item['selection']
        make_or_miss = item['selection_line'].split()[-1]

        # Initialize the player entry if not already present
        if player_name not in formatted_data:
            formatted_data[player_name] = {'To Make the Cut': None, 'To Miss the Cut': None}

        # Assign the odds to the appropriate category
        if make_or_miss == 'yes':
            formatted_data[player_name]['To Make the Cut'] = item['price']
        elif make_or_miss == 'no':
            formatted_data[player_name]['To Miss the Cut'] = item['price']

    # Displaying a portion of the formatted data for brevity
    df = pd.DataFrame.from_dict(formatted_data, orient='index')

    # Setting the golfer's name as the index
    df.index.name = 'Golfer_name'
    df.reset_index(inplace=True)
    df.rename({'Golfer_name':'name'},axis=1,inplace=True)

    return df

def get_pga_futures(tournament_name):
    API_ENDPOINT = "https://api.opticodds.com/api/v3/futures"
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
    API_ENDPOINT = "https://api.opticodds.com/api/v3/futures/odds"
    all_data = []

    def chunk_list(lst, size=5):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    for batch in chunk_list(future_ids):
        PARAMS = {
            "key": os.environ.get('ODDSJAM_API_KEY'),
            "sport": "golf",
            "league": "PGA",
            "sportsbook": "DraftKings",
            "future_id": batch
        }

        response = requests.get(API_ENDPOINT, params=PARAMS)
        if response.status_code != 200:
            continue

        odds_data = response.json().get("data", [])
        print(odds_data)
        all_data.extend(odds_data)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.explode('odds')

    df['Golfer_name'] = df['odds'].apply(lambda x: x.get('selection'))
    df['price'] = df['odds'].apply(lambda x: x.get('price'))
    df['market_name'] = df['name']

    df = df[['Golfer_name', 'market_name', 'price']]

    pivot_df = df.pivot(index='Golfer_name', columns='market_name', values='price')
    pivot_df.columns.name = None
    pivot_df.reset_index(inplace=True)

    
    return pivot_df
def process_golf_data(file_path,tournament='Truist Championship 2025'):
    pga_futures = get_pga_futures(tournament_name=tournament)

    future_ids = [future["id"] for future in pga_futures]
    future_names = [future["name"] for future in pga_futures]

    keywords = ['winner',"top_10", "top_5", "top_20", "top_30", "top_40",'make_the_cut']

    filtered_futures = [s for s in future_ids if any(keyword in s for keyword in keywords) and "first_round" not in s]
    make_cut_ids = [s for s in filtered_futures if "make_the_cut" in s]
    filtered_futures = [s for s in filtered_futures if "make_the_cut" not in s]
    print(filtered_futures)
    #print(make_cut_ids)
    if make_cut_ids:
        make_cut_id = make_cut_ids[0]
        print(make_cut_id)# assuming you want the first match
    else:
        make_cut_id = None  # or handle the case where there is no match
    odds_data = get_draftkings_odds(future_ids=filtered_futures)
    print(odds_data)
    odds_data.reset_index(inplace=True)
    odds_data.rename({'Golfer_name':'name'},axis=1,inplace=True)
    cut_data = get_to_make_the_cut_odds(MakeCutID=make_cut_id)
    print(cut_data)
    odds_data = odds_data.merge(cut_data,how='left',left_on='name',right_on='name')
    print(odds_data)
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
    #print(fdDF)

        # Merge dataframes on the keys
    merged = fdDF.merge(odds_data.rename(columns={'name': 'matched_key'}), on='matched_key', how='left')

        # Drop the matched_key column for cleaner output
    merged.drop(columns=['matched_key'], inplace=True)

    columns_to_transform = ['To Make the Cut','top 5', 'top 10', 'top 20', 'top 30', 'top 40','winner']

    for col in columns_to_transform:
        if col in merged.columns:
            for idx, value in merged[col].items():
                merged.at[idx, col] = transform_odds(value)
    final_columns = ['name', 'cost', 'winner','To Make the Cut', 'top 5', 'top 10', 'top 20', 'top 30', 'top 40']
    existing_columns = [col for col in final_columns if col in merged.columns]
    final = merged[existing_columns]
    return final

def dk_process_golf_data(file_path,tournament='Truist Championship 2025'):
    pga_futures = get_pga_futures(tournament_name=tournament)

    future_ids = [future["id"] for future in pga_futures]
    future_names = [future["name"] for future in pga_futures]

    keywords = ['winner',"top_10", "top_5", "top_20", "top_30", "top_40",'make_the_cut']

    filtered_futures = [s for s in future_ids if any(keyword in s for keyword in keywords) and "first_round" not in s]
    make_cut_ids = [s for s in filtered_futures if "make_the_cut" in s]
    filtered_futures = [s for s in filtered_futures if "make_the_cut" not in s]
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

    columns_to_transform = ['To Make the Cut','top 5', 'top 10', 'top 20', 'top 30', 'top 40','winner']

    for col in columns_to_transform:
        if col in merged.columns:
            for idx, value in merged[col].items():
                merged.at[idx, col] = transform_odds(value)
    final_columns = ['name', 'cost', 'winner', 'To Make the Cut','top 5', 'top 10', 'top 20', 'top 30', 'top 40']
    existing_columns = [col for col in final_columns if col in merged.columns]
    final = merged[existing_columns]
    return final
## NBA
def nba_get_todays_game_ids():
    # Initialize the client (assuming you have a client setup)
    Client = OddsJamClient(os.environ.get('ODDSJAM_API_KEY'))
    Client.UseV2()
    
    # Get games for the NBA league
    GamesResponse = Client.GetGames(league='nba')
    Games = GamesResponse.Games
    
    # Extract relevant game data
    games_data = [{'game_id': game.id, 'start_date': game.start_date} for game in Games]
    games_df = pd.DataFrame(games_data)
    
    # Remove timezone info and convert start_date to datetime
    games_df['start_date'] = games_df['start_date'].astype(str).str[:-6]
    games_df['start_date'] = pd.to_datetime(games_df['start_date'])
    
    # Define today's and tomorrow's dates
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    # Filter for games happening today or tomorrow
    selected_games = games_df[games_df['start_date'].dt.date.isin([today, tomorrow])]
    
    # Return the list of game IDs for today and tomorrow
    return selected_games['game_id'].tolist()
def nhl_get_todays_game_ids():
    # Initialize the client (assuming you have a client setup)
    Client = OddsJamClient(os.environ.get('ODDSJAM_API_KEY'))
    Client.UseV2()
    # Get games for the league
    GamesResponse = Client.GetGames(league='nhl')
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

def get_nba_odds_for_dfs_roster(dfs_players, game_ids):
    desired_markets = ['Player Points', 'Player Rebounds', 'Player Assists', 
                       'Player Steals + Blocks', 'Player Turnovers', 
                       'Player Double Double', 'Player Triple Double']
    
    URL = "https://api-external.oddsjam.com/api/v2/game-odds"
    API_KEY = os.environ.get('ODDSJAM_API_KEY')
    sportsbooks_priority = ['DraftKings', 'Pinnacle', 'FanDuel', 'BetOnline']  # Order of priority

    # Break game IDs into chunks
    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]
    all_data = []

    # For each chunk, fetch odds
    for chunk in game_id_chunks:
        for sportsbook in sportsbooks_priority:
            params = {
                'key': API_KEY,
                'sportsbook': sportsbook,
                'game_id': chunk,
                'market_name': desired_markets,
                'is_main': 'true'
            }
            response = requests.get(URL, params=params)

            if response.status_code == 200:
                odds_data = response.json()['data']
                all_data.extend(odds_data)
                if odds_data:
                    break  # Exit loop if data is retrieved for this chunk
            else:
                print(f"Error {response.status_code}: {response.text}")

    # Process the data into a DataFrame
    desired_columns = ["name", "Player Points", "Player Points odds", "Player Rebounds", 
                       "Player Rebounds odds", "Player Assists", "Player Assists odds", 
                       "Player Steals + Blocks", "Player Steals + Blocks odds", 
                       "Player Turnovers", "Player Turnovers odds", 
                       "Player Double Double odds", "Player Triple Double odds"]
    df = pd.DataFrame(columns=desired_columns)

    # Filter odds to include only DFS roster players
    for entry in all_data:
        for odds_entry in entry['odds']:
            player_name = odds_entry['selection']
            if player_name not in dfs_players:
                continue  # Skip players not in DFS list

            market_name = odds_entry['market_name']
            selection_points = odds_entry['selection_points']
            price = odds_entry['price']
            selection_line = odds_entry['selection_line']

            # Check if player already exists in the DataFrame
            player_index = df.index[df['name'] == player_name]
            if not player_index.empty:
                player_index = player_index[0]
            else:
                # Add new player data if it doesn't exist
                new_row = pd.DataFrame({'name': [player_name]})
                df = pd.concat([df, new_row], ignore_index=True)
                player_index = df.index[df['name'] == player_name][0]

            # Map data to DataFrame based on market name and selection line
            if pd.isna(df.at[player_index, f'{market_name} odds']):
                if market_name in ["Player Double Double", "Player Triple Double"]:
                    df.loc[player_index, f'{market_name} odds'] = price
                elif selection_line == 'over':
                    df.loc[player_index, f'{market_name}'] = selection_points
                    df.loc[player_index, f'{market_name} odds'] = price

    return df


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
    
    #rgdf = pd.read_csv(file_path2)
    #rgdf = rgdf[['PLAYER','FPTS']]
    #rgdf = rgdf[rgdf['FPTS'].notna()]
    #merged_df = final.merge(rgdf, left_on='player', right_on='PLAYER', how='left')

    # Define the columns to check for NaN
    numerical_columns = ['pts','pts odds','rebounds','rebounds odds','assists','assists odds',
        'steals and blocks','steals and blocks odds','turnovers','turnovers odds','DD odds','TD odds']
    # Create a mask where all numerical columns are NaN
    #all_na_mask = final[numerical_columns].isna().all(axis=1)
    # Replace 'pts' only where all numerical columns are NaN
    #final.loc[all_na_mask, 'pts'] = final.loc[all_na_mask, 'FPTS']
    # Drop the unnecessary columns
    #FdDF = final.drop(columns=['PLAYER', 'FPTS'])
    return final


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

## New NBA
def get_nba_odds_for_dfs_roster(dfs_players, game_ids):
    desired_markets = ['Player Points', 'Player Rebounds', 'Player Assists', 
                       'Player Steals + Blocks', 'Player Turnovers', 
                       'Player Double Double', 'Player Triple Double']
    
    URL = "https://api-external.oddsjam.com/api/v2/game-odds"
    API_KEY = os.environ.get('ODDSJAM_API_KEY')
    sportsbooks_priority = ["DraftKings", "Pinnacle", "FanDuel", "BetOnline", "Bet365", "Caesars",
                            'BetRivers', 'Unibet', 'Betfred', "Betfair Exchange", "Fliff",
                            "PointsBet", "BetMGM", "Circa Sports", "William Hill"]

    # Helper function to match player names using fuzzy matching
    def find_best_match(name, players, threshold=90):
        match, score = process.extractOne(name, players)
        return match if score >= threshold else None

    # Break game IDs into chunks and fetch odds data
    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]
    all_data = []
    for chunk in game_id_chunks:
        for sportsbook in sportsbooks_priority:
            params = {
                'key': API_KEY,
                'sportsbook': sportsbook,
                'game_id': chunk,
                'market_name': desired_markets,
                'is_main': 'true'
            }
            response = requests.get(URL, params=params)
            print(f"Fetching data from {sportsbook}...")  # Debug statement

            if response.status_code == 200:
                odds_data = response.json().get('data', [])
                print(f"{sportsbook} data: {len(odds_data)} entries retrieved.")  # Debug
                
                for entry in odds_data:
                    for odds_entry in entry['odds']:
                        odds_entry['source'] = sportsbook  # Track the source sportsbook
                all_data.extend(odds_data)
            else:
                print(f"Error {response.status_code}: {response.text}")

    # Initialize the DataFrame with desired columns
    desired_columns = ["name", "Player Points", "Player Points odds", "Player Points source",
                       "Player Rebounds", "Player Rebounds odds", "Player Rebounds source",
                       "Player Assists", "Player Assists odds", "Player Assists source",
                       "Player Steals + Blocks", "Player Steals + Blocks odds", "Player Steals + Blocks source",
                       "Player Turnovers", "Player Turnovers odds", "Player Turnovers source",
                       "Player Double Double odds", "Player Double Double source",
                       "Player Triple Double odds", "Player Triple Double source"]
    df = pd.DataFrame(columns=desired_columns)

    # Process and map data for DFS players
    for entry in all_data:
        for odds_entry in entry['odds']:
            player_name = odds_entry['selection']
            matched_player_name = find_best_match(player_name, dfs_players)

            if not matched_player_name:
                continue  # Skip if no close match is found

            market_name = odds_entry['market_name']
            selection_points = odds_entry['selection_points']
            price = odds_entry['price']
            selection_line = odds_entry['selection_line']
            source = odds_entry['source']

            # Add or find player in DataFrame
            if matched_player_name not in df['name'].values:
                new_row = pd.DataFrame({'name': [matched_player_name]})
                df = pd.concat([df, new_row], ignore_index=True)

            # Retrieve player index
            player_index = df.index[df['name'] == matched_player_name][0]

            # Only update if column is empty or current source is higher priority
            if pd.isna(df.at[player_index, f'{market_name} odds']) or \
               sportsbooks_priority.index(source) < sportsbooks_priority.index(df.at[player_index, f'{market_name} source']):
                if market_name in ["Player Double Double", "Player Triple Double"]:
                    df.loc[player_index, f'{market_name} odds'] = price
                    df.loc[player_index, f'{market_name} source'] = source
                elif selection_line == 'over':
                    df.loc[player_index, f'{market_name}'] = selection_points
                    df.loc[player_index, f'{market_name} odds'] = price
                    df.loc[player_index, f'{market_name} source'] = source

    # Display the percentage of data by source per market for debugging
    source_columns = [col for col in df.columns if 'source' in col]
    for col in source_columns:
        source_counts = df[col].value_counts(normalize=True) * 100
        print(f"Percentage of data from each sportsbook for {col.replace(' source', '')}:")
        print(source_counts)
        print()

    return df

def fd_nba_process_data(file_path):
    # Load game IDs and FanDuel DFS player list
    ids = nba_get_todays_game_ids()
    print(f"Today's Game IDs: {ids}")  # Debugging output
    fdDF = pd.read_csv(file_path)

    # Convert player names to a list for passing to the odds function without cleaning suffixes
    dfs_players = fdDF['Nickname'].tolist()

    # Get NBA odds for DFS roster, relying on fuzzy matching for name discrepancies
    df = get_nba_odds_for_dfs_roster(dfs_players, ids)

    # Select relevant columns from FanDuel data and merge with odds data
    fdDF = fdDF[['Nickname', 'Roster Position', 'Salary', 'Team']]
    merged_df = fdDF.merge(df, left_on='Nickname', right_on='name', how='left')

    # Apply transformations on odds with error handling
    columns_to_transform = ['Player Points odds', 'Player Rebounds odds', 'Player Assists odds',
                            'Player Steals + Blocks odds', 'Player Turnovers odds', 
                            "Player Double Double odds", "Player Triple Double odds"]

    for col in columns_to_transform:
        merged_df[col] = merged_df[col].apply(lambda x: transform_odds(x) if pd.notna(x) else x)

    # Rename columns for the final DataFrame
    merged_df.rename(columns={
        'Team': 'team',
        'Nickname': 'player',
        'Roster Position': 'position',
        'Salary': 'cost',
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
    }, inplace=True)

    # Select final output columns
    final_columns = ['team', 'player', 'position', 'cost', 'pts', 'pts odds', 
                     'rebounds', 'rebounds odds', 'assists', 'assists odds', 
                     'steals and blocks', 'steals and blocks odds', 
                     'turnovers', 'turnovers odds', 'DD odds', 'TD odds']
    final_df = merged_df[final_columns]

    return final_df
def dk_nba_process_data(file_path):
    # Load game IDs and FanDuel DFS player list
    ids = nba_get_todays_game_ids()
    print(f"Today's Game IDs: {ids}")  # Debugging output
    dkDF = pd.read_csv(file_path)

    # Convert player names to a list for passing to the odds function without cleaning suffixes
    dfs_players = dkDF['Name'].tolist()

    # Get NBA odds for DFS roster, relying on fuzzy matching for name discrepancies
    df = get_nba_odds_for_dfs_roster(dfs_players, ids)

    # Select relevant columns from FanDuel data and merge with odds data
    dkDF = dkDF[['Name', 'Roster Position', 'Salary', 'TeamAbbrev']]
    merged_df = dkDF.merge(df, left_on='Name', right_on='name', how='left')

    # Apply transformations on odds with error handling
    columns_to_transform = ['Player Points odds', 'Player Rebounds odds', 'Player Assists odds',
                            'Player Steals + Blocks odds', 'Player Turnovers odds', 
                            "Player Double Double odds", "Player Triple Double odds"]

    for col in columns_to_transform:
        merged_df[col] = merged_df[col].apply(lambda x: transform_odds(x) if pd.notna(x) else x)

    # Rename columns for the final DataFrame
    merged_df.rename(columns={
        'TeamAbbrev': 'team',
        'Name': 'player',
        'Roster Position': 'position',
        'Salary': 'cost',
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
    }, inplace=True)
    merged_df['team'] = merged_df['team'].replace({'PHX': 'PHO'})
    # Select final output columns
    final_columns = ['team', 'player', 'position', 'cost', 'pts', 'pts odds', 
                     'rebounds', 'rebounds odds', 'assists', 'assists odds', 
                     'steals and blocks', 'steals and blocks odds', 
                     'turnovers', 'turnovers odds', 'DD odds', 'TD odds']
    final_df = merged_df[final_columns]

    return final_df


# Display the corrected results for user
# Define a function to clean the data as specified
def clean_and_prepare_data(df):
    # Select relevant columns without filtering any rows
    cleaned_df = df[['PLAYER', 'TEAM','PTS','AST','TRB','BK','ST','TO']].copy()
    
    # Rename columns for clarity and consistency
    cleaned_df.rename(columns={
        'PLAYER': 'player',
        'TEAM': 'team',
        'PTS': 'points',
        'TRB': 'rebounds',
        'AST': 'assists',
        'ST': 'steals',
        'BK': 'blocks',
        'TO':'turnovers'
    }, inplace=True)
    # Remove any quotes around data (e.g., if data was parsed as strings)
    #cleaned_df = cleaned_df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
    
    # Fill missing values with zero to retain all rows
    cleaned_df.fillna(0, inplace=True)
    
    # Predict odds for getting a double-double and a triple-double
    cleaned_df['double_double_odds'] = (
        (cleaned_df[['points', 'rebounds', 'assists']] >= 10).sum(axis=1) >= 2
    ).astype(int)  # Predict odds as binary (1 if likely to get a double-double, else 0)
    
    cleaned_df['triple_double_odds'] = (
        (cleaned_df[['points', 'rebounds', 'assists']] >= 10).sum(axis=1) >= 3
    ).astype(int)  # Predict odds as binary (1 if likely to get a triple-double, else 0)
    
    return cleaned_df
def probability_to_american_odds(probability):
    """
    Converts a probability (0 to 1) into American betting odds.
    
    Parameters:
    - probability (float): The probability value (e.g., 0.75 for 75%).
    
    Returns:
    - int: American odds (positive or negative).
    """
    if probability <= 0 or probability >= 1:
        return None  # Handle edge cases or probabilities out of bounds
    
    if probability >= 0.5:
        # Negative odds for probability >= 50%
        return int(-100 * probability / (1 - probability))
    else:
        # Positive odds for probability < 50%
        return int(100 * (1 - probability) / probability)


def simulate_dd_td_probabilities_corrected(df, n_simulations=10000, std_dev_factor=0.25):
    """
    Corrected version of the Monte Carlo simulation to calculate the probability of hitting a double-double or 
    triple-double for each player.
    """
    results = []
    
    for _, row in df.iterrows():
        # Player information and stats
        player = row['player']
        points_avg = row['points']
        rebounds_avg = row['rebounds']
        assists_avg = row['assists']
        s_b = row['steals'] + row['blocks']
        
        # Using higher variability (25% of average) to simulate realistic game-to-game fluctuations
        points_std = points_avg * std_dev_factor
        rebounds_std = rebounds_avg * std_dev_factor
        assists_std = assists_avg * std_dev_factor
        
        # Generate random values for points, rebounds, and assists over multiple simulations
        points_sim = np.random.normal(points_avg, points_std, n_simulations)
        rebounds_sim = np.random.normal(rebounds_avg, rebounds_std, n_simulations)
        assists_sim = np.random.normal(assists_avg, assists_std, n_simulations)
        
        # Calculate double-double and triple-double hits based on correct criteria
        double_double_hits = np.sum((points_sim >= 10).astype(int) + (rebounds_sim >= 10).astype(int) + (assists_sim >= 10).astype(int) >= 2)
        triple_double_hits = np.sum((points_sim >= 10).astype(int) + (rebounds_sim >= 10).astype(int) + (assists_sim >= 10).astype(int) >= 3)
        
        # Calculate probabilities
        double_double_prob = double_double_hits / n_simulations
        triple_double_prob = triple_double_hits / n_simulations
        
        # Append the results
        results.append({
            'player': player,
            'team': row['team'],
            'points': points_avg,
            'rebounds': rebounds_avg,
            'assists': assists_avg,
            's_b': s_b,
            'double_double_prob': double_double_prob,
            'triple_double_prob': triple_double_prob
        })
    
    # Create a DataFrame to store results
    results_df = pd.DataFrame(results)

    # Rename columns for consistency

    # Apply the probability to American odds conversion function
    results_df['double_double_odds'] = results_df['double_double_prob'].apply(probability_to_american_odds)
    results_df['triple_double_odds'] = results_df['triple_double_prob'].apply(probability_to_american_odds)
    results_df.rename(columns={'points': 'pts', 's_b': 'steals and blocks','double_double_odds':'DD odds',
                               'triple_double_odds':'TD odds'}, inplace=True)
    # Add default odds for other metrics
    results_df['pts odds'] = -115
    results_df['assists odds'] = -115
    results_df['rebounds odds'] = -115
    results_df['steals and blocks odds'] = -115
    columns_to_transform = ['pts odds', 'assists odds', 'rebounds odds',
                            'steals and blocks odds',  
                            "DD odds", "TD odds"]

    for col in columns_to_transform:
        results_df[col] = results_df[col].apply(lambda x: transform_odds(x) if pd.notna(x) else x)
    
    return results_df

def merge_odds_with_projections(df_odds, df_projections):
    """
    Merges df_odds and df_projections using fuzzy matching on 'player' column,
    filling missing values in df_odds with data from df_projections.
    """
    # Step 1: Create a fuzzy matching map between players in df_odds and df_projections
    player_map = {}
    for player in df_odds['player'].unique():
        match_result = process.extractOne(player, df_projections['player'].unique(), score_cutoff=80)
        if match_result:
            match, score = match_result
            player_map[player] = match
        else:
            # If no match found, set to None or skip
            player_map[player] = None

    # Apply the mapping to create a new 'matched_player' column in df_odds
    df_odds['matched_player'] = df_odds['player'].map(player_map)
    
    # Remove rows with no matches in 'matched_player'
    df_odds = df_odds[df_odds['matched_player'].notna()]

    # Step 2: Merge DataFrames on 'team' and 'matched_player' with suffixes to differentiate
    merged_df = df_odds.merge(df_projections, left_on=['team', 'matched_player'], right_on=['team', 'player'], how='left', suffixes=('', '_proj'))

    # Step 3: Columns to fill from projections only if they are missing in odds data
    columns_to_fill = ['pts', 'pts odds', 'rebounds', 'rebounds odds', 'assists', 'assists odds', 
                       'steals and blocks', 'steals and blocks odds', 'turnovers', 'turnovers odds', 
                       'DD odds', 'TD odds']

    # Use combine_first to fill missing values in df_odds columns with values from df_projections
    for col in columns_to_fill:
        if col in merged_df.columns and f"{col}_proj" in merged_df.columns:
            merged_df[col] = merged_df[col].combine_first(merged_df[f"{col}_proj"])

    # Step 4: Drop the extra columns created from projections and 'matched_player'
    merged_df.drop([f"{col}_proj" for col in columns_to_fill if f"{col}_proj" in merged_df.columns], axis=1, inplace=True)
    merged_df.drop(columns=['matched_player'], inplace=True)
    
    # Step 5: Keep only the specified columns
    final_columns = ['team', 'player', 'position', 'cost', 'pts', 'pts odds', 'rebounds', 'rebounds odds', 
                     'assists', 'assists odds', 'steals and blocks', 'steals and blocks odds', 'turnovers', 
                     'turnovers odds', 'DD odds', 'TD odds']
    
    return merged_df[final_columns]

## EV Betting
def get_todays_game_ids_v3(api_key, league, is_live='false'):
    endpoint = "https://api.opticodds.com/api/v3/fixtures"
    if league == 'MLB':  # Current date and time
        today = datetime.now(timezone.utc)
        future_date = today + timedelta(hours=72)
    else:  # Current date and time
        today = datetime.now(timezone.utc)
        future_date = today + timedelta(hours=48*4)

    # Parameters
    params = {
        "key": api_key,
        "league": league,
        "start_date_after": today,
        "start_date_before": future_date
    }

    # Make the request
    response = requests.get(endpoint, params=params)

    # Check response
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    api_response = response.json()
    game_data = {
        game['id']: f"{game['home_team_display']} vs {game['away_team_display']}"
        for game in api_response['data']
    }
    return game_data
# Fetch game data dynamically and filter based on player or game markets
def fetch_game_data_ev(game_ids, api_key, market_type='player', sport='baseball', league='MLB', sportsbooks=None, include_player_name=True, is_live='false'):
    # Validate inputs
    if sportsbooks is None:
        sportsbooks = ['Pinnacle', 'FanDuel', 'DraftKings']

    # Fetch game names
    game_data_dict = get_todays_game_ids_v3(api_key, league, is_live=is_live)
    if not game_data_dict:
        print("No game data found.")
        return pd.DataFrame()
    if league == 'NFL':
        markets = [
    "Player Passing Touchdowns",
    "Player Interceptions",
    "Player Receptions",
    "Player Longest Passing Completion",
    "Player Passing Completions",
    "Player Receiving Yards",
    "Player Passing Attempts",
    "Player Kicking Points",
    "Player Passing Yards",
    "Player Touchdowns",
    "Player Rushing Yards",
    "Player Longest Reception"]
    elif league == 'NBA':
        markets = ['Player Assists',
       'Player Double Double', 'Player Made Threes', 'Player Points',
       'Player Points + Rebounds + Assists', 'Player Rebounds',
       'Player Blocks', 'Player Points + Assists',
       'Player Points + Rebounds', 'Player Rebounds + Assists',
       'Player Steals', 'Player Steals + Blocks', 'Player Turnovers',
       '1st Quarter Player Assists', '1st Quarter Player Points',
       '1st Quarter Player Rebounds', 'Player Triple Double']
    elif league == 'MLB':
        markets = ["Player Hits Allowed",
    "Player Strikeouts",
    "Player Earned Runs",
    "Player Home Runs Yes/No",
    "Player Home Runs",
    "Player Bases",
    "Player Outs"]
    url = "https://api.opticodds.com/api/v3/fixtures/odds"
    all_data = []  # Collect all data across sportsbooks and chunks

    for chunk in [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]:
        for sportsbook in sportsbooks:
            params = {
                'key': api_key,
                'sportsbook': sportsbook,
                'fixture_id': chunk,
                'market_name': markets
            }
            if is_live != 'false':
                params['status'] = 'live'

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json().get('data', [])
                for game_data in data:
                    # Add sportsbook info to each record
                    for item in game_data.get('odds', []):
                        all_data.append({
                            'Game ID': game_data.get('id', 'Unknown'),
                            'Game Name': game_data_dict.get(game_data.get('id', 'Unknown'), 'Unknown Game'),
                            'Bet Name': item.get('name', None),
                            'Market Name': item.get('market', ''),
                            'Sportsbook': sportsbook,
                            'line': item.get('points', None),
                            'Odds': item.get('price', None),
                            **({'Player Name': item.get('selection', 'Unknown')} if include_player_name and market_type == 'player' else {})
                        })
            else:
                print(f"Error fetching data for sportsbook {sportsbook}: {response.status_code} - {response.text}")

    return pd.DataFrame(all_data)

def get_player_ev_bets(api_key, sport, league, sportsbook='Caesars', is_live='false'):
    # Retrieve game IDs for today's games
    game_ids = get_todays_game_ids_v3(api_key, league, is_live=is_live)
    print(game_ids)
    # Fetch player props
    player_props_df = fetch_game_data_ev(
        list(game_ids.keys()), api_key, market_type='player', sport=sport, league=league,
        sportsbooks=['Pinnacle','Blue Book','DraftKings'], is_live=is_live
    )
    return player_props_df
def process_player_props(player_props_df):
    """
    Process the player props dataframe to pair Over and Under bets.
    """
    # Separate Over and Under bets
    over_bets = player_props_df[
        ['Sport',"Game ID", "Game Name","Market Name", "line", "Bet Name", "Player Name", "Sportsbook", "Odds"]
    ][player_props_df["Bet Name"].str.contains("Over", na=False)]

    under_bets = player_props_df[
        ['Sport',"Game ID","Game Name", "Market Name", "line", "Bet Name", "Player Name", "Sportsbook", "Odds"]
    ][player_props_df["Bet Name"].str.contains("Under", na=False)]

    # Pair Over and Under bets by Game ID, Market Name, line, and Player Name
    paired_player_props = pd.merge(
        over_bets,
        under_bets,
        on=["Sport","Game ID","Game Name", "Market Name", "line", "Player Name", "Sportsbook"],
        suffixes=("_A", "_B")
    )
    
    return paired_player_props

@app.route('/EVBetting')
def ev_page():
    return render_template('EV_upload.html')
@app.route('/ev_process', methods=['POST'])
def ev_betting():
    sports_leagues = [
        {'sport': 'basketball', 'league': 'NBA'},
        {'sport':'baseball','league':'MLB'}
    ]
    combined_df = pd.DataFrame()

    # Loop through each sport and league
    for pair in sports_leagues:
        sport = pair['sport']
        league = pair['league']
        
        # Fetch the data using your API function
        pdf = get_player_ev_bets(api_key=os.environ.get('ODDSJAM_API_KEY'), sport=sport, league=league, sportsbook="DraftKings", is_live="false")
        print(pdf.columns)
        pdf = pdf[pdf['Market Name'].str.contains("Player")]  # Example filter
        pdf['Sport'] = sport
        combined_df = pd.concat([combined_df, pdf], ignore_index=True)

    paired_player_props = process_player_props(combined_df)
    pivot_df = paired_player_props.pivot_table(
        index=["Sport", "Game ID", "Game Name", "Market Name", "Player Name"],
        columns="Sportsbook",
        values=["Bet Name_A", "Odds_A", "line", "Bet Name_B", "Odds_B"],
        aggfunc="first"
    )
    print(pivot_df.head())

    # Flatten MultiIndex columns
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df.reset_index(inplace=True)

    # Reorganize and rename columns for final output
    pivot_df = pivot_df[['Sport', "Game Name", "Market Name", "Player Name",
                         'line_Pinnacle', 'Odds_A_Pinnacle', 'Odds_B_Pinnacle',
                         'line_DraftKings', 'Odds_A_DraftKings', 'Odds_B_DraftKings',
                         'line_Blue Book', 'Odds_A_Blue Book', 'Odds_B_Blue Book']]
    pivot_df.rename({'Odds_A_Pinnacle': 'Odds_Over_Pinnacle',
                     'Odds_A_DraftKings': 'Odds_Over_DraftKings',
                     'Odds_A_Blue Book': 'Odds_Over_FanDuel',
                     'Odds_B_Pinnacle': 'Odds_Under_Pinnacle',
                     'Odds_B_DraftKings': 'Odds_Under_DraftKings',
                     'Odds_B_Blue Book': 'Odds_Under_FanDuel',
                     'line_Blue Book': 'line_FanDuel'}, axis=1, inplace=True)

    # Save the file to disk in the UPLOAD_FOLDER
    output_filename = "EV_Betting_Data.xlsx"
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    pivot_df.insert(4, 'Blank_1', '')  # After Player Name
    pivot_df.insert(8, 'Blank_2', '')  # After Pinnacle columns
    pivot_df.insert(12, 'Blank_3', '')  # After FanDuel columns
    with pd.ExcelWriter(output_filepath, engine='xlsxwriter') as writer:
        pivot_df.to_excel(writer, index=False, sheet_name="Data")

        # Access the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets["Data"]

# Dynamically get pivot_df columns
        actual_columns = list(pivot_df.columns)

        # Create a new list with blank placeholders at specific positions
        formatted_headers = []
        for i, col in enumerate(actual_columns):
            formatted_headers.append(col)
        # Write headers dynamically including blank columns
        for col_num, header in enumerate(formatted_headers):
            worksheet.write(0, col_num, header)
        # Adjust column width for readability
        worksheet.set_column(0, len(formatted_headers), 15)

    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/')
def index():
    return render_template('updated_upload.html')
@app.route('/process_pick6', methods=['POST'])
def process_pick6():
    league = request.form['league']
    
    # Call the function to get the processed Pick 6 data
    final_df_filtered_output = pick_6_data(league)

    # Create separate DataFrames for each market category
    market_categories = final_df_filtered_output['Category'].unique()
    market_tables = {market: final_df_filtered_output[final_df_filtered_output['Category'] == market] for market in market_categories}

    # Create the Excel file with the full list and individual market tables
    output_excel_path = f"{league}_pick6_markets.xlsx"
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        # Assuming 'league' is passed in and determines if it's NFL or MLB
        if league == 'NFL':
            # Write the "All Markets" table
            final_df_filtered_output.to_excel(writer, sheet_name='Markets Overview', index=False, startrow=1)
            worksheet = writer.sheets['Markets Overview']
            worksheet.write(0, 0, 'All Markets')
            
            # Only include "Passing TDs" and "Receptions" tables for NFL
            row_position = len(final_df_filtered_output) + 4  # Start after "All Markets"
            
            if 'Player Passing Touchdowns' in market_tables:
                worksheet.write(row_position - 1, 0, 'Player Passing Touchdowns')  # Add title
                market_tables['Player Passing Touchdowns'].to_excel(writer, sheet_name='Markets Overview', index=False, startrow=row_position)
                row_position += len(market_tables['Player Passing Touchdowns']) + 4
            
            if 'Player Receptions' in market_tables:
                worksheet.write(row_position - 1, 0, 'Receptions')  # Add title
                market_tables['Player Receptions'].to_excel(writer, sheet_name='Markets Overview', index=False, startrow=row_position)
                row_position += len(market_tables['Player Receptions']) + 4
        elif league == 'MLB':
            # Handle the MLB case, writing all categories like before
            final_df_filtered_output.to_excel(writer, sheet_name='Markets Overview', index=False, startrow=1)
            worksheet = writer.sheets['Markets Overview']
            worksheet.write(0, 0, 'Full List')

            row_position = len(final_df_filtered_output) + 4  # Start after full list
            for market, df in market_tables.items():
                worksheet.write(row_position - 1, 0, market)  # Add the market title above each table
                df.to_excel(writer, sheet_name='Markets Overview', index=False, startrow=row_position)
                row_position += len(df) + 4  # Add space and title for the next table
        elif league == 'NBA':
            # Handle the MLB case, writing all categories like before
            final_df_filtered_output.to_excel(writer, sheet_name='Markets Overview', index=False, startrow=1)
            worksheet = writer.sheets['Markets Overview']
            worksheet.write(0, 0, 'Full List')

            row_position = len(final_df_filtered_output) + 4  # Start after full list
            for market, df in market_tables.items():
                worksheet.write(row_position - 1, 0, market)  # Add the market title above each table
                df.to_excel(writer, sheet_name='Markets Overview', index=False, startrow=row_position)
                row_position += len(df) + 4  # Add space and title for the next table
        elif league == 'NHL':
            # Handle the MLB case, writing all categories like before
            final_df_filtered_output.to_excel(writer, sheet_name='Markets Overview', index=False, startrow=1)
            worksheet = writer.sheets['Markets Overview']
            worksheet.write(0, 0, 'Full List')

            row_position = len(final_df_filtered_output) + 4  # Start after full list
            for market, df in market_tables.items():
                worksheet.write(row_position - 1, 0, market)  # Add the market title above each table
                df.to_excel(writer, sheet_name='Markets Overview', index=False, startrow=row_position)
                row_position += len(df) + 4  # Add space and title for the next table



    # Send the generated Excel file to the user
    return send_from_directory('.', output_excel_path, as_attachment=True)

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
    #db_proj = final_proj.copy()
    #db_proj['run_timestamp'] = pd.Timestamp.now()
    # Save the DataFrame with the timestamp to PostgreSQL using 'append'
    #db_proj.to_sql('mlb_fd_optimized', engine, if_exists='append', index=False)

    output_filename = "fanduel_optimized_mlb.csv"
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
    #final_proj.to_sql('mlb_fd_optimized', engine, if_exists='append', index=False)

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
    # Create a copy of the DataFrame to add a timestamp column for the database
    #db_proj = final_proj.copy()
    #db_proj['run_timestamp'] = pd.Timestamp.now()

    # Save the DataFrame with the timestamp to PostgreSQL using 'append'
    #db_proj.to_sql('mlb_dk_optimized', engine, if_exists='append', index=False)
    output_filename = "draftkings_optimized_mlb.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/process_nfl_fd_optimized', methods=['POST'])
def process_file_nfl_fd_optomized():
    if 'file' not in request.files:
        return 'No file uploaded!'
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = optomized_nfl_pulls(filepath)  # Assuming you're processing the DraftKings file the same way
    #db_proj = final_proj.copy()
    #db_proj['run_timestamp'] = pd.Timestamp.now()

    # Save the DataFrame with the timestamp to PostgreSQL using 'append'
    #db_proj.to_sql('nfl_fd_optimized', engine, if_exists='append', index=False)

    output_filename = "optimized_processed_output_nfl.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)
@app.route('/process_nfl_dk_optimized', methods=['POST'])
def process_file_nfl_dk_optomized():
    if 'file' not in request.files:
        return 'No file uploaded!'
    
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = optimized_nfl_pulls_dk(filepath)  # Assuming you're processing the DraftKings file the same way
    #final_proj.to_sql('nfl_dk_optimized', engine, if_exists='append', index=False)
    output_filename = "optimized_dk_processed_output_nfl.csv"
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
    tournament_name = get_current_week_pga_tournament_name()
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    final_proj = dk_process_golf_data(filepath,tournament=tournament_name) 

    output_filename = "dk_processed_output_pga.csv"
    final_proj.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], output_filename), index=False)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

@app.route('/NBA')
def nba_page():
    return render_template('nba_upload.html')

@app.route('/process_nba_fd', methods=['POST'])
def nba_process():
    # Check if both files are present in the upload
    if 'fd_salaries' not in request.files or 'nba_projections' not in request.files:
        return 'Both files (FD Salaries and NBA Projections) need to be uploaded!', 400

    # Save FanDuel salaries file to get file path
    fd_salaries_file = request.files['fd_salaries']
    fd_salaries_filename = fd_salaries_file.filename
    fd_salaries_filepath = os.path.join(app.config['UPLOAD_FOLDER'], fd_salaries_filename)
    fd_salaries_file.save(fd_salaries_filepath)

    # Load NBA projections directly from uploaded file and clean it
    nba_projections_file = request.files['nba_projections']
    nba_projections_df = pd.read_csv(nba_projections_file)
    nba_cleaned_df = clean_and_prepare_data(nba_projections_df)

    # Run the corrected simulation to calculate DD and TD probabilities
    df_projections = simulate_dd_td_probabilities_corrected(nba_cleaned_df)

    # Process FanDuel data using the file path
    df_odds = fd_nba_process_data(file_path=fd_salaries_filepath)

    # Merge odds with projections
    df_final = merge_odds_with_projections(df_odds=df_odds, df_projections=df_projections)

    # Save the final output to a CSV file
    output_filename = "fd_processed_output_nba.csv"
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    df_final.to_csv(output_filepath, index=False)

    # Send the final CSV file for download
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)
@app.route('/process_nba_dk', methods=['POST'])
def nba_process_dk():
    # Check if both files are present in the upload
    if 'dk_salaries' not in request.files or 'nba_projections' not in request.files:
        return 'Both files (DK Salaries and NBA Projections) need to be uploaded!', 400

    # Save FanDuel salaries file to get file path
    dk_salaries_file = request.files['dk_salaries']
    dk_salaries_filename = dk_salaries_file.filename
    dk_salaries_filepath = os.path.join(app.config['UPLOAD_FOLDER'], dk_salaries_filename)
    dk_salaries_file.save(dk_salaries_filepath)

    # Load NBA projections directly from uploaded file and clean it
    nba_projections_file = request.files['nba_projections']
    nba_projections_df = pd.read_csv(nba_projections_file)
    nba_cleaned_df = clean_and_prepare_data(nba_projections_df)

    # Run the corrected simulation to calculate DD and TD probabilities
    df_projections = simulate_dd_td_probabilities_corrected(nba_cleaned_df)

    # Process FanDuel data using the file path
    df_odds = dk_nba_process_data(file_path=dk_salaries_filepath)

    # Merge odds with projections
    df_final = merge_odds_with_projections(df_odds=df_odds, df_projections=df_projections)

    # Save the final output to a CSV file
    output_filename = "dk_processed_output_nba.csv"
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    df_final.to_csv(output_filepath, index=False)

    # Send the final CSV file for download
    return send_from_directory(app.config['UPLOAD_FOLDER'], output_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=80)

@app.route('/process_nba_fd_old', methods=['POST'])
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

@app.route('/process_nba_dk_old', methods=['POST'])
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

#MLB Optimizer
@app.route('/MLBOptimizer', methods=['GET', 'POST'])
def mlb_optimizer():
    if request.method == 'POST':
        try:
            batter_df = batter_data_dfs()
            pitcher_df = pitcher_data_dfs()
            fp_df = generate_fantasy_projections(batter_df, pitcher_df)
        except Exception as e:
            return f"Error generating fantasy projections: {e}", 500

        dk_file = request.files.get('dk_file')
        if not dk_file:
            return "Please upload DraftKings salary file.", 400

        try:
            dk_df = pd.read_csv(dk_file)
        except Exception as e:
            return f"Error reading DraftKings salary file: {e}", 400

        # Parameters from form
        filename_prefix = request.form.get('filename_prefix', 'dk_mlb')
        tournament = request.form.get('tournament', 'Main')
        num_lineups = int(request.form.get('num_lineups', 5))
        salary_cap = int(request.form.get('salary_cap', 50000))

        # Run optimizer and save combined upload CSV
        try:
            all_lineups, combined_upload_df = generate_multiple_lineups(
                fp_df, dk_df, num_lineups=num_lineups, 
                salary_cap=salary_cap, tournament=tournament, 
                filename_prefix=filename_prefix
            )
        except Exception as e:
            return f"Error generating lineups: {e}", 500

        # Save to upload folder
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)

        date_str = pd.Timestamp.now().strftime('%Y%m%d')
        output_filename = f"{filename_prefix}_{tournament}_{date_str}_all_lineups.csv"
        output_path = os.path.join(upload_folder, output_filename)

        combined_upload_df.to_csv(output_path, index=False)

        return send_from_directory(upload_folder, output_filename, as_attachment=True)

    return render_template('mlb_optimizer.html')

@app.route('/download_lineup/<filename>')
def download_lineup_file(filename):
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Requested file does not exist.", 404


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)




