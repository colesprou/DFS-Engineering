import os
import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path
from azure.storage.blob import BlobClient
from azure.core.exceptions import ResourceNotFoundError
from io import BytesIO
import json
from dotenv import load_dotenv
import os

load_dotenv()



def pitcher_data_dfs():
    from app_dk import get_todays_game_ids_v3
    todays_ids = get_todays_game_ids_v3(api_key=os.environ.get('ODDSJAM_API_KEY'), league="MLB")
    game_ids = list(todays_ids.keys())
    desired_markets = ['Player Strikeouts','Player Outs','Player Walks','Player Earned Runs',
                       'Player To Record Win','Player Hits Allowed']

    URL = "https://api.opticodds.com/api/v3/fixtures/odds"
    API_KEY =os.environ.get('ODDSJAM_API_KEY')

    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]
    all_odds = []

    for chunk in game_id_chunks:
        params = {
            'key': API_KEY,
            'sportsbook':['DraftKings', 'FanDuel', 'Pinnacle', 'BetOnline'],
            'fixture_id': chunk,
            'market': desired_markets,
            'is_main':'true'
        }
        response = requests.get(URL, params=params)
        if response.status_code == 200:
            all_odds.extend(response.json()['data'])
        else:
            print(f"Error {response.status_code}: {response.text}")

    rows = []
    for fixture in all_odds:
        for odd in fixture.get('odds', []):
            player = odd['selection']
            market = odd['market_id']
            sportsbook = odd['sportsbook']
            direction = odd['selection_line']  # 'over', 'under', 'yes', 'no'
            points = odd.get('points')
            price = odd.get('price')

            rows.append({
                'player': player,
                'market': market,
                'points': points,
                'sportsbook': sportsbook,
                'direction': direction,
                'price': price
            })

    df = pd.DataFrame(rows)

    # Separate 'Player To Record Win' from others
    win_df = df[df['market'] == 'player_to_record_win']
    other_df = df[df['market'] != 'player_to_record_win']

    # Process win_df: map yes/no odds to implied probability
    def odds_to_prob(odds):
        if odds is None:
            return None
        return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

    # Pivot yes/no prices wide per player, sportsbook
    win_pivot = win_df.pivot_table(
        index=['player', 'market', 'sportsbook'],
        columns='direction',
        values='price',
        aggfunc='first'
    ).reset_index()
    # Calculate implied probability for 'yes' (winner) devigged as:
    # P(yes) + P(no) = total, normalize to get true P(yes)
    def devig_win_prob(row):
        p_yes = odds_to_prob(row.get('yes'))
        p_no = odds_to_prob(row.get('no'))
        total = (p_yes or 0) + (p_no or 0)
        if total > 0:
            return p_yes / total
        return None

    win_pivot['tp'] = win_pivot.apply(devig_win_prob, axis=1)
    win_pivot.rename({'yes':'over','no':'under'},axis=1,inplace=True)
    # Process other_df as before: pivot over/under and devig implied prob
    pivoted = other_df.pivot_table(
        index=['player', 'market', 'points', 'sportsbook'],
        columns='direction',
        values='price',
        aggfunc='first'
    ).reset_index()
    pivoted.columns.name = None
    
    def devig_implied_prob(over, under):
        if pd.isna(over) or pd.isna(under):
            return None
        p_over = odds_to_prob(over)
        p_under = odds_to_prob(under)
        total = p_over + p_under
        return p_over / total if total > 0 else None

    pivoted['tp'] = pivoted.apply(lambda row: devig_implied_prob(row.get('over'), row.get('under')), axis=1)
    
    # Combine both
    combined_df = pd.concat([pivoted, win_pivot], ignore_index=True)
    combined_df['points'] = combined_df['points'].fillna(-1)  # or 'NA' if points is string
    # Group average by player/market/points
    avg_tp_df = combined_df.groupby(['player', 'market', 'points']).agg(tp_avg=('tp', 'mean')).reset_index()

    return avg_tp_df
def batter_data_dfs():
    from app_dk import get_todays_game_ids_v3
    todays_ids = get_todays_game_ids_v3(api_key=os.environ.get('ODDSJAM_API_KEY'),league="MLB")
    game_ids = list(todays_ids.keys())
    desired_markets = [
        "Player Home Runs", "Player RBIs", "Player Runs",
        "Player Stolen Bases", "Player Singles", "Player Doubles",
        "Player Triples", "Player Batting Walks"
    ]

    URL = "https://api.opticodds.com/api/v3/fixtures/odds"
    API_KEY = os.environ.get('ODDSJAM_API_KEY')
    sportsbooks = ['DraftKings', 'Blue Book', 'Pinnacle', 'BetOnline']

    game_id_chunks = [game_ids[i:i + 5] for i in range(0, len(game_ids), 5)]
    all_data = []

    for chunk in game_id_chunks:
        params = {
            'key': API_KEY,
            'sportsbook': sportsbooks,
            'fixture_id': chunk,
            'market': desired_markets,
            'is_main': 'true'
        }
        response = requests.get(URL, params=params)
        if response.status_code == 200 and 'data' in response.json():
            all_data.extend(response.json()['data'])
        else:
            print(f"Error {response.status_code}: {response.text}")

    rows = []
    for game in all_data:
        for odd in game.get('odds', []):
            player = odd.get('selection', 'Unknown')
            market = odd.get('market', '')
            book = odd.get('sportsbook', '')
            direction = odd.get('selection_line', '')
            points = odd.get('points')
            price = odd.get('price')
            rows.append({
                'player': player,
                'market': market,
                'points': points,
                'sportsbook': book,
                'direction': direction,
                'price': price
            })

    df = pd.DataFrame(rows)

    # Pivot over/under side-by-side
    pivoted = df.pivot_table(
        index=['player', 'market', 'points', 'sportsbook'],
        columns='direction',
        values='price',
        aggfunc='first'
    ).reset_index()
    pivoted.to_csv('QC_Batter_Props.csv',index=False)

    pivoted.columns.name = None

    def devig_implied_prob(over, under):
        def odds_to_prob(odds):
            return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

        if pd.isna(over) and pd.isna(under):
            return None
        elif pd.isna(under):
            return odds_to_prob(over)  # fallback
        elif pd.isna(over):
            return 1 - odds_to_prob(under)  # fallback
        else:
            p_over = odds_to_prob(over)
            p_under = odds_to_prob(under)
            total = p_over + p_under
            return p_over / total if total > 0 else None

    # Apply devig
    pivoted['tp'] = pivoted.apply(lambda row: devig_implied_prob(row['over'], row['under']), axis=1)

    # Market average per player/market/line
    avg_tp_df = (
        pivoted
        .groupby(['player', 'market', 'points'])
        .agg(tp_avg=('tp', 'mean'))
        .reset_index()
    )

    return avg_tp_df
def generate_fantasy_projections(batter_df, pitcher_df):
    import pandas as pd

    batter_df['expected'] = batter_df['tp_avg'] * np.ceil(batter_df['points'])
    pitcher_df['expected'] = pitcher_df['tp_avg'] * np.ceil(pitcher_df['points'])

    batter_wide = batter_df.pivot_table(
        index='player',
        columns='market',
        values='expected',
        aggfunc='first'
    ).reset_index().fillna(0)

    pitcher_wide = pitcher_df.pivot_table(
        index='player',
        columns='market',
        values='expected',
        aggfunc='first'
    ).reset_index().fillna(0)

    # Warn if key stats are missing
    hitter_stats = [
        "Player Singles", "Player Doubles", "Player Triples",
        "Player Home Runs", "Player RBIs", "Player Runs",
        "Player Batting Walks", "Player Stolen Bases"
    ]
    pitcher_stats = [
        "player_strikeouts", "player_outs", "player_to_record_win",
        "player_earned_runs", "player_hits_allowed", "player_walks"
    ]

    #warn_missing_markets(batter_wide, hitter_stats, label="batter")
    #warn_missing_markets(pitcher_wide, pitcher_stats, label="pitcher")

    def hitter_fp(row):
        return (
            row.get("Player Singles", 0) * 3 +
            row.get("Player Doubles", 0) * 5 +
            row.get("Player Triples", 0) * 8 +
            row.get("Player Home Runs", 0) * 10 +
            row.get("Player RBIs", 0) * 1.5 +
            row.get("Player Runs", 0) * 1.5 +
            row.get("Player Batting Walks", 0) * 2 +
            row.get("Player Stolen Bases", 0) * 5
        )

    def pitcher_fp(row):
        return (
            row.get("player_strikeouts", 0) * 2 +
            row.get("player_outs", 0) * 0.75 +
            row.get("Player To Record Win", 0) * 4 +
            row.get("player_earned_runs", 0) * -2 +
            row.get("player_hits_allowed", 0) * -0.6 +
            row.get("player_walks", 0) * -0.6
        )

    batter_wide["fantasy_points"] = batter_wide.apply(hitter_fp, axis=1)
    batter_wide["type"] = "batter"

    pitcher_wide["fantasy_points"] = pitcher_wide.apply(pitcher_fp, axis=1)
    pitcher_wide["type"] = "pitcher"

    return pd.concat([batter_wide, pitcher_wide], ignore_index=True)
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatusOptimal
from collections import defaultdict
from datetime import datetime

def normalize_id(player_id: str) -> str:
    if '_' in player_id:
        player_id = player_id.split('_', 1)[1]
    return player_id.replace('_', '-')

def convert_lineup_to_upload_row(lineup_df):
    upload_order = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
    lineup_df["Normalized ID"] = lineup_df["ID"].astype(str).apply(normalize_id)
    upload_dict = defaultdict(list)
    for _, row in lineup_df.iterrows():
        upload_dict[row['Position']].append(row['Normalized ID'])
    row = [upload_dict[pos].pop(0) if upload_dict[pos] else '' for pos in upload_order]
    return row

def generate_multiple_lineups(fp_df, dk_df, num_lineups=20, salary_cap=50000, tournament='Main', filename_prefix='dk_mlb'):
    from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatusOptimal
    from datetime import datetime

    # Helper function
    def run_pitcher_first_filtered_optimizer(fp_df, dk_df, num_lineups, salary_cap, tournament, filename_prefix):
        dk_df["opponent"] = dk_df.apply(lambda row: extract_opponent(row["TeamAbbrev"], row["Game Info"]), axis=1)
        df = pd.merge(fp_df, dk_df[["Name", "Salary", "Roster Position", "ID", "TeamAbbrev", "opponent"]], on="Name", how="inner")
        df = df[df["Salary"].notnull() & df["Roster Position"].notnull()]
        df["fantasy_points"] = pd.to_numeric(df["fantasy_points"], errors="coerce").fillna(0)
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0)
        seen_lineups = set()
        # Expand multi-position players
        expanded = []
        for _, row in df.iterrows():
            for pos in row["Roster Position"].split("/"):
                expanded.append({
                    "Name": row["Name"],
                    "Position": pos.strip(),
                    "Salary": row["Salary"],
                    "fantasy_points": row["fantasy_points"],
                    "ID": row["ID"],
                    "TeamAbbrev": row["TeamAbbrev"],
                    "opponent": row["opponent"]
                })
        flat_df = pd.DataFrame(expanded).drop_duplicates(subset=["Name", "Position"]).reset_index(drop=True)

        position_limits = {
            "P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3
        }

        all_lineups = []
        all_upload_rows = []
        retries = 100

        for _ in range(retries):
            prob = LpProblem("Filtered_Lineup", LpMaximize)
            decision_vars = {i: LpVariable(f"select_{i}", cat="Binary") for i in flat_df.index}

            prob += lpSum(flat_df.loc[i, "fantasy_points"] * decision_vars[i] for i in flat_df.index)
            prob += lpSum(flat_df.loc[i, "Salary"] * decision_vars[i] for i in flat_df.index) <= salary_cap
            prob += lpSum(decision_vars[i] for i in flat_df.index) == 10

            for name in flat_df["Name"].unique():
                indices = flat_df[flat_df["Name"] == name].index
                prob += lpSum(decision_vars[i] for i in indices) <= 1

            for pos, req in position_limits.items():
                indices = flat_df[flat_df["Position"] == pos].index
                prob += lpSum(decision_vars[i] for i in indices) == req

            for prev_indices in seen_lineups:
                prob += lpSum([decision_vars[i] for i in prev_indices]) <= 9

            status = prob.solve()
            if status != LpStatusOptimal:
                continue

            selected_indices = frozenset(i for i in flat_df.index if decision_vars[i].value() == 1)
            selected = flat_df.loc[list(selected_indices)]

            hitter_teams = selected[selected["Position"] != "P"]["TeamAbbrev"].value_counts()
            if any(hitter_teams > 5):
                continue

            if selected_indices not in seen_lineups:
                seen_lineups.add(selected_indices)
                lineup_df = selected[["Name", "Position", "Salary", "fantasy_points", "ID"]].copy()
                lineup_df.rename(columns={"fantasy_points": "Fantasy Points"}, inplace=True)
                all_lineups.append(lineup_df)
                upload_row = convert_lineup_to_upload_row(lineup_df)
                all_upload_rows.append(upload_row)

            if len(all_lineups) >= num_lineups:
                break

        upload_order = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        combined_upload_df = pd.DataFrame(all_upload_rows, columns=upload_order)

        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"{filename_prefix}_{tournament}_{date_str}_all_lineups.csv"
        combined_upload_df.to_csv(filename, index=False)

        return all_lineups, combined_upload_df

    # Preprocessing
    fp_df["Name"] = fp_df["player"].str.replace(r"[^A-Za-z\s]", "", regex=True).str.strip()
    dk_df["Name"] = dk_df["Name"].str.replace(r"[^A-Za-z\s]", "", regex=True).str.strip()
    dk_df["opponent"] = dk_df.apply(lambda row: extract_opponent(row["TeamAbbrev"], row["Game Info"]), axis=1)
    df = pd.merge(fp_df, dk_df[["Name", "Salary", "Roster Position", "ID", "TeamAbbrev", "opponent"]], on="Name", how="inner")
    df = df[df["Salary"].notnull() & df["Roster Position"].notnull()]
    df["fantasy_points"] = pd.to_numeric(df["fantasy_points"], errors="coerce").fillna(0)
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0)

    expanded = []
    for _, row in df.iterrows():
        for pos in row["Roster Position"].split("/"):
            expanded.append({
                "Name": row["Name"],
                "Position": pos.strip(),
                "Salary": row["Salary"],
                "fantasy_points": row["fantasy_points"],
                "ID": row["ID"],
                "TeamAbbrev": row["TeamAbbrev"],
                "opponent": row["opponent"]
            })
    flat_df = pd.DataFrame(expanded).drop_duplicates(subset=["Name", "Position"]).reset_index(drop=True)

    position_limits = {
        "P": 2, "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3
    }

    all_lineups = []
    all_upload_rows = []
    retries = 100
    seen_lineups = set()
    for _ in range(retries):
        prob = LpProblem("Pitcher_First", LpMaximize)
        decision_vars = {i: LpVariable(f"select_{i}", cat="Binary") for i in flat_df.index}

        prob += lpSum(flat_df.loc[i, "fantasy_points"] * decision_vars[i] for i in flat_df.index)
        prob += lpSum(flat_df.loc[i, "Salary"] * decision_vars[i] for i in flat_df.index) <= salary_cap
        prob += lpSum(decision_vars[i] for i in flat_df.index) == 10

        for name in flat_df["Name"].unique():
            indices = flat_df[flat_df["Name"] == name].index
            prob += lpSum(decision_vars[i] for i in indices) <= 1

        for pos, req in position_limits.items():
            indices = flat_df[flat_df["Position"] == pos].index
            prob += lpSum(decision_vars[i] for i in indices) == req

        for prev_indices in seen_lineups:
            prob += lpSum([decision_vars[i] for i in prev_indices]) <= 9

        status = prob.solve()
        if status != LpStatusOptimal:
            continue

        selected_indices = frozenset(i for i in flat_df.index if decision_vars[i].value() == 1)
        selected = flat_df.loc[list(selected_indices)]

        pitcher_teams = selected[selected['Position'] == 'P']['TeamAbbrev'].tolist()
        filtered_df = flat_df[
            ~((flat_df['Position'] != 'P') & (flat_df['opponent'].isin(pitcher_teams)))
        ]

        prob = LpProblem("Filtered_Lineup", LpMaximize)
        decision_vars = {i: LpVariable(f"select_{i}", cat="Binary") for i in filtered_df.index}

        prob += lpSum(filtered_df.loc[i, "fantasy_points"] * decision_vars[i] for i in filtered_df.index)
        prob += lpSum(filtered_df.loc[i, "Salary"] * decision_vars[i] for i in filtered_df.index) <= salary_cap
        prob += lpSum(decision_vars[i] for i in filtered_df.index) == 10

        for name in filtered_df["Name"].unique():
            indices = filtered_df[filtered_df["Name"] == name].index
            prob += lpSum(decision_vars[i] for i in indices) <= 1

        for pos, req in position_limits.items():
            indices = filtered_df[filtered_df["Position"] == pos].index
            prob += lpSum(decision_vars[i] for i in indices) == req

        for prev_indices in seen_lineups:
            prob += lpSum([decision_vars[i] for i in prev_indices]) <= 9

        status = prob.solve()
        if status != LpStatusOptimal:
            continue

        selected_indices = frozenset(i for i in filtered_df.index if decision_vars[i].value() == 1)
        selected = filtered_df.loc[list(selected_indices)]

        hitter_teams = selected[selected["Position"] != "P"]["TeamAbbrev"].value_counts()
        if any(hitter_teams > 5):
            continue

        if selected_indices not in seen_lineups:
            seen_lineups.add(selected_indices)
            lineup_df = selected[["Name", "Position", "Salary", "fantasy_points", "ID"]].copy()
            lineup_df.rename(columns={"fantasy_points": "Fantasy Points"}, inplace=True)
            all_lineups.append(lineup_df)
            upload_row = convert_lineup_to_upload_row(lineup_df)
            all_upload_rows.append(upload_row)

        if len(all_lineups) >= num_lineups:
            break

    if len(all_lineups) == 0:
        print("No clean lineups found. Retrying with filtered optimizer.")
        return run_pitcher_first_filtered_optimizer(fp_df, dk_df, num_lineups, salary_cap, tournament, filename_prefix)

    upload_order = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
    combined_upload_df = pd.DataFrame(all_upload_rows, columns=upload_order)

    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{filename_prefix}_{tournament}_{date_str}_all_lineups.csv"
    combined_upload_df.to_csv(filename, index=False)

    return all_lineups, combined_upload_df
# Extract opponent team from 'Game Info'
def extract_opponent(team, game_info):
    try:
        teams = game_info.split()[0].split("@")
        if team == teams[0]:
            return teams[1]
        elif team == teams[1]:
            return teams[0]
    except:
        return None
