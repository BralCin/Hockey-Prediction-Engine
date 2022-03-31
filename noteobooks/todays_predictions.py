import pandas as pd
import numpy as np
import re
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')

## Prepare Game Results

df_games_season = pd.read_csv('../data/games/21-22_season.csv')

df_games_season['season'] = 2021
df_games_season['home_or_away'] = 'HOME'

game_results = df_games_season.rename(columns={"Date": "gameDate", "Home": "playerTeam", "Visitor": "opposingTeam", "Score.1": "goalsFor", "Score": "goalsAgainst"})

# define fucntion to create column of game results
# 2 is a win for 'playerTeam'
# 1 is a win for 'opposingTeam'
# 0 is a game that went to shootout
def game_result_label_race(row):
    if row['goalsFor'] > row['goalsAgainst']:
        return 'team 1'
    if row['goalsFor'] < row['goalsAgainst']:
        return 'team 2'

# define fucntion that encodes home team
# 1 if team1 is home team
# 2 if team2 is home team
def home_team_t1_label_race(row):
    if row['home_or_away'] == 'HOME':
        return 1
    if row['home_or_away'] == 'AWAY':
        return 0
    
    
def home_team_t2_label_race(row):
    if row['home_or_away'] == 'HOME':
        return 0
    if row['home_or_away'] == 'AWAY':
        return 1   

game_results['home_or_away_t1'] = np.nan
game_results['home_or_away_t2'] = np.nan

# apply functions to crate column of game results and encode home team
game_results['result'] = game_results.apply(lambda row: game_result_label_race(row), axis=1)
game_results['home_or_away_t1'] = game_results.apply(lambda row: home_team_t1_label_race(row), axis=1)
game_results['home_or_away_t2'] = game_results.apply(lambda row: home_team_t2_label_race(row), axis=1)

# rename, drop and order columns for usability + reindexing
game_results.rename(columns={'playerTeam': 'team1', 'opposingTeam': 'team2'}, inplace=True)
game_results.drop(['goalsFor', 'goalsAgainst', 'home_or_away'], axis=1, inplace=True)
game_results = game_results[['gameDate', 'season', 'team1', 'team2', 'result', 'home_or_away_t1', 'home_or_away_t2']].sort_values('gameDate').reset_index(drop=True)

# convert date to python date
game_results['gameDate']= pd.to_datetime(game_results['gameDate'])

teams = ['Anaheim Ducks',
         'Arizona Coyotes',
         'Boston Bruins',
         'Buffalo Sabres',
         'Calgary Flames',
         'Carolina Hurricanes',
         'Chicago Blackhawks',
         'Colorado Avalanche',
         'Columbus Blue Jackets',
         'Dallas Stars',
         'Detroit Red Wings',
         'Edmonton Oilers',
         'Florida Panthers',
         'Los Angeles Kings',
         'Minnesota Wild',
         'Montreal Canadiens',
         'Nashville Predators',
         'New Jersey Devils',
         'New York Islanders',
         'New York Rangers',
         'Ottawa Senators',
         'Philadelphia Flyers',
         'Pittsburgh Penguins',
         'San Jose Sharks',
         'Seattle Kraken',
         'St Louis Blues',
         'Tampa Bay Lightning',
         'Toronto Maple Leafs',
         'Vancouver Canucks',
         'Vegas Golden Knights',
         'Washington Capitals',
         'Winnipeg Jets']

teams_abv = ['ANA', 'ARI', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 
             'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL',
             'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 
             'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG']


def t1_abv_label_race(row):
    for i in range(32):
        if row['team1'] == teams[i]:
            return teams_abv[i]

        
def t2_abv_label_race(row):
    for i in range(32):
        if row['team2'] == teams[i]:
            return teams_abv[i]   

game_results['team1'] = game_results['team1'].str.replace('.', '')
game_results['team2'] = game_results['team2'].str.replace('.', '')

game_results['team1'] = game_results.apply(lambda row: t1_abv_label_race(row), axis=1)
game_results['team2'] = game_results.apply(lambda row: t2_abv_label_race(row), axis=1)

upcoming_games = game_results[game_results.isna().any(axis=1)]
game_results = game_results[~game_results.isna().any(axis=1)]

today = datetime.today().strftime('%Y-%m-%d')
todays_games = upcoming_games.loc[upcoming_games['gameDate'] == today].drop(['result'], axis=1).reset_index(drop=True)

todays_games

## Web

# lines
req = Request('https://moneypuck.com/moneypuck/playerData/seasonSummary/2021/regular/lines.csv')
req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
content = urlopen(req)

tracked_statistics1 = ['name', 'season', 'team', 'icetime',
                      'flurryScoreVenueAdjustedxGoalsFor', 'xOnGoalFor', 'reboundxGoalsFor',
                      'penaltiesAgainst', 'takeawaysFor',
                      'lowDangerxGoalsFor', 'mediumDangerxGoalsFor','highDangerxGoalsFor',
                      'flurryScoreVenueAdjustedxGoalsAgainst', 'xOnGoalAgainst', 'reboundxGoalsAgainst',
                      'penaltiesFor', 'takeawaysAgainst',
                      'lowDangerxGoalsAgainst', 'mediumDangerxGoalsAgainst','highDangerxGoalsAgainst']

df_lines = pd.read_csv(content)
df_lines = df_lines[tracked_statistics1]

# select stats to regularize by games played
lines_reg = tracked_statistics1[4:]

# clean team names
df_lines['team'] = df_lines['team'].str.replace('.', '')

# isolate line and pairing stats from each other 
df_lines = df_lines[tracked_statistics1].reset_index(drop=True)

# regularize to icetime
df_lines[lines_reg] = df_lines[lines_reg].divide(df_lines['icetime'], axis='index').multiply(50000, axis='index')

# stats to weight
major_stats = ['flurryScoreVenueAdjustedxGoalsFor', 'flurryScoreVenueAdjustedxGoalsAgainst', 
               'penaltiesFor', 'penaltiesAgainst',
               'takeawaysFor', 'takeawaysAgainst',
               'mediumDangerxGoalsFor', 'mediumDangerxGoalsAgainst',
               'highDangerxGoalsFor', 'highDangerxGoalsAgainst']

minor_stats = ['xOnGoalFor', 'xOnGoalAgainst', 
               'reboundxGoalsFor', 'reboundxGoalsAgainst',
               'lowDangerxGoalsFor', 'lowDangerxGoalsAgainst']

# skater
req = Request('https://moneypuck.com/moneypuck/playerData/seasonSummary/2021/regular/skaters.csv')
req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
content = urlopen(req)

tracked_statistics2 = ['name', 'position', 'season', 'team', 'icetime',
                      'OnIce_F_flurryScoreVenueAdjustedxGoals', 'OnIce_F_xOnGoal', 'OnIce_F_reboundxGoals',
                      'penaltiesDrawn', 'I_F_takeaways',
                      'OnIce_F_lowDangerxGoals', 'OnIce_F_mediumDangerxGoals', 'OnIce_F_highDangerxGoals', 
                      'OnIce_A_flurryScoreVenueAdjustedxGoals', 'OnIce_A_xOnGoal', 'OnIce_A_reboundxGoals',
                      'penalties', 'I_F_giveaways',
                      'OnIce_A_lowDangerxGoals', 'OnIce_A_mediumDangerxGoals', 'OnIce_A_highDangerxGoals']

df_skaters = pd.read_csv(content)
df_skaters = df_skaters.loc[(df_skaters['situation'] == '5on5')]
df_skaters['name'] = df_skaters['name'].str.split().str[1]
df_skaters = df_skaters[tracked_statistics2]
df_skaters.columns = tracked_statistics2

# goalie
req1 = Request('https://moneypuck.com/moneypuck/playerData/seasonSummary/2021/regular/goalies.csv')
req1.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
content1 = urlopen(req1)

req2 = Request('https://moneypuck.com/moneypuck/playerData/seasonSummary/2020/regular/goalies.csv')
req2.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
content2 = urlopen(req2)

df_goalies = pd.read_csv(content1, header=None)
df_goalies_labels = pd.read_csv(content2)
df_goalies = pd.DataFrame(data=df_goalies.values, columns=df_goalies_labels.columns)

tracked_statistics3 = ['name', 'team', 'season', 'games_played',
                      'xRebounds',  'xOnGoal', 'xPlayContinuedInZone',
                      'lowDangerxGoals', 'mediumDangerxGoals', 'highDangerxGoals',
                      'xPlayStopped', 'xPlayContinuedOutsideZone', 'xFreeze']


# select stats to regularize by games played
goalie_reg = tracked_statistics3[4:]

# isolate gaolie stats from all situations and drop columns for usability 
df_goalies = df_goalies[tracked_statistics3].loc[(df_goalies['situation'] == '5on5')]
df_goalies['team'] = df_goalies['team'].str.replace('.', '')

# scale goalie stats for wieghting
scaler = MinMaxScaler()
df_goalies[goalie_reg] = scaler.fit_transform(df_goalies[goalie_reg])

# weight stats
## weight minor stats
goalie_weighting = goalie_reg[:6]
df_goalies[goalie_weighting] = df_goalies[goalie_weighting] * 1
## weight major stats
goalie_weighting = goalie_reg[6:]
df_goalies[goalie_weighting] = df_goalies[goalie_weighting] * 2

# create goalie metric and drop old stats
df_goalies['goalie_strength_neg'] = df_goalies.loc[:,'xPlayStopped':'xFreeze'].mean(axis = 1)
df_goalies['goalie_strength_pos'] = df_goalies.loc[:,'xRebounds':'highDangerxGoals'].mean(axis = 1).mul(-.5)
df_goalies['goalie_strength'] = df_goalies['goalie_strength_neg'] + df_goalies['goalie_strength_pos']
df_goalies.drop(goalie_reg, axis=1, inplace=True)
df_goalies.drop(['goalie_strength_neg', 'goalie_strength_pos', 'games_played'], axis=1, inplace=True)

df_goalies['season'] = 2021

# split off first name leaving only last name
df_goalies['name'] = df_goalies['name'].str.split().str[1]

# team
req1 = Request('https://moneypuck.com/moneypuck/playerData/seasonSummary/2021/regular/teams.csv')
req1.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
content1 = urlopen(req1)

req2 = Request('https://moneypuck.com/moneypuck/playerData/seasonSummary/2020/regular/teams.csv')
req2.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
content2 = urlopen(req2)

df_teams = pd.read_csv(content1, header=None)
df_teams_labels = pd.read_csv(content2)
df_teams = pd.DataFrame(data=df_teams.values, columns=df_teams_labels.columns)

tracked_statistics = ['name', 'season',
                      'xGoalsPercentage',
                      'flurryScoreVenueAdjustedxGoalsFor', 'penaltiesFor', 'dZoneGiveawaysAgainst',
                      'flurryScoreVenueAdjustedxGoalsAgainst', 'penaltiesAgainst', 'dZoneGiveawaysFor']

team_stats_5on5 = df_teams[tracked_statistics].loc[(df_teams['situation'] == '5on5')].add_suffix('_5on5').rename(columns={"name_5on5": "name", "season_5on5": "season"})
team_stats_5on4 = df_teams[tracked_statistics].loc[(df_teams['situation'] == '5on4')].add_suffix('_5on4').rename(columns={"name_5on4": "name", "season_5on4": "season"})
team_stats_4on5 = df_teams[tracked_statistics].loc[(df_teams['situation'] == '4on5')].add_suffix('_4on5').rename(columns={"name_4on5": "name", "season_4on5": "season"})

team_stats_5on5['team_strength_pos'] = team_stats_5on5.loc[:,'flurryScoreVenueAdjustedxGoalsFor_5on5':'dZoneGiveawaysAgainst_5on5'].mean(axis = 1)
team_stats_5on5['team_strength_neg'] = team_stats_5on5.loc[:,'flurryScoreVenueAdjustedxGoalsAgainst_5on5':].mean(axis = 1).mul(-.5)
team_stats_5on5['team_strength_5on5'] = team_stats_5on5.iloc[:, -2:].sum(axis=1)
team_stats_5on5.drop(['team_strength_pos', 'team_strength_neg'], axis=1, inplace=True)

team_stats_5on4['team_strength_pos'] = team_stats_5on4.loc[:,'flurryScoreVenueAdjustedxGoalsFor_5on4':'dZoneGiveawaysAgainst_5on4'].mean(axis = 1)
team_stats_5on4['team_strength_neg'] = team_stats_5on4.loc[:,'flurryScoreVenueAdjustedxGoalsAgainst_5on4':].mean(axis = 1).mul(-.5)
team_stats_5on4['team_strength_5on4'] = team_stats_5on4.iloc[:, -2:].sum(axis=1)
team_stats_5on4.drop(['team_strength_pos', 'team_strength_neg', 'xGoalsPercentage_5on4'], axis=1, inplace=True)

team_stats_4on5['team_strength_pos'] = team_stats_4on5.loc[:,'flurryScoreVenueAdjustedxGoalsFor_4on5':'dZoneGiveawaysAgainst_4on5'].mean(axis = 1)
team_stats_4on5['team_strength_neg'] = team_stats_4on5.loc[:,'flurryScoreVenueAdjustedxGoalsAgainst_4on5':].mean(axis = 1).mul(-.5)
team_stats_4on5['team_strength_4on5'] = team_stats_4on5.iloc[:, -2:].sum(axis=1)
team_stats_4on5.drop(['team_strength_pos', 'team_strength_neg', 'xGoalsPercentage_4on5'], axis=1, inplace=True)

team_stats = pd.merge(team_stats_5on5, team_stats_5on4, how='left', on=['name', 'season'])
team_stats = pd.merge(team_stats, team_stats_4on5, how='left', on=['name', 'season'])

team_stats['name'] = team_stats['name'].str.replace('.', '')
team_stats.fillna(team_stats.median(), inplace=True)
df_teams = team_stats[['name', 'season', 'xGoalsPercentage_5on5', 'team_strength_5on5', 'team_strength_5on4', 'team_strength_4on5']]

# time series
req = Request('https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv')
req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
content = urlopen(req)

all_games_df = pd.read_csv(content)
ts = all_games_df

ts.sort_values('gameDate', inplace=True)

ts = ts.loc[(ts['situation'] == '5on5') & (ts['playoffGame'] == 0)]
ts[['team']] = ts[['team']].apply(lambda x: x.str.replace('.', ''))

ts['gameDate'] =  pd.to_datetime(ts['gameDate'], format = '%Y%m%d')

ts = ts[['team', 'gameDate', 'xGoalsPercentage']]

teams_list = list(ts['team'].unique())

timeseries_teams = {}

for i  in range(len(teams_list)):
    timeseries_teams[f'{teams_list[i]}'] = ts.loc[ts['team'] == teams_list[i]]

for i in teams_list:
    df_ts_team = timeseries_teams[i].copy()
    df_ts_team['xGoalsPercentage_last_3'] = df_ts_team['xGoalsPercentage'].rolling(window=3, closed= "left").mean().fillna(.5)
    df_ts_team['xGoalsPercentage_last_5'] = df_ts_team['xGoalsPercentage'].rolling(window=5, closed= "left").mean().fillna(.5)
    df_ts_team['xGoalsPercentage_last_10'] = df_ts_team['xGoalsPercentage'].rolling(window=10, closed= "left").mean().fillna(.5)
    timeseries_teams[i] = df_ts_team

df_ts = pd.concat(timeseries_teams.values(), ignore_index=True).drop(['xGoalsPercentage'], axis=1)
df_ts = df_ts.replace(['TB', 'NJ', 'LA', 'SJ'], ['TBL', 'NJD', 'LAK', 'SJS'])
df_ts = df_ts.sort_values(['team', 'gameDate']).drop_duplicates('team', keep='last')

teams = [x.lower() for x in ['Anaheim Ducks',
                             'Arizona Coyotes',
                             'Boston Bruins',
                             'Buffalo Sabres',
                             'Calgary Flames',
                             'Carolina Hurricanes',
                             'Chicago Blackhawks',
                             'Colorado Avalanche',
                             'Columbus Blue Jackets',
                             'Dallas Stars',
                             'Detroit Red Wings',
                             'Edmonton Oilers',
                             'Florida Panthers',
                             'Los Angeles Kings',
                             'Minnesota Wild',
                             'Montreal Canadiens',
                             'Nashville Predators',
                             'New Jersey Devils',
                             'New York Islanders',
                             'New York Rangers',
                             'Ottawa Senators',
                             'Philadelphia Flyers',
                             'Pittsburgh Penguins',
                             'San Jose Sharks',
                             'Seattle Kraken',
                             'St Louis Blues',
                             'Tampa Bay Lightning',
                             'Toronto Maple Leafs',
                             'Vancouver Canucks',
                             'Vegas Golden Knights',
                             'Washington Capitals',
                             'Winnipeg Jets']]

teams_abv = ['ANA', 'ARI', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 
             'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL',
             'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 
             'SEA', 'STL', 'TBL', 'TOR', 'VAN', 'VGK', 'WSH', 'WPG']


for i in range(len(teams)):
    teams[i] = teams[i].replace(" ", "-")

df_lines['name'] = df_lines['name'].str.split('-')
df_lines['name'] = df_lines['name'].apply(sorted)
df_lines['name'] = df_lines['name'].apply(lambda x: '-'.join(map(str, x)))

options = FirefoxOptions()
options.add_argument("--headless")
driver = webdriver.Firefox(options=options)

lineups = {}

for i in range(len(teams)):
    driver.get(f"https://www.dailyfaceoff.com/teams/{teams[i]}/line-combinations/")

    content = driver.page_source
    soup = BeautifulSoup(content)

    players = soup.findAll('span', attrs={'class':'player-name'})
    players = [str(i) for i in players]
    
    for j in range(len(players)):
        s = players[j]
        players[j] = re.search('<span class="player-name">(.*)</span>', s).group(1).split()[-1]
            
    skaters = players[:18] + players[-2:]
    lines = [skaters[:3], skaters[3:6], skaters[6:9], skaters[9:12], skaters[12:14], skaters[14:16], skaters[16:18]]
    goalies = [players[36], players[37]]
    
    df_s = pd.DataFrame({'name': lines})
    df_s['name'][0:7] = df_s['name'][0:7].apply(sorted)
    df_s['name'][0:7] = df_s['name'][0:7].apply(lambda x: '-'.join(map(str, x)))
    df_s = pd.merge(df_s, df_lines, how='left', on='name')
    
    for k in range(7):
        if df_s.iloc[k].isna().sum() != 0:
            if k < 4:
                line = df_s['name'][k].split('-')
                line_players = pd.concat([df_skaters.loc[(df_skaters['name'] == line[0]) & (df_skaters['team'] == teams_abv[i])],
                                          df_skaters.loc[(df_skaters['name'] == line[1]) & (df_skaters['team'] == teams_abv[i])],
                                          df_skaters.loc[(df_skaters['name'] == line[2]) & (df_skaters['team'] == teams_abv[i])]])

                line_players.loc[0] = line_players.mean()
                line_players = line_players.loc[[0]]

                line_players['name'] = '-'.join(line)
                line_players['team'] = teams_abv[i]
                line_players.drop(['position'], axis=1, inplace=True)
            
                df_s.iloc[k] = line_players.iloc[0]
            else:
                line = df_s['name'][k].split('-')
                line_players = pd.concat([df_skaters.loc[(df_skaters['name'] == line[0]) & (df_skaters['team'] == teams_abv[i])],
                                          df_skaters.loc[(df_skaters['name'] == line[1]) & (df_skaters['team'] == teams_abv[i])]])

                line_players.loc[0] = line_players.mean()
                line_players = line_players.loc[[0]]

                line_players['name'] = '-'.join(line)
                line_players['team'] = teams_abv[i]
                line_players.drop(['position'], axis=1, inplace=True)

                df_s.iloc[k] = line_players.iloc[0]
        else:
            pass
                     
    # weight stats
    df_s[major_stats] = df_s[major_stats] * 2
    df_s[minor_stats] = df_s[minor_stats] * 1

    # remove duplicates
    df_s['name'] = df_s['name'].str.split('-').apply(sorted)
    df_s['name'] = df_s['name'].apply(lambda x: '-'.join(map(str, x)))
    df_s = df_s.sort_values('icetime', ascending=False)
    df_s = df_s.drop_duplicates(subset='name', keep="first")
    
    # create line/pair metrics and drop old stats
    df_s['strength_pos'] = df_s.loc[:,'flurryScoreVenueAdjustedxGoalsFor':'highDangerxGoalsFor'].mean(axis = 1)
    df_s['strength_neg'] = df_s.loc[:,'flurryScoreVenueAdjustedxGoalsAgainst':].mean(axis = 1).mul(-.5)
    df_s['strength'] = df_s.iloc[:, -2:].sum(axis=1)
    df_s.drop(['strength_pos', 'strength_neg', 'icetime'], axis=1, inplace=True)
    
    df_s = df_s[['name', 'season', 'team', 'strength']].sort_index().reset_index(drop=True).fillna(df_s.mode().iloc[0])
    df_l = df_s[['season', 'team']].iloc[:1]

    df_l['line1_strength'] = df_s['strength'][0] * .32
    df_l['line2_strength'] = df_s['strength'][1] * .27
    df_l['line3_strength'] = df_s['strength'][2] * .22
    df_l['line4_strength'] = df_s['strength'][3] * .19
    df_l['forward_strength'] = df_l.loc[:, 'line1_strength':'line4_strength'].sum(axis=1)
    
    df_l['pair1_strength'] = df_s['strength'][4] * .39
    df_l['pair2_strength'] = df_s['strength'][5] * .33
    df_l['pair3_strength'] = df_s['strength'][6] * .28
    df_l['defense_strength'] = df_l.loc[:, 'pair1_strength':'pair3_strength'].sum(axis=1)
    
    df_s = df_l.drop(['line1_strength', 'line2_strength', 'line3_strength', 'line4_strength',
                              'pair1_strength', 'pair2_strength', 'pair3_strength'], axis=1).reset_index(drop=True)
    
    # create goalie df
    df_g = pd.DataFrame({'name': goalies, 'team': [teams_abv[i], teams_abv[i]]})
    df_g = pd.merge(df_g, df_goalies, how='left', on=['name', 'team']).drop(['name'], axis=1).reset_index(drop=True)
    df_g.drop(df_g.tail(1).index,inplace=True)    
    df_g['team'] = teams_abv[i]
    df_g['season'] = 2021
    
    # create team df
    df_t = df_teams.loc[df_teams['name'] == teams_abv[i]]
    df_t = df_t.rename(columns={'name': 'team'})
     
    # create lineup dictionary
    lineups[f'{teams_abv[i]}'] = {'skaters': df_s, 'goalies': df_g, 'team': df_t}


teams = []
for team in lineups:
    lineups[team]['team']['season'] = lineups[team]['team']['season'].astype('int')
    lineups[team]['goalies']['season'] = lineups[team]['goalies']['season'].astype('int')
    
    df = pd.merge(lineups[team]['team'], lineups[team]['goalies'], how='left', on=['team', 'season'])
    df = pd.merge(df, lineups[team]['skaters'], how='left', on=['team', 'season'])
    
    df['gameDate'] = today
    df_ts['gameDate'] = today   
    
    df = pd.merge(df, df_ts, how='left', on=['team', 'gameDate'])
    df.drop(['gameDate'], axis=1, inplace=True)
    teams.append(df)
    
teams_df = pd.concat(teams).fillna(.75)
teams_df.drop_duplicates(keep=False,inplace=True)

t1_team_stats = teams_df.rename(columns={'team': 'team1'})
t2_team_stats = teams_df.rename(columns={'team': 'team2'})

input_df_2 = pd.merge(todays_games, t1_team_stats, how='left', on=['team1', 'season'])
input_df_2 = pd.merge(input_df_2, t2_team_stats, how='left', on=['team2', 'season'], suffixes=('_1', '_2'))

# change teams to single column with tuples and set as index
input_df_2['teams'] = input_df_2[['team1', 'team2']].apply(tuple, axis=1)
input_df_2 = input_df_2.drop(['team1', 'team2', 'season'], axis=1)

input_df_2['xGoalsPercentage_5on5_1'] = input_df_2['xGoalsPercentage_5on5_1'].astype('float')
input_df_2['xGoalsPercentage_5on5_2'] = input_df_2['xGoalsPercentage_5on5_2'].astype('float')

# load dataset and check for errors
input_df = pd.read_csv('../data/output/input_df.csv')
# remove shootouts
input_df = input_df[input_df.result != 'shootout']

# set data and labels as X and y
X = input_df.drop(['teams', 'result', 'gameDate'], axis=1)
y = input_df['result']
Z = input_df_2.drop(['teams'], axis=1)
Z = Z[X.columns]

# encode labels
label_encoder = LabelEncoder()
label_encoded_y = label_encoder.fit_transform(y)

# scale data
scaler = StandardScaler()
scaler.fit_transform(X)

# train test split
X_train = X
y_train = y
X_test = X
y_test = y

# construct base of XGBoost model
model = xgb.XGBClassifier(
    n_jobs=-1,
    tree_method='gpu_hist',
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    
    colsample_bytree=.73,
    learning_rate=.102,
    max_depth=2,
    reg_lambda=.0775,
    subsample=.975
).fit(
    X_train, y_train,
    verbose=False,
    early_stopping_rounds=10, 
    eval_set=[(X_test, y_test)],
)

# run tests
prediction = model.predict_proba(Z) 

final_predictions = input_df_2[['teams']]
final_predictions['predictions'] = prediction.tolist()

print(final_predictions)