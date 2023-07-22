import sys
import pandas as pd
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
import urllib

from mplsoccer import Pitch, VerticalPitch, PyPizza, FontManager
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing

st.set_page_config(page_title='Lapangbola xG Dashboard')
st.header('Assign xG value to shots')
st.markdown('Created by: Prana - R&D Division Lapangbola.com')

sys.path.append("xgmodel.py")
import xgmodel
from xgmodel import calculate_xG
from xgmodel import xgfix

with st.expander("BACA INI DULU."):
    st.write("Aplikasinya kelihatan error karena kedua file yang diperlukan belum diupload, upload dulu. Untuk file timeline, pastikan tambahkan kolom X, Y, dan GW dulu. Format excelnya gak usah ada yang diganti, ya. Untuk file report excelnya, langsung upload aja, gak ada yang perlu diubah.")

col1, col2 = st.columns(2)

with col1:
    tl_data = st.file_uploader("Upload file timeline excel!")
    try:
        df_tl = pd.read_excel(tl_data, skiprows=[0])
        df_t = df_tl[['Team','Act Name','Action', 'Min', 'Sub 1', 'Sub 2', 'Sub 3', 'Sub 4', 'GW', 'X', 'Y']]
        df_t = df_t[(df_t['Action']=='shoot on target') | (df_t['Action']=='shoot off target') | (df_t['Action']=='shoot blocked') | (df_t['Action']=='goal') | (df_t['Action']=='penalty goal') | (df_t['Action']=='penalty missed')]
        df_t = df_t.reset_index()
        df_t = df_t.sort_values(by=['index'], ascending=False)
    except ValueError:
        st.error("Please upload the timeline file")


with col2:
    m_data = st.file_uploader("Upload file report excel!")
    try:
        df_m = pd.read_excel(m_data, skiprows=[0])
        team1 = df_m['Team'][0]
        team2 = df_m['Opponent'][0]
        df_m2 = df_m[['Name']]
    except ValueError:
        st.error("Please upload the excel report file")

colx, coly = st.columns(2)
with colx:
    filter = st.selectbox('Select Team', [team1, team2])
github_url = 'https://github.com/google/fonts/blob/main/ofl/poppins/Poppins-Bold.ttf'
url = github_url + '?raw=true'

response = urllib.request.urlopen(url)
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()

bold = fm.FontProperties(fname=f.name)
path_eff = [path_effects.Stroke(linewidth=2, foreground='#ffffff'),
            path_effects.Normal()]

shots = df_t

shots['Mins'] = shots['Min']
shots['Mins'] = shots['Mins'].astype(float)
shots = shots[shots['X'].notna()]

data = shots[['Act Name', 'Team', 'Action', 'Mins', 'Sub 1', 'Sub 2', 'Sub 3', 'Sub 4', 'X', 'Y']]

data.loc[(data['Action'].str.contains('penalty goal')), 'Sub 3'] = 'Penalty'
data.loc[(data['Action'].str.contains('penalty goal')), 'Sub 1'] = 'Right Foot'
data.loc[(data['Action'].str.contains('penalty missed')), 'Sub 3'] = 'Penalty'
data.loc[(data['Action'].str.contains('penalty missed')), 'Sub 2'] = 'Right Foot'
data.loc[(data['Action'].str.contains('penalty')), 'X'] = 90
data.loc[(data['Action'].str.contains('penalty')), 'Y'] = 50
#data.loc[(data['Action'].str.contains('penalty missed')) & ((data['Sub 1'].str.contains('Saved')) | (data['Sub 1'].str.contains('Cleared'))), 'Action'] = 'shoot on target'
#data.loc[(data['Action'].str.contains('penalty missed')) & ((data['Sub 1'].str.contains('Woodwork')) | (data['Sub 1'].str.contains('High')) | (data['Sub 1'].str.contains('Wide'))), 'Action'] = 'shoot off target'

data['Action'] = data['Action'].replace(['shoot on target','shoot off target','shoot blocked','goal','penalty goal','penalty missed'],
                                        ['Shot On','Shot Off','Shot Blocked','Goal','Goal','Shot Off'])
dft = data.groupby('Action', as_index=False)
temp = pd.DataFrame(columns = ['Player', 'Team','Event','Mins',
                                 'Shot Type','Situation','X','Y'])
if (('Goal' in data['Action'].unique()) == True):
  df1 = dft.get_group('Goal')
  df1f = df1[['Act Name', 'Team', 'Action', 'Mins', 'Sub 1', 'Sub 3', 'X', 'Y']]
  df1f.rename(columns = {'Action':'Event', 'Sub 1':'Shot Type', 'Sub 3':'Situation', 'Act Name':'Player'}, inplace = True)
else:
  df1f = temp.copy()

if (('Shot On' in data['Action'].unique()) == True):
  df2 = dft.get_group('Shot On')
  df2f = df2[['Act Name','Team', 'Action', 'Mins', 'Sub 2', 'Sub 3', 'X', 'Y']]
  df2f.rename(columns = {'Action':'Event', 'Sub 2':'Shot Type', 'Sub 3':'Situation', 'Act Name':'Player'}, inplace = True)
else:
  df2f = temp.copy()

if (('Shot Off' in data['Action'].unique()) == True):
  df3 = dft.get_group('Shot Off')
  df3f = df3[['Act Name','Team', 'Action', 'Mins', 'Sub 3', 'Sub 4', 'X', 'Y']]
  df3f.rename(columns = {'Action':'Event', 'Sub 3':'Shot Type', 'Sub 4':'Situation', 'Act Name':'Player'}, inplace = True)
else:
  df3f = temp.copy()

if (('Shot Blocked' in data['Action'].unique()) == True):
  df4 = dft.get_group('Shot Blocked')
  df4f = df4[['Act Name','Team', 'Action', 'Mins', 'Sub 2', 'Sub 3', 'X', 'Y']]
  df4f.rename(columns = {'Action':'Event', 'Sub 2':'Shot Type', 'Sub 3':'Situation', 'Act Name':'Player'}, inplace = True)
else:
  df4f = temp.copy()

sa = pd.concat([df1f, df2f, df3f, df4f])
#sa = pd.concat([df2f, df3f, df4f])
sa = sa.dropna()
sa.loc[(sa['Situation'].str.contains('Open play')), 'Situation'] = 'Open Play'
sa.loc[(sa['Situation'].str.contains('Freekick')), 'Situation'] = 'Set-Piece Free Kick'
sa.loc[(sa['Shot Type'].str.contains('Header')), 'Shot Type'] = 'Head'

#########
df_co = sa.sort_values(by=['Mins'], ascending=False)

df_co['x'] = 100-df_co['X']
df_co['y'] = df_co['Y']
df_co['c'] = abs(df_co['Y']-50)

x=df_co['x']*1.05
y=df_co['c']*0.68

df_co['X2']=(100-df_co['X'])*1.05
df_co['Y2']=df_co['Y']*0.68

df_co['Distance'] = np.sqrt(x**2 + y**2)
c=7.32
a=np.sqrt((y-7.32/2)**2 + x**2)
b=np.sqrt((y+7.32/2)**2 + x**2)
k = (c**2-a**2-b**2)/(-2*a*b)
gamma = np.arccos(k)
if gamma.size<0:
  gamma = np.pi + gamma
df_co['Angle Rad'] = gamma
df_co['Angle Degrees'] = gamma*180/np.pi

def wasitgoal(row):
  if row['Event'] == 'Goal':
    return 1
  else:
    return 0
df_co['goal'] = df_co.apply(lambda row: wasitgoal(row), axis=1)

df_co = df_co.sort_values(by=['Mins'])
df_co = df_co.reset_index()

shotdata = df_co[['Player','Team','Event','Mins','Shot Type','Situation','X2','Y2','Distance','Angle Rad','Angle Degrees','goal']]

########
shots = shotdata.dropna()
shots.rename(columns = {'Player':'player', 'Event':'event', 'Mins':'mins', 'Shot Type':'shottype',
                        'Situation':'situation','X2':'X','Y2':'Y', 'Distance':'distance',
                        'Angle Rad':'anglerad','Angle Degrees':'angledeg'}, inplace = True)

body_part_list=[]
for index, rows in shots.iterrows():
    if (rows['shottype']=='Right Foot') or (rows['shottype']=='Left Foot'):
        body_part_list.append('Foot')
    elif (rows['shottype']=='Head'):
        body_part_list.append('Head')
    else:
        body_part_list.append('Head')

shots['body_part']=body_part_list

shots=shots[shots.body_part != 'Other']
shots=shots.sort_values(by=['mins'])
shots.loc[(shots['situation'].str.contains('Set')), 'situation'] = 'Indirect'
shots.loc[(shots['situation'].str.contains('Corner')), 'situation'] = 'Indirect'
shots.loc[(shots['situation'].str.contains('Throw')), 'situation'] = 'Open Play'
shots.loc[(shots['situation'].str.contains('Counter')), 'situation'] = 'Open Play'
#shots = shots[(shots['event']!='Shot Blocked')].reset_index() #jangan lupa dihapus nanti
dfxg = shots[['distance', 'angledeg', 'body_part', 'situation', 'goal']]

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dfxg["body_part"])
dfxg['body_part_num'] = label_encoder.transform(dfxg["body_part"])

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(dfxg["situation"])
dfxg['situation_num'] = label_encoder.transform(dfxg["situation"])

#assigning the xG
xG=dfxg.apply(calculate_xG, axis=1) 
dfxg = dfxg.assign(xG=xG)
dfxg['xG'] = dfxg.apply(lambda row: xgfix(row), axis=1)

fixdata = df_co[['Player', 'Team', 'Event', 'Mins', 'X', 'Y']]
fixdata['xG'] = dfxg['xG']

tempdata = fixdata[['Player', 'Team', 'xG']]
tempdata = tempdata.groupby(['Player', 'Team'], as_index=False).sum()
tempdata = tempdata.rename(columns={'Player':'Name'})

findata = pd.merge(df_m2,tempdata,on='Name',how='left')
findata['xG'].fillna(0, inplace=True)
findata['xG'] = round(findata['xG'],2)

with coly:
    df_players = fixdata[fixdata['Team']==filter].reset_index(drop=True)
    pilter = st.selectbox('Select Player', pd.unique(df_players['Player']))
    all_players = st.checkbox('Select All Players')

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')
csv = convert_df(findata)

st.download_button(label='Download Data Excel+xG!',
                   data=csv,
                   file_name='Player+xG_'+team1+'vs'+team2+'.csv',
                   mime='text/csv')

#Attempts Map
fig, ax = plt.subplots(figsize=(20, 20), dpi=500)
pitch = VerticalPitch(half=True, pitch_type='wyscout', corner_arcs=True,
                      pitch_color='#ffffff', line_color='#000000',
                      stripe_color='#fcf8f7', goal_type='box', pad_bottom=5,
                      pad_right=0.5, pad_left=0.5, stripe=True, linewidth=3.5)
pitch.draw(ax=ax)

df_team = fixdata[fixdata['Team'] == filter].reset_index(drop=True)
goal = df_team[df_team['Event']=='Goal']['Event'].count()
son = df_team[df_team['Event']=='Shot On']['Event'].count()
soff = df_team[df_team['Event']=='Shot Off']['Event'].count()
sblocked = df_team[df_team['Event']=='Shot Blocked']['Event'].count()
xgtot = round((df_team['xG'].sum()),2)

df_player = df_players[df_players['Player'] == pilter].reset_index(drop=True)
goalp = df_player[df_player['Event']=='Goal']['Event'].count()
shots = df_player[df_player['Event']!='Goal']['Event'].count() + goalp
xgtotp = round((df_player['xG'].sum()),2)
gps = round((goalp/shots)*100,1)
xgps = round((xgtotp/shots),2)

if all_players:
    for i in range(len(df_team)):
      if (df_team['Event'][i] == 'Goal' or df_team['Event'][i] == 'Penalty Goal'):
        ax.scatter(df_team['Y'][i], df_team['X'][i], s=df_team['xG'][i]*10000,
                   c='#7ed957', marker='o', edgecolors='#000000', lw=3.5)
      elif (df_team['Event'][i] == 'Shot On'):
        ax.scatter(df_team['Y'][i], df_team['X'][i], s=df_team['xG'][i]*10000,
                   c='#f2ff00', marker='o', edgecolors='#000000', lw=3.5)
      elif (df_team['Event'][i] == 'Shot Off'):
        ax.scatter(df_team['Y'][i], df_team['X'][i], s=df_team['xG'][i]*10000,
                   c='#a6a6a6', marker='o', edgecolors='#000000', lw=3.5)
      else:
        ax.scatter(df_team['Y'][i], df_team['X'][i], s=df_team['xG'][i]*10000,
                   c='#e66009', marker='o', edgecolors='#000000', lw=3.5)

    annot_texts = ['Goals', 'Shots\nOn Target', 'Shots\nOff Target', 'Shots\nBlocked', 'xG Total']
    annot_x = [10.83 + x*17.83 for x in range(0,5)]
    annot_stats = [goal, son, soff, sblocked, xgtot]

    for x, s, h in zip(annot_x, annot_texts, annot_stats):
      #ax.add_patch(FancyBboxPatch((x, 62), 7, 3.5, fc='#ffffff', ec='#ffffff', lw=2))
      ax.annotate(text=s, size=22, xy=(x+3.5, 56.5), xytext=(0,-18),
                  textcoords='offset points', color='black', ha='center',
                  zorder=9, va='center', fontproperties=bold, path_effects=path_eff)
      ax.annotate(text=h, size=78, xy=(x+3.5, 60), xytext=(0,-18),
                  textcoords='offset points', color='black', ha='center',
                  zorder=9, va='center', fontproperties=bold, path_effects=path_eff)

    ax.add_patch(FancyBboxPatch((0, 45), 200, 4.5, fc='#ffffff', ec='#ffffff', lw=2))

    annot_x = [4 + x*25 for x in range(0,4)]
    annot_texts = ['Goals', 'Shots On Target', 'Shots Off Target', 'Shots Blocked']

    ax.scatter(4, 48, s=800, c='#7ed957', lw=3.5,
               marker='o', edgecolors='#000000')
    ax.scatter(29, 48, s=800, c='#f2ff00', lw=3.5,
               marker='o', edgecolors='#000000')
    ax.scatter(54, 48, s=800, c='#a6a6a6', lw=3.5,
               marker='o', edgecolors='#000000')
    ax.scatter(79, 48, s=800, c='#e66009', lw=3.5,
               marker='o', edgecolors='#000000')

    for x, s in zip(annot_x, annot_texts):
      ax.annotate(text=s, size=24, xy=(x+2.5, 49), xytext=(0,-18),
                  textcoords='offset points', color='black', ha='left',
                  zorder=9, va='center', fontproperties=bold)

    ax.add_patch(FancyBboxPatch((0.65, 50.5), 35, 1.35, fc='#cbfd06', ec='#cbfd06', lw=2))
    ax.annotate(text=filter, size=26, xy=(1, 52), xytext=(0,-18),
                textcoords='offset points', color='black', ha='left',
                zorder=9, va='center', fontproperties=bold)

    ax.annotate(text='-Nilai xG->', size=21, xy=(87, 54), xytext=(0,-18),
                textcoords='offset points', color='black', ha='left',
                zorder=9, va='center', fontproperties=bold, path_effects=path_eff)
    ax.scatter(87.5, 51.15, s=300, c='#a6a6a6', lw=2,
               marker='o', edgecolors='#000000')
    ax.scatter(90.5, 51.25, s=500, c='#a6a6a6', lw=2,
               marker='o', edgecolors='#000000')
    ax.scatter(93.5, 51.35, s=700, c='#a6a6a6', lw=2,
               marker='o', edgecolors='#000000')
    ax.scatter(97, 51.45, s=900, c='#a6a6a6', lw=2,
               marker='o', edgecolors='#000000')
    fig.savefig('smap.jpg', dpi=500, bbox_inches='tight')
    st.pyplot(fig)
    
else:
    for i in range(len(df_player)):
      if (df_player['Event'][i] == 'Goal' or df_player['Event'][i] == 'Penalty Goal'):
        ax.scatter(df_player['Y'][i], df_player['X'][i], s=df_player['xG'][i]*10000,
                   c='#7ed957', marker='o', edgecolors='#000000', lw=3.5)
      elif (df_player['Event'][i] == 'Shot On'):
        ax.scatter(df_player['Y'][i], df_player['X'][i], s=df_player['xG'][i]*10000,
                   c='#f2ff00', marker='o', edgecolors='#000000', lw=3.5)
      elif (df_player['Event'][i] == 'Shot Off'):
        ax.scatter(df_player['Y'][i], df_player['X'][i], s=df_player['xG'][i]*10000,
                   c='#a6a6a6', marker='o', edgecolors='#000000', lw=3.5)
      else:
        ax.scatter(df_player['Y'][i], df_player['X'][i], s=df_player['xG'][i]*10000,
                   c='#e66009', marker='o', edgecolors='#000000', lw=3.5)

    annot_texts = ['Goals', 'xG', 'Shots', 'Conversion\nRatio (%)', 'xG/Shots']
    annot_x = [10.83 + x*17.83 for x in range(0,5)]
    annot_stats = [goalp, xgtotp, shots, gps, xgps]

    for x, s, h in zip(annot_x, annot_texts, annot_stats):
      #ax.add_patch(FancyBboxPatch((x, 62), 7, 3.5, fc='#ffffff', ec='#ffffff', lw=2))
      ax.annotate(text=s, size=22, xy=(x+3.5, 56.5), xytext=(0,-18),
                  textcoords='offset points', color='black', ha='center',
                  zorder=9, va='center', fontproperties=bold, path_effects=path_eff)
      ax.annotate(text=h, size=78, xy=(x+3.5, 60), xytext=(0,-18),
                  textcoords='offset points', color='black', ha='center',
                  zorder=9, va='center', fontproperties=bold, path_effects=path_eff)

    ax.add_patch(FancyBboxPatch((0, 45), 200, 4.5, fc='#ffffff', ec='#ffffff', lw=2))

    annot_x = [4 + x*25 for x in range(0,4)]
    annot_texts = ['Goals', 'Shots On Target', 'Shots Off Target', 'Shots Blocked']

    ax.scatter(4, 48, s=800, c='#7ed957', lw=3.5,
               marker='o', edgecolors='#000000')
    ax.scatter(29, 48, s=800, c='#f2ff00', lw=3.5,
               marker='o', edgecolors='#000000')
    ax.scatter(54, 48, s=800, c='#a6a6a6', lw=3.5,
               marker='o', edgecolors='#000000')
    ax.scatter(79, 48, s=800, c='#e66009', lw=3.5,
               marker='o', edgecolors='#000000')

    for x, s in zip(annot_x, annot_texts):
      ax.annotate(text=s, size=24, xy=(x+2.5, 49), xytext=(0,-18),
                  textcoords='offset points', color='black', ha='left',
                  zorder=9, va='center', fontproperties=bold)

    ax.add_patch(FancyBboxPatch((0.65, 50.5), 45, 1.35, fc='#cbfd06', ec='#cbfd06', lw=2))
    ax.annotate(text=pilter, size=26, xy=(1, 52), xytext=(0,-18),
                textcoords='offset points', color='black', ha='left',
                zorder=9, va='center', fontproperties=bold)

    ax.annotate(text='-Nilai xG->', size=21, xy=(87, 54), xytext=(0,-18),
                textcoords='offset points', color='black', ha='left',
                zorder=9, va='center', fontproperties=bold, path_effects=path_eff)
    ax.scatter(87.5, 51.15, s=300, c='#a6a6a6', lw=2,
               marker='o', edgecolors='#000000')
    ax.scatter(90.5, 51.25, s=500, c='#a6a6a6', lw=2,
               marker='o', edgecolors='#000000')
    ax.scatter(93.5, 51.35, s=700, c='#a6a6a6', lw=2,
               marker='o', edgecolors='#000000')
    ax.scatter(97, 51.45, s=900, c='#a6a6a6', lw=2,
               marker='o', edgecolors='#000000')
    fig.savefig('smap.jpg', dpi=500, bbox_inches='tight')
    st.pyplot(fig)
    
with open('smap.jpg', 'rb') as img:
          fn = 'AttemptsMap_'+filter+'.jpg'
          btn = st.download_button(label="Download Attempts Map!", data=img,
                                   file_name=fn, mime="image/jpg")
