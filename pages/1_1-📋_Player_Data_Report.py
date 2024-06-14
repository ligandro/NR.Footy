
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, NoNorm
from matplotlib import cm
import matplotlib.gridspec as gridspec
import numpy as np
from mplsoccer import PyPizza, add_image, FontManager

from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib
import json
import yaml

from scipy.stats import zscore
import pandas as pd
from scipy.stats import rankdata

# Bar graphs
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from matplotlib import cm
from highlight_text import fig_text, ax_text
from matplotlib.colors import LinearSegmentedColormap, NoNorm
from matplotlib import cm
import matplotlib.gridspec as gridspec
import numpy as np
from mplsoccer import PyPizza, add_image, FontManager
from mplsoccer import Pitch, VerticalPitch
import cmasher as cmr
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from PIL import Image
import urllib
import json
import os
import math
from selenium.webdriver.chrome.service import Service
#import modules and packages
import requests
from bs4 import BeautifulSoup
import json
import datetime
from ast import literal_eval
from scipy import stats
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon



from matplotlib.colors import LinearSegmentedColormap  
pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                       ['#15242e', '#4393c4'], N=10)

red_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                       ['#FFBEBE', '#FF0000'], N=10)
el_greco_violet_cmap = LinearSegmentedColormap.from_list("El Greco Violet - 10 colors",
                                                         ['#332a49', '#8e78a0'], N=10)
el_greco_yellow_cmap = LinearSegmentedColormap.from_list("El Greco Yellow - 10 colors",
                                                         ['#7c2e2a', '#f2dd44'], N=10)
flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 10 colors",
                                                  ['#e3aca7', '#c03a1d'], N=10)
# same color maps but with 100 colors
pearl_earring_cmap_100 = LinearSegmentedColormap.from_list("Pearl Earring - 100 colors",
                                                           ['#15242e', '#4393c4'], N=100)
el_greco_violet_cmap_100 = LinearSegmentedColormap.from_list("El Greco Violet - 100 colors",
                                                             ['#3b3154', '#8e78a0'], N=100)
el_greco_yellow_cmap_100 = LinearSegmentedColormap.from_list("El Greco Yellow - 100 colors",
                                                             ['#7c2e2a', '#f2dd44'], N=100)
flamingo_cmap_100 = LinearSegmentedColormap.from_list("Flamingo - 100 colors",
                                                      ['#e3aca7', '#c03a1d'], N=100)

from mplsoccer import Radar

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['_grid_dimensions', '_draw_grid', 'grid', 'grid_dimensions']


def _grid_dimensions(ax_aspect=1, figheight=9, nrows=1, ncols=1,
                     grid_height=0.715, grid_width=0.95, space=0.05,
                     left=None, bottom=None,
                     endnote_height=0, endnote_space=0.01,
                     title_height=0, title_space=0.01,
                     ):

    # dictionary for holding dimensions
    dimensions = {'figheight': figheight, 'nrows': nrows, 'ncols': ncols,
                  'grid_height': grid_height, 'grid_width': grid_width,
                  'title_height': title_height, 'endnote_height': endnote_height,
                  }

    if left is None:
        left = (1 - grid_width) / 2

    if title_height == 0:
        title_space = 0

    if endnote_height == 0:
        endnote_space = 0

    error_msg_height = ('The axes extends past the figure height. '
                        'Reduce one of the bottom, endnote_height, endnote_space, grid_height, '
                        'title_space or title_height so the total is ≤ 1.')
    error_msg_width = ('The grid axes extends past the figure width. '
                       'Reduce one of the grid_width or left so the total is ≤ 1.')

    axes_height = (endnote_height + endnote_space + grid_height +
                   title_height + title_space)
    if axes_height > 1:
        raise ValueError(error_msg_height)

    if bottom is None:
        bottom = (1 - axes_height) / 2

    if bottom + axes_height > 1:
        raise ValueError(error_msg_height)

    if left + grid_width > 1:
        raise ValueError(error_msg_width)

    dimensions['left'] = left
    dimensions['bottom'] = bottom
    dimensions['title_space'] = title_space
    dimensions['endnote_space'] = endnote_space

    if (nrows > 1) and (ncols > 1):
        dimensions['figwidth'] = figheight * grid_height / grid_width * (((1 - space) * ax_aspect *
                                                                          ncols / nrows) +
                                                                         (space * (ncols - 1) / (
                                                                                 nrows - 1)))
        dimensions['spaceheight'] = grid_height * space / (nrows - 1)
        dimensions['spacewidth'] = dimensions['spaceheight'] * figheight / dimensions['figwidth']
        dimensions['axheight'] = grid_height * (1 - space) / nrows

    elif (nrows > 1) and (ncols == 1):
        dimensions['figwidth'] = figheight * grid_height / grid_width * (
                1 - space) * ax_aspect / nrows
        dimensions['spaceheight'] = grid_height * space / (nrows - 1)
        dimensions['spacewidth'] = 0
        dimensions['axheight'] = grid_height * (1 - space) / nrows

    elif (nrows == 1) and (ncols > 1):
        dimensions['figwidth'] = figheight * grid_height / grid_width * (space + ax_aspect * ncols)
        dimensions['spaceheight'] = 0
        dimensions['spacewidth'] = grid_height * space * figheight / dimensions['figwidth'] / (
                ncols - 1)
        dimensions['axheight'] = grid_height

    else:  # nrows=1, ncols=1
        dimensions['figwidth'] = figheight * grid_height * ax_aspect / grid_width
        dimensions['spaceheight'] = 0
        dimensions['spacewidth'] = 0
        dimensions['axheight'] = grid_height

    dimensions['axwidth'] = dimensions['axheight'] * ax_aspect * figheight / dimensions['figwidth']

    return dimensions


def _draw_grid(dimensions, left_pad=0, right_pad=0, axis=True, grid_key='grid'):

    dims = dimensions
    bottom_coordinates = np.tile(dims['spaceheight'] + dims['axheight'],
                                 reps=dims['nrows'] - 1).cumsum()
    bottom_coordinates = np.insert(bottom_coordinates, 0, 0.)
    bottom_coordinates = np.repeat(bottom_coordinates, dims['ncols'])
    grid_bottom = dims['bottom'] + dims['endnote_height'] + dims['endnote_space']
    bottom_coordinates = bottom_coordinates + grid_bottom
    bottom_coordinates = bottom_coordinates[::-1]

    left_coordinates = np.tile(dims['spacewidth'] + dims['axwidth'],
                               reps=dims['ncols'] - 1).cumsum()
    left_coordinates = np.insert(left_coordinates, 0, 0.)
    left_coordinates = np.tile(left_coordinates, dims['nrows'])
    left_coordinates = left_coordinates + dims['left']

    fig = plt.figure(figsize=(dims['figwidth'], dims['figheight']))
    axs = []
    for idx, bottom_coord in enumerate(bottom_coordinates):
        axs.append(fig.add_axes((left_coordinates[idx], bottom_coord,
                                 dims['axwidth'], dims['axheight'])))
    axs = np.squeeze(np.array(axs).reshape((dims['nrows'], dims['ncols'])))
    if axs.size == 1:
        axs = axs.item()
    result_axes = {grid_key: axs}

    title_left = dims['left'] + left_pad
    title_width = dims['grid_width'] - left_pad - right_pad

    if dims['title_height'] > 0:
        ax_title = fig.add_axes(
            (title_left, grid_bottom + dims['grid_height'] + dims['title_space'],
             title_width, dims['title_height']))
        if axis is False:
            ax_title.axis('off')
        result_axes['title'] = ax_title

    if dims['endnote_height'] > 0:
        ax_endnote = fig.add_axes((title_left, dims['bottom'],
                                   title_width, dims['endnote_height']))
        if axis is False:
            ax_endnote.axis('off')
        result_axes['endnote'] = ax_endnote

    if dims['title_height'] == 0 and dims['endnote_height'] == 0:
        return fig, result_axes[grid_key]  # no dictionary if just grid
    return fig, result_axes  # else dictionary


def grid(ax_aspect=1, figheight=9, nrows=1, ncols=1,
         grid_height=0.715, grid_width=0.95, space=0.05,
         left=None, bottom=None,
         endnote_height=0, endnote_space=0.01,
         title_height=0, title_space=0.01, axis=True, grid_key='grid'):
   
 
    dimensions = _grid_dimensions(ax_aspect=ax_aspect, figheight=figheight, nrows=nrows,
                                  ncols=ncols,
                                  grid_height=grid_height, grid_width=grid_width, space=space,
                                  left=left, bottom=bottom,
                                  endnote_height=endnote_height, endnote_space=endnote_space,
                                  title_height=title_height, title_space=title_space,
                                  )
    fig, ax = _draw_grid(dimensions, axis=axis, grid_key=grid_key)
    return fig, ax



def grid_dimensions(ax_aspect, figwidth, figheight, nrows, ncols, max_grid, space):
  
    if ncols > 1 and nrows == 1:
        grid1 = max_grid * figheight / figwidth * (space + ax_aspect * ncols)
        grid2 = max_grid / figheight * figwidth / (space + ax_aspect * ncols)
    elif ncols > 1 or nrows > 1:
        extra = space * (ncols - 1) / (nrows - 1)
        grid1 = max_grid * figheight / figwidth * (((1 - space) * ax_aspect *
                                                    ncols / nrows) + extra)
        grid2 = max_grid / figheight * figwidth / (((1 - space) * ax_aspect *
                                                    ncols / nrows) + extra)
    else:  # nrows=1, ncols=1
        grid1 = max_grid * figheight / figwidth * ax_aspect
        grid2 = max_grid / figheight * figwidth / ax_aspect

    # decide whether the max_grid is the grid_width or grid_height and set the other value
    if (grid1 > 1) | ((grid2 >= grid1) & (grid2 <= 1)):
        return max_grid, grid2
    return grid1, max_grid

from PIL import Image
import urllib
import json
import os
import math

#import modules and packages
import requests
from bs4 import BeautifulSoup
import json
import datetime



st.markdown("""<p style="font-family:Century Gothic; color:white; font-size: 60px; font-weight: bold;">PLAYER DATA REPORT</p>""",unsafe_allow_html=True)

st.markdown(
    """
    <p style="font-family:Century Gothic; color:white; font-size: 18px;">
    ➢Create Player Data Report for any player in any position 
    <p style="font-family:Century Gothic; color:white; font-size: 18px;">➢Select Position and Minimum Minutes accordingly
    <p style="font-family:Century Gothic; color:white; font-size: 18px;">NOTE : If error occurs,it means player is not in data or minimum minutes and position are wrong
    <p style="font-family:Century Gothic; color:white; font-size: 18px;">NOTE : For now minimum percentile cutoff for columns is set at 70, meaning you can only plot columns where a player's stat percentile is greater than 70

    </p>
    """,unsafe_allow_html=True
)


# Custom styled markdown for the upload prompt
st.markdown("""
    <style>
    .custom-title {
        font-family: Century Gothic;
        color: white;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 2px; /* Adjust the margin-bottom to reduce the gap */
    }
    </style>
    <p class="custom-title">Upload the data</p>
    """, unsafe_allow_html=True)

# File uploader widget with adjusted styling to reduce gap
st.markdown("""
    <style>
    .uploaded-file {
        margin-top: -20px; /* Adjust the margin-top to reduce the gap */
    }
    </style>
    """, unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="xlsx", key="fileUploader", label_visibility="collapsed")

if uploaded_file is not None:
    # Read the file to a dataframe using pandas
    liga = pd.read_excel(uploaded_file)
    if st.checkbox("Display Data"):
        st.write(liga)

st.markdown("""
    <style>
    .slider-label {
        font-family: Century Gothic;
        font-size: 22px;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<span class='slider-label'>Set Minimum Minutes Played</span>", unsafe_allow_html=True)
minutes = st.slider("", 0, 5000, format=None, key=None)

st.markdown("""
    <style>
    .slider-label {
        font-family: Century Gothic;
        font-size: 22px;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<span class='slider-label'>Set Age Limit</span>", unsafe_allow_html=True)
age = st.slider("", 15, 45, format=None, key=None)

pos_options = ("Forward", "Midfielder", "Defender","Goalkeeper")
st.markdown("""
    <style>
    .slider-label {
        font-family: Century Gothic;
        font-size: 22px;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<span class='slider-label'>Select Position</span>", unsafe_allow_html=True)
pos = st.radio("", pos_options, index=0)

if pos == "Forward":
    forward_options = ("Winger","Central Striker")
    sec_pos = st.selectbox("Select Forward Position", forward_options, index=0, help="Select the forward position")
elif pos == "Midfielder":
    midfielder_options = ("Attacking Mid", "Central Mid", "Defensive Midfielder")
    sec_pos = st.selectbox("Select Midfielder Position", midfielder_options, index=0, help="Select the midfielder position")
elif pos == "Defender":
    defender_options = ("Center Back", "Full Back")
    sec_pos = st.selectbox("Select Defender Position", defender_options, index=0, help="Select the defender position")
elif pos == "Goalkeeper":
    sec_pos = "Goalkeeper"

defend  = liga[liga["Minutes played"] >= minutes]
defend = defend[defend["Age"] <= age] 
if sec_pos == 'Central Striker':
    defend = defend[defend['Position'].isin(["CF"])]
elif sec_pos == 'Winger':
    wings = ["RW","LWF",'RWF','LW']
    defend = defend[defend['Position'].apply(lambda x: any(pos in x.split(',') for pos in wings))]
elif sec_pos == 'Attacking Mid':
    defend = defend[ defend['Position'].str.contains('AMF') ]
elif sec_pos == 'Central Mid':
    defend = defend[defend['Position'].str.split(',').str[0].str.contains('CM')]
elif sec_pos == 'Defensive Midfielder':
    defend = defend[ defend['Position'].str.contains('D') ] 
elif sec_pos == 'Center Back':
    defend = defend[ defend['Position'].str.contains('CB') ]
elif sec_pos == 'Full Back':
    full_backs = ["RB",'LB','RWB','LWB']
    defend = defend[defend['Position'].apply(lambda x: any(pos in x.split(',') for pos in full_backs))]
elif sec_pos == 'Goalkeeper':
    defend = defend[defend['Position'].isin(["GK"])]

col_list = (defend.Player.unique()).tolist()
st.markdown("""
    <style>
    .slider-label {
        font-family: Century Gothic;
        font-size: 22px;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<span class='slider-label'>Select Name of Player</span>", unsafe_allow_html=True)
player_name = st.selectbox("",options = col_list)


if sec_pos == 'Central Striker':
    defend["Succ Dribbles per 90"] = defend['Dribbles per 90'] *defend['Successful dribbles, %'] * 0.01
    defend["Acc Passes to final3rd per 90"] = defend['Passes to final third per 90'] *defend['Accurate passes to final third, %'] * 0.01
    defend["Acc Passes PenArea per 90"] = defend['Passes to penalty area per 90'] *defend['Accurate passes to penalty area, %'] * 0.01
    defend["Acc Through Passes per 90"] = defend['Through passes per 90'] *defend['Accurate through passes, %'] * 0.01


    z_scores_df = defend.copy()  # Create a copy to preserve the original DataFrame
    numeric_columns = defend.select_dtypes(include='number').columns

    # Calculate Z-scores using the correct standard deviation for each column
    for col in numeric_columns:
        z_scores_df[col] = zscore(defend[col])

    # Min-Max scaling to scale Z-scores from 0 to 100
    z_scores_df = ((z_scores_df[numeric_columns] - z_scores_df[numeric_columns].min()) / 
                (z_scores_df[numeric_columns].max() - z_scores_df[numeric_columns].min())) * 100
    # Print the resulting DataFrame with Z-scores
    # Combine scaled values with non-numeric columns
    z_scores_df = pd.concat([defend[['Player','Team']], z_scores_df], axis=1)


    # 1. Play Making : xA,key passes,through balls.assists
    weights = [0.1,0.2,0.4,0.2,0.1]
    z_scores_df["Play_Making"] = (z_scores_df['Assists per 90'] * weights[0] +
                                z_scores_df['xA per 90'] * weights[1] +
                                z_scores_df['Key passes per 90'] * weights[2]+z_scores_df['Acc Through Passes per 90'] * weights[3]+
                                z_scores_df['Acc Passes PenArea per 90'] * weights[4]
                                )
    # 1.  Goal_Threat : 
    weights = [0.5,0.3,0.1,0.1]
    z_scores_df["Goal_Threat"] = (z_scores_df['Non-penalty goals per 90'] * weights[0] +
                                z_scores_df['xG per 90'] * weights[1] +
                                z_scores_df['Shots per 90'] * weights[2]+z_scores_df['Shots on target, %'] * weights[3])
    # 1.  Dribbling : 
    weights = [0.2,0.4,0.2,0.2]

    z_scores_df["Possession"] = (z_scores_df['Touches in box per 90'] * weights[0] +
                                z_scores_df['Succ Dribbles per 90'] * weights[1]+z_scores_df['Progressive runs per 90'] * weights[2] +
                                z_scores_df['Successful attacking actions per 90'] * weights[3] )


    # Combined Stats
    weights = [0.15,0.7,0.15]
    z_scores_df["Striker_Score"] = (z_scores_df['Play_Making'] * weights[0] +
                                z_scores_df['Goal_Threat'] * weights[1] +
                                z_scores_df['Possession'] * weights[2])

    # Min-Max scaling to scale values to a ranking between 50 and 100
    min_value = z_scores_df["Striker_Score"] .min()
    max_value = z_scores_df["Striker_Score"] .max()

    z_scores_df["Striker_Score"]  = 50 + ((z_scores_df["Striker_Score"]  - min_value) / (max_value - min_value)) * 50


    ratingss = z_scores_df.Striker_Score

    defend['Striker_Score'] = ratingss

    mean_values = defend.select_dtypes(include='number').mean().to_frame().T

    # Set a name for the index to distinguish the mean row
    mean_values.index = ['mean']

    # Concatenate the mean values as a new row
    defend = pd.concat([defend, mean_values], ignore_index=True)

    defend = defend.fillna("Avg")
    last_index = len(defend) - 1
    defend.iloc[last_index, 3] = 5000

    # defend.drop_duplicates(subset=['Player', 'Team','Position'])

    defend_p = defend.copy()
    # Function to calculate percentiles
    def calculate_percentiles(series):
        return series.rank(pct=True) * 100

    # Select all numerical columns
    numerical_columns = defend_p.select_dtypes(include='number').columns

    # Apply the percentile calculation to all numerical columns
    defend_p[numerical_columns] = defend_p[numerical_columns].apply(calculate_percentiles)


    filter_prem = defend_p.copy()
    # prem = prem[prem["Age"] < 30]
    x =player_name
    player_fil = filter_prem.loc[filter_prem['Player']==x]



    # Select all numerical columns
    numerical_columns = player_fil.select_dtypes(include='number').columns

    # Set percentile filter
    # min_percentile = st.slider("Set minimum percentile cutoff", 0, 99, format=None, key=None)
    min_percentile = 70
    player_fil = player_fil[player_fil[numerical_columns] > min_percentile].dropna(axis=1)


    req_columns = player_fil.columns


    columns = req_columns
    selected_columns = []

    st.markdown("""
        <style>
        .slider-label {
            font-family: Century Gothic;
            font-size: 22px;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<span class='slider-label'>Select Columns to plot</span>", unsafe_allow_html=True)

    for col in columns:
        selected = st.checkbox(col, value=False, key=col)
        if selected:
            selected_columns.append(col)

    # st.write("Selected columns:", selected_columns)
    status1 = [ "No","Yes"]
    status = st.radio("Show Plot",horizontal=True,options = status1)

    if status =="Yes":

        req_columns = selected_columns
                # Radar
        color_1 = "#F98866"
        color_2 = "#408EC6"
        x =	player_name
        prem = defend.copy()
        player = prem.loc[prem['Player']==x]
        player2 = prem.loc[prem['Player']== "Avg"]

        Name = str(player.iloc[0,0])

        kik = prem.copy()

        stat1 = []
        stat2 = []
        for x in req_columns:
            stat1.extend(player[x])
            stat2.extend(player2[x])

        params = req_columns


        lower_is_better = []
        # minimum range value and maximum range value for parameters
        min_range= []
        max_range =[]
        for x in req_columns:
            min_range.append(kik.loc[:,x].min())
            max_range.append(kik.loc[:,x].max())          
        stat11 = [ round(x, 2) for x in stat1]        
        stat22 = [ round(x, 2) for x in stat2]  

        radar = Radar(params, min_range, max_range,
                    lower_is_better=lower_is_better,
                    # whether to round any of the labels to integers instead of decimal places
                    round_int=[False]*len(params),
                    num_rings=3,  # the number of concentric circles (excluding center circle)
                    # if the ring_width is more than the center_circle_radius then
                    # the center circle radius will be wider than the width of the concentric circles
                    ring_width=1, center_circle_radius=1)



        fig, ax = radar.setup_axis()
        color44 = "black"
        color45= "#000544"
        color46 ='#4FC0E8'
        fig.set_facecolor("black")
        ax.patch.set_facecolor(color44)
        radar.setup_axis(ax=ax, facecolor='None')  # format axis as a radar
        rings_inner = radar.draw_circles(ax=ax, facecolor=color44, edgecolor=color46)
        radar_output = radar.draw_radar_compare(stat11, stat22, ax=ax,
                                                kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                                kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
        radar_poly, radar_poly2, vertices1, vertices2 = radar_output
        range_labels = radar.draw_range_labels(ax=ax, fontsize=15,color='white',
                                                font="STXihei",weight="bold")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=15,color='white',
                                            font="STXihei",weight="bold")

        ax.scatter(vertices1[:, 0], vertices1[:, 1],
                                    c="#00f2c1", edgecolors='#00f2c1', marker='o', s=30, zorder=2)
        ax.scatter(vertices2[:, 0], vertices2[:, 1],
                                c='#d80499', edgecolors='#d80499', marker='o', s=30, zorder=2)

        # add title
        fig.text(
        0.515, 0.934, f"Center Forward Radar",
        path_effects=[path_effects.Stroke(linewidth=0.2, foreground="white"), path_effects.Normal()],
        ha="center", font = "Century Gothic",size =39,color="white",fontweight="bold"
        )



        # add credits
        notes = f'Stats compared only with\nplayers with minutes > {minutes}'
        fig.text(
        0.895, 0.08, f"{notes}",
        font = "Century Gothic",size =13,color="white",
        ha="right"
        )

        # Define circle properties
        center = (0.38, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#00f2c1')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.40, 0.895, f"{Name}",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )

        # Define circle properties
        center = (0.6, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#d80499')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.62, 0.895, f"Average",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )
        plt.savefig("data/images/pizza.jpg",dpi =500, bbox_inches='tight')
        # st.pyplot(fig)
        
        defend.reset_index(drop=True, inplace=True)  # Reset the index after filtering

        # select stats
        x =player_name
        player_df = defend.loc[defend['Player']==x]

        # Define the desired order of metrics
        metrics = req_columns

        # Calculate percentile ranks for each metric
        percentile_ranks = {}
        for metric in metrics:
            percentile_ranks[metric] = rankdata(defend[metric], method='average') / len(defend) * 99

        # Define colors and create a colormap
        colors = ['red', 'orange', 'yellow', 'green']
        percentile_ranges = [ 25, 50, 75, 100]
        cmap = ListedColormap(colors)


        # Create a bar graph with black background and facecolor
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
        ax.set_facecolor('black')


        ax.axvline(x=25, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=50, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=75, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=100, linestyle='--', color='white', linewidth=0.8)

        for i, metric in enumerate(metrics):
            metric_value = player_df[metric].values[0]
            percentile_rank = percentile_ranks[metric][player_df.index[0]]

            color_index = next(idx for idx, pct in enumerate(percentile_ranges) if pct >= percentile_rank)
            
            # Assign color based on the index
            color = cmap(color_index)

        # Assign color based on percentile rank
            bar = ax.barh(i, percentile_rank, height=0.3, alpha=0.7, color=color,edgecolor ="white",linewidth =0.8 ,zorder=1)
            ax.text(
                105, bar[0].get_y() + bar[0].get_height() / 2, f'{metric_value:.2f}', va='center', ha='left',
                font='STXihei', size=15, color='white'
            )

        # ax.axvline(50, linestyle='--', label='50th Percentile Rank',color='white')
        # ax.axvline(25, linestyle='--', label='25th Percentile Rank', color='white')
        # ax.axvline(75, linestyle='--', label='75th Percentile Rank', color='white')

        new_labels = req_columns


        # Set the new labels for the y-axis ticks
        ax.set_yticklabels(new_labels,font='Tw Cen MT', color='white') 

        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_yticks(range(len(metrics)))

        max_percentile_rank = max([max(percentile_ranks[metric]) for metric in metrics])
        ax.set_xlim(0, max_percentile_rank + 10)

        ax.set_xlabel('Percentile Ranks', color='white',fontname = "Tw Cen MT",size=20)
        # ax.set_ylabel('Metrics', color='white',fontname = "Tw Cen MT",size=20)
        plt.xticks(fontname = "Tw Cen MT",color="white",size=12)
        ax.set_xticks([0, 25, 50,75, 100])

        plt.yticks(fontname = "Tw Cen MT",color="white",size=15)


        ax.legend().remove()


        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.spines['left'].set_visible(False)  # Hide the left spine

        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')

        top_boxes = [
            Rectangle((i * 20, 20), 20, 5, color=colors[i], edgecolor='black', linewidth=1) for i in range(len(colors))
        ]

        texts =["Bottom 25%","From 25-50%","From 50-75%","Top 25%"]
        for i, box in enumerate(top_boxes):
            ax.add_patch(box)
            ax.text(
                i * 24 + 15, -0.8, f'{texts[i]}',font="Tw Cen MT", va='center', ha='center', color=colors[i], fontweight='bold', fontsize=25
            )

        plt.subplots_adjust(left=0.2)

        # # add credits
        # notes = f'All Stats per 90'
        # fig.text(
        # 0.27, 0.067, f"{notes}",
        # font = "Century Gothic",size =11,color="white",
        # ha="right"
        # )


        plt.savefig("data/images/bar.jpg",dpi =500, bbox_inches='tight')

        fig = plt.figure(figsize = (10,8), dpi = 300)

        fig.set_facecolor('black')
        plt.rcParams['hatch.linewidth'] = .02

        # one-liner
        player_df = defend.loc[defend['Player']==player_name]


        time = float(player_df.iloc[0,6])
        Name = str(player_df.iloc[0,0])
        Team = str(player_df.iloc[0,2])
        age =player_df.iloc[0,4]

        if len(player_name) > 20:
            fn = 14
        elif len(player_name) > 15:
            fn = 17
        else : 
            fn = 22

        im1 = plt.imread(r"C:\Users\lolen\OneDrive\Documents\Coding\Neurotactic Essentials\Images\Neuro.png")
        ax_image = add_image( im1, fig, left=1.02, bottom=0.982, width=0.13, height=0.13 )   # these values might differ when you are plotting




        # NAME

        #Heading
        str_text = f'''Player Data Report'''

        fig_text(
            x = 0.55, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="white",
            bbox=dict(boxstyle='round', facecolor='none', edgecolor='#4A9BD4', linewidth=2)
        )


        #Heading
        str_text = f'''--------------------------------------------------'''

        fig_text(
            x = 0.15, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )


        #Heading
        str_text = f'''-----------------------------------------------'''

        fig_text(
            x = 0.775, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )



        Attacking_Mid_score = z_scores_df[z_scores_df['Player']==x]["Striker_Score"].values[0]
        attack_df = z_scores_df.sort_values(by='Striker_Score', ascending=False).reset_index(drop=True)
        Attacking_Mid_rank = attack_df.index[attack_df['Player'] == x].values[0] +1 


        fig.text(
        x = 1.14, y = 0.945, s = f"Centre Forward Rating : {Attacking_Mid_score:.2f} | Rank : {Attacking_Mid_rank}/{len(defend)}",
        ha="right", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )

        # add radar
        im1 = plt.imread(r"data/images/pizza.jpg")
        ax_image = add_image(
                im1, fig, left=0.691, bottom=0.4, width=0.53, height=0.53
            )   # these values might differ when you are plotting


        im1 = plt.imread(r"data/images/bar.jpg")
        ax_image = add_image(
                im1, fig, left=0.14 ,bottom=0.4, width=0.6, height=0.6
            )   # these values might differ when you are plotting

        fig_text(
            x = 0.154, y = 1.04, 
            s = f'<{player_name}>',
            highlight_textprops=[{"color":"#FFD230"}],
            fontname ="Century Gothic",path_effects=[path_effects.Stroke(linewidth=0.4, foreground="#BD8B00"), path_effects.Normal()],
            va = 'bottom', ha = 'left',
            fontsize = fn,  weight = 'bold',color="white"
        )

        fig.text(
        x = 0.15, y = 1.024, s = f" Minutes Played : {time:.0f} | Age : {age:.0f} | Season : 23-24",
        ha="left", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )
        st.pyplot(fig)


elif sec_pos == 'Winger':
    defend["Succ Dribbles per 90"] = defend['Dribbles per 90'] *defend['Successful dribbles, %'] * 0.01
    defend["Acc Passes to final3rd per 90"] = defend['Passes to final third per 90'] *defend['Accurate passes to final third, %'] * 0.01
    defend["Acc Passes PenArea per 90"] = defend['Passes to penalty area per 90'] *defend['Accurate passes to penalty area, %'] * 0.01
    defend["Acc Through Passes per 90"] = defend['Through passes per 90'] *defend['Accurate through passes, %'] * 0.01
    defend["Acc Crosses per 90"] = defend['Crosses per 90'] *defend['Accurate crosses, %'] * 0.01
    defend["Pass Prog Acc"] =  defend['Progressive passes per 90'] * defend['Accurate progressive passes, %'] * 0.01


    z_scores_df = defend.copy()  # Create a copy to preserve the original DataFrame
    numeric_columns = defend.select_dtypes(include='number').columns

    # Calculate Z-scores using the correct standard deviation for each column
    for col in numeric_columns:
        z_scores_df[col] = zscore(defend[col])

    # Min-Max scaling to scale Z-scores from 0 to 100
    z_scores_df = ((z_scores_df[numeric_columns] - z_scores_df[numeric_columns].min()) / 
                (z_scores_df[numeric_columns].max() - z_scores_df[numeric_columns].min())) * 100
    # Print the resulting DataFrame with Z-scores
    # Combine scaled values with non-numeric columns
    z_scores_df = pd.concat([defend[['Player','Team']], z_scores_df], axis=1)

    # 1. Play Making : xA,key passes,through balls.assists
    weights = [0.1,0.2,0.4,0.2,0.1]
    z_scores_df["Play_Making"] = (z_scores_df['Assists per 90'] * weights[0] +
                                z_scores_df['xA per 90'] * weights[1] +
                                z_scores_df['Key passes per 90'] * weights[2]+z_scores_df['Acc Through Passes per 90'] * weights[3]
                                )
    # 1.  Goal_Threat : 
    weights = [0.5,0.3,0.1,0.1]
    z_scores_df["Goal_Threat"] = (z_scores_df['Non-penalty goals per 90'] * weights[0] +
                                z_scores_df['xG per 90'] * weights[1] +
                                z_scores_df['Shots per 90'] * weights[2]+z_scores_df['Shots on target, %'] * weights[3])
    # 1.  Dribbling : 
    weights = [0.2,0.4,0.2,0.1,0.1]
    z_scores_df["Possession"] = (z_scores_df['Touches in box per 90'] * weights[0] +
                                z_scores_df['Succ Dribbles per 90'] * weights[1]+z_scores_df['Progressive runs per 90'] * weights[2] +
                                z_scores_df['Successful attacking actions per 90'] * weights[3]+z_scores_df['Received passes per 90'] * weights[4]  )

    # 1.  Passing : 
    weights = [0.3,0.3,0.2,0.2]

    z_scores_df["Passing"] = (z_scores_df['Acc Passes to final3rd per 90'] * weights[0] +
                                z_scores_df['Acc Crosses per 90'] * weights[1]+z_scores_df['Acc Passes PenArea per 90'] * weights[2] +
                            z_scores_df['Deep completions per 90'] * weights[3] )


    weights = [0.3,0.2,0.4,0.3]
    z_scores_df["Winger_Score"] = (z_scores_df['Play_Making'] * weights[0] +
                                z_scores_df['Goal_Threat'] * weights[1] +
                                z_scores_df['Possession'] * weights[2]+z_scores_df['Passing'] * weights[3])


    # Min-Max scaling to scale values to a ranking between 50 and 100
    min_value = z_scores_df["Winger_Score"] .min()
    max_value = z_scores_df["Winger_Score"] .max()

    z_scores_df["Winger_Score"]  = 30 + ((z_scores_df["Winger_Score"]  - min_value) / (max_value - min_value)) * 70


    ratingss = z_scores_df.Winger_Score

    defend['Winger_Score'] = ratingss

    mean_values = defend.select_dtypes(include='number').mean().to_frame().T

    # Set a name for the index to distinguish the mean row
    mean_values.index = ['mean']

    # Concatenate the mean values as a new row
    defend = pd.concat([defend, mean_values], ignore_index=True)

    defend = defend.fillna("Avg")
    last_index = len(defend) - 1
    defend.iloc[last_index, 3] = 5000

    # defend.drop_duplicates(subset=['Player', 'Team','Position'])

    defend_p = defend.copy()
    # Function to calculate percentiles
    def calculate_percentiles(series):
        return series.rank(pct=True) * 100

    # Select all numerical columns
    numerical_columns = defend_p.select_dtypes(include='number').columns

    # Apply the percentile calculation to all numerical columns
    defend_p[numerical_columns] = defend_p[numerical_columns].apply(calculate_percentiles)


    filter_prem = defend_p.copy()
    # prem = prem[prem["Age"] < 30]
    x =player_name
    player_fil = filter_prem.loc[filter_prem['Player']==x]



    # Select all numerical columns
    numerical_columns = player_fil.select_dtypes(include='number').columns

    # Set percentile filter
    # min_percentile = st.slider("Set minimum percentile cutoff", 0, 99, format=None, key=None)
    min_percentile = 70
    player_fil = player_fil[player_fil[numerical_columns] > min_percentile].dropna(axis=1)


    req_columns = player_fil.columns


    columns = req_columns
    selected_columns = []

    st.markdown("""
        <style>
        .slider-label {
            font-family: Century Gothic;
            font-size: 22px;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<span class='slider-label'>Select Columns to plot</span>", unsafe_allow_html=True)

    for col in columns:
        selected = st.checkbox(col, value=False, key=col)
        if selected:
            selected_columns.append(col)

    # st.write("Selected columns:", selected_columns)
    status1 = [ "No","Yes"]
    status = st.radio("Show Plot",horizontal=True,options = status1)

    if status =="Yes":
        req_columns = selected_columns
                # Radar
        color_1 = "#F98866"
        color_2 = "#408EC6"
        x =	player_name
        prem = defend.copy()
        player = prem.loc[prem['Player']==x]
        player2 = prem.loc[prem['Player']== "Avg"]

        Name = str(player.iloc[0,0])

        kik = prem.copy()

        stat1 = []
        stat2 = []
        for x in req_columns:
            stat1.extend(player[x])
            stat2.extend(player2[x])

        params = req_columns


        lower_is_better = []
        # minimum range value and maximum range value for parameters
        min_range= []
        max_range =[]
        for x in req_columns:
            min_range.append(kik.loc[:,x].min())
            max_range.append(kik.loc[:,x].max())          
        stat11 = [ round(x, 2) for x in stat1]        
        stat22 = [ round(x, 2) for x in stat2]  

        radar = Radar(params, min_range, max_range,
                    lower_is_better=lower_is_better,
                    # whether to round any of the labels to integers instead of decimal places
                    round_int=[False]*len(params),
                    num_rings=3,  # the number of concentric circles (excluding center circle)
                    # if the ring_width is more than the center_circle_radius then
                    # the center circle radius will be wider than the width of the concentric circles
                    ring_width=1, center_circle_radius=1)



        fig, ax = radar.setup_axis()
        color44 = "black"
        color45= "#000544"
        color46 ='#4FC0E8'
        fig.set_facecolor(color44)
        ax.patch.set_facecolor(color44)
        radar.setup_axis(ax=ax, facecolor='None')  # format axis as a radar
        rings_inner = radar.draw_circles(ax=ax, facecolor=color44, edgecolor=color46)
        radar_output = radar.draw_radar_compare(stat11, stat22, ax=ax,
                                                kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                                kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
        radar_poly, radar_poly2, vertices1, vertices2 = radar_output
        range_labels = radar.draw_range_labels(ax=ax, fontsize=15,color='white',
                                                font="STXihei",weight="bold")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=15,color='white',
                                            font="STXihei",weight="bold")

        ax.scatter(vertices1[:, 0], vertices1[:, 1],
                                    c="#00f2c1", edgecolors='#00f2c1', marker='o', s=30, zorder=2)
        ax.scatter(vertices2[:, 0], vertices2[:, 1],
                                c='#d80499', edgecolors='#d80499', marker='o', s=30, zorder=2)

        # add title
        fig.text(
        0.515, 0.934,  f"Winger Radar",
        path_effects=[path_effects.Stroke(linewidth=0.2, foreground="white"), path_effects.Normal()],
        ha="center", font = "Century Gothic",size =39,color="white",fontweight="bold"
        )



        # add credits
        notes = f'Stats compared only with\nplayers with minutes > {minutes} and age <= {age}'
        fig.text(
        0.895, 0.08, f"{notes}",
        font = "Century Gothic",size =13,color="white",
        ha="right"
        )

        # Define circle properties
        center = (0.38, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#00f2c1')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.40, 0.895, f"{Name}",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )

        # Define circle properties
        center = (0.6, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#d80499')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.62, 0.895, f"Average",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )
        plt.savefig("data/images/pizza.jpg",dpi =500, bbox_inches='tight')

        
        defend.reset_index(drop=True, inplace=True)  # Reset the index after filtering

        # select stats
        x =player_name
        player_df = defend.loc[defend['Player']==x]

        # Define the desired order of metrics
        metrics = req_columns

        # Calculate percentile ranks for each metric
        percentile_ranks = {}
        for metric in metrics:
            percentile_ranks[metric] = rankdata(defend[metric], method='average') / len(defend) * 99

        # Define colors and create a colormap
        colors = ['red', 'orange', 'yellow', 'green']
        percentile_ranges = [ 25, 50, 75, 100]
        cmap = ListedColormap(colors)


        # Create a bar graph with black background and facecolor
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
        ax.set_facecolor('black')


        ax.axvline(x=25, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=50, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=75, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=100, linestyle='--', color='white', linewidth=0.8)

        for i, metric in enumerate(metrics):
            metric_value = player_df[metric].values[0]
            percentile_rank = percentile_ranks[metric][player_df.index[0]]

            color_index = next(idx for idx, pct in enumerate(percentile_ranges) if pct >= percentile_rank)
            
            # Assign color based on the index
            color = cmap(color_index)

        # Assign color based on percentile rank
            bar = ax.barh(i, percentile_rank, height=0.3, alpha=0.7, color=color,edgecolor ="white",linewidth =0.8 ,zorder=1)
            ax.text(
                105, bar[0].get_y() + bar[0].get_height() / 2, f'{metric_value:.2f}', va='center', ha='left',
                font='STXihei', size=15, color='white'
            )

        # ax.axvline(50, linestyle='--', label='50th Percentile Rank',color='white')
        # ax.axvline(25, linestyle='--', label='25th Percentile Rank', color='white')
        # ax.axvline(75, linestyle='--', label='75th Percentile Rank', color='white')

        new_labels = req_columns


        # Set the new labels for the y-axis ticks
        ax.set_yticklabels(new_labels,font='Tw Cen MT', color='white') 

        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_yticks(range(len(metrics)))

        max_percentile_rank = max([max(percentile_ranks[metric]) for metric in metrics])
        ax.set_xlim(0, max_percentile_rank + 10)

        ax.set_xlabel('Percentile Ranks', color='white',fontname = "Tw Cen MT",size=20)
        # ax.set_ylabel('Metrics', color='white',fontname = "Tw Cen MT",size=20)
        plt.xticks(fontname = "Tw Cen MT",color="white",size=12)
        ax.set_xticks([0, 25, 50,75, 100])

        plt.yticks(fontname = "Tw Cen MT",color="white",size=15)


        ax.legend().remove()


        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.spines['left'].set_visible(False)  # Hide the left spine

        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')

        top_boxes = [
            Rectangle((i * 20, 20), 20, 5, color=colors[i], edgecolor='black', linewidth=1) for i in range(len(colors))
        ]

        texts =["Bottom 25%","From 25-50%","From 50-75%","Top 25%"]
        for i, box in enumerate(top_boxes):
            ax.add_patch(box)
            ax.text(
                i * 24 + 15, -0.8, f'{texts[i]}',font="Tw Cen MT", va='center', ha='center', color=colors[i], fontweight='bold', fontsize=25
            )

        plt.subplots_adjust(left=0.2)

        # # add credits
        # notes = f'All Stats per 90'
        # fig.text(
        # 0.27, 0.067, f"{notes}",
        # font = "Century Gothic",size =11,color="white",
        # ha="right"
        # )


        plt.savefig("data/images/bar.jpg",dpi =500, bbox_inches='tight')

        fig = plt.figure(figsize = (10,8), dpi = 300)

        fig.set_facecolor('black')
        plt.rcParams['hatch.linewidth'] = .02

        # one-liner
        player_df = defend.loc[defend['Player']==player_name]


        time = float(player_df.iloc[0,6])
        Name = str(player_df.iloc[0,0])
        Team = str(player_df.iloc[0,2])
        age =player_df.iloc[0,4]

        if len(player_name) > 20:
            fn = 14
        elif len(player_name) > 15:
            fn = 17
        else : 
            fn = 22

        im1 = plt.imread(r"C:\Users\lolen\OneDrive\Documents\Coding\Neurotactic Essentials\Images\Neuro.png")
        ax_image = add_image( im1, fig, left=1.02, bottom=0.982, width=0.13, height=0.13 )   # these values might differ when you are plotting




        # NAME

        #Heading
        str_text = f'''Player Data Report'''

        fig_text(
            x = 0.55, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="white",
            bbox=dict(boxstyle='round', facecolor='none', edgecolor='#4A9BD4', linewidth=2)
        )


        #Heading
        str_text = f'''--------------------------------------------------'''

        fig_text(
            x = 0.15, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )


        #Heading
        str_text = f'''-----------------------------------------------'''

        fig_text(
            x = 0.775, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )



        Attacking_Mid_score = z_scores_df[z_scores_df['Player']==x]["Winger_Score"].values[0]
        attack_df = z_scores_df.sort_values(by='Winger_Score', ascending=False).reset_index(drop=True)
        Attacking_Mid_rank = attack_df.index[attack_df['Player'] == x].values[0] +1 


        fig.text(
        x = 1.14, y = 0.945, s = f"Winger Rating : {Attacking_Mid_score:.2f} | Rank : {Attacking_Mid_rank}/{len(defend)}",
        ha="right", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )


        # add radar
        im1 = plt.imread(r"data/images/pizza.jpg")
        ax_image = add_image(
                im1, fig, left=0.691, bottom=0.4, width=0.53, height=0.53
            )   # these values might differ when you are plotting


        im1 = plt.imread(r"data/images/bar.jpg")
        ax_image = add_image(
                im1, fig, left=0.14 ,bottom=0.4, width=0.6, height=0.6
            )   # these values might differ when you are plotting

        fig_text(
            x = 0.154, y = 1.04, 
            s = f'<{player_name}-{Team}>',
            highlight_textprops=[{"color":"#FFD230"}],
            fontname ="Century Gothic",path_effects=[path_effects.Stroke(linewidth=0.4, foreground="#BD8B00"), path_effects.Normal()],
            va = 'bottom', ha = 'left',
            fontsize = fn,  weight = 'bold',color="white"
        )

        fig.text(
        x = 0.15, y = 1.024, s = f" Minutes Played : {time:.0f} | Age : {age:.0f} | Season : 23-24",
        ha="left", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )
        st.pyplot(fig)

elif sec_pos == 'Attacking Mid':
    defend["Succ Dribbles per 90"] = defend['Dribbles per 90'] *defend['Successful dribbles, %'] * 0.01
    defend["Acc Passes per 90"] = defend['Passes per 90'] *defend['Accurate passes, %'] * 0.01
    defend["Possesion Lost per 90"] = (defend['Dribbles per 90'] -defend["Succ Dribbles per 90"]) + (defend['Passes per 90'] - defend["Acc Passes per 90"])
    defend["Acc Passes to final3rd per 90"] = defend['Passes to final third per 90'] *defend['Accurate passes to final third, %'] * 0.01
    defend["Acc Passes PenArea per 90"] = defend['Passes to penalty area per 90'] *defend['Accurate passes to penalty area, %'] * 0.01
    defend["Acc Through Passes per 90"] = defend['Through passes per 90'] *defend['Accurate through passes, %'] * 0.01
    defend["Acc Crosses per 90"] = defend['Crosses per 90'] *defend['Accurate crosses, %'] * 0.01
    defend["Pass Prog Acc"] =  defend['Progressive passes per 90'] * defend['Accurate progressive passes, %'] * 0.01



    z_scores_df = defend.copy()  # Create a copy to preserve the original DataFrame
    numeric_columns = defend.select_dtypes(include='number').columns

    # Calculate Z-scores using the correct standard deviation for each column
    for col in numeric_columns:
        z_scores_df[col] = zscore(defend[col])

    # Min-Max scaling to scale Z-scores from 0 to 100
    z_scores_df = ((z_scores_df[numeric_columns] - z_scores_df[numeric_columns].min()) / 
                (z_scores_df[numeric_columns].max() - z_scores_df[numeric_columns].min())) * 100
    # Print the resulting DataFrame with Z-scores
    # Combine scaled values with non-numeric columns
    z_scores_df = pd.concat([defend[['Player','Team']], z_scores_df], axis=1)


        # 1. Play Making : xA,key passes,through balls.assists
    weights = [0.1,0.2,0.4,0.2,0.1]
    z_scores_df["Play_Making"] = (z_scores_df['Assists per 90'] * weights[0] +
                                z_scores_df['xA per 90'] * weights[1] +
                                z_scores_df['Key passes per 90'] * weights[2]+z_scores_df['Acc Through Passes per 90'] * weights[3]
                                )
    # 1.  Goal_Threat : 
    weights = [0.5,0.3,0.1,0.1]
    z_scores_df["Goal_Threat"] = (z_scores_df['Non-penalty goals per 90'] * weights[0] +
                                z_scores_df['xG per 90'] * weights[1] +
                                z_scores_df['Shots per 90'] * weights[2]+z_scores_df['Shots on target, %'] * weights[3])
    # 1.  Dribbling : 
    weights = [0.2,0.4,0.2,0.1,0.1]
    z_scores_df["Possession"] = (z_scores_df['Touches in box per 90'] * weights[0] +
                                z_scores_df['Succ Dribbles per 90'] * weights[1]+z_scores_df['Progressive runs per 90'] * weights[2] +
                                z_scores_df['Successful attacking actions per 90'] * weights[3]+z_scores_df['Received passes per 90'] * weights[4]  )

    # 1.  Passing : 
    weights = [0.3,0.3,0.2,0.2]

    z_scores_df["Passing"] = (z_scores_df['Acc Passes to final3rd per 90'] * weights[0] +
                                z_scores_df['Acc Crosses per 90'] * weights[1]+z_scores_df['Acc Passes PenArea per 90'] * weights[2] +
                            z_scores_df['Deep completions per 90'] * weights[3] )


    weights = [0.4,0.15,0.15,0.3]
    z_scores_df["Attacking_Mid"] = (z_scores_df['Play_Making'] * weights[0] +
                                z_scores_df['Goal_Threat'] * weights[1] +
                                z_scores_df['Possession'] * weights[2]+z_scores_df['Passing'] * weights[3])


    # Min-Max scaling to scale values to a ranking between 50 and 100
    min_value = z_scores_df["Attacking_Mid"] .min()
    max_value = z_scores_df["Attacking_Mid"] .max()

    z_scores_df["Attacking_Mid"]  = 30 + ((z_scores_df["Attacking_Mid"]  - min_value) / (max_value - min_value)) * 70


    ratingss = z_scores_df.Attacking_Mid

    defend['Attacking_Mid'] = ratingss

    mean_values = defend.select_dtypes(include='number').mean().to_frame().T

    # Set a name for the index to distinguish the mean row
    mean_values.index = ['mean']

    # Concatenate the mean values as a new row
    defend = pd.concat([defend, mean_values], ignore_index=True)

    defend = defend.fillna("Avg")
    last_index = len(defend) - 1
    defend.iloc[last_index, 3] = 5000

    # defend.drop_duplicates(subset=['Player', 'Team','Position'])

    defend_p = defend.copy()
    # Function to calculate percentiles
    def calculate_percentiles(series):
        return series.rank(pct=True) * 100

    # Select all numerical columns
    numerical_columns = defend_p.select_dtypes(include='number').columns

    # Apply the percentile calculation to all numerical columns
    defend_p[numerical_columns] = defend_p[numerical_columns].apply(calculate_percentiles)


    filter_prem = defend_p.copy()
    # prem = prem[prem["Age"] < 30]
    x =player_name
    player_fil = filter_prem.loc[filter_prem['Player']==x]



    # Select all numerical columns
    numerical_columns = player_fil.select_dtypes(include='number').columns

    # Set percentile filter
    # min_percentile = st.slider("Set minimum percentile cutoff", 0, 99, format=None, key=None)
    min_percentile = 70
    player_fil = player_fil[player_fil[numerical_columns] > min_percentile].dropna(axis=1)


    req_columns = player_fil.columns


    columns = req_columns
    selected_columns = []

    st.markdown("""
        <style>
        .slider-label {
            font-family: Century Gothic;
            font-size: 22px;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<span class='slider-label'>Select Columns to plot</span>", unsafe_allow_html=True)

    for col in columns:
        selected = st.checkbox(col, value=False, key=col)
        if selected:
            selected_columns.append(col)

    # st.write("Selected columns:", selected_columns)
    status1 = [ "No","Yes"]
    status = st.radio("Show Plot",horizontal=True,options = status1)

    if status =="Yes":

        req_columns = selected_columns
                # Radar
        color_1 = "#F98866"
        color_2 = "#408EC6"
        x =	player_name
        prem = defend.copy()
        player = prem.loc[prem['Player']==x]
        player2 = prem.loc[prem['Player']== "Avg"]

        Name = str(player.iloc[0,0])

        kik = prem.copy()

        stat1 = []
        stat2 = []
        for x in req_columns:
            stat1.extend(player[x])
            stat2.extend(player2[x])

        params = req_columns


        lower_is_better = []
        # minimum range value and maximum range value for parameters
        min_range= []
        max_range =[]
        for x in req_columns:
            min_range.append(kik.loc[:,x].min())
            max_range.append(kik.loc[:,x].max())          
        stat11 = [ round(x, 2) for x in stat1]        
        stat22 = [ round(x, 2) for x in stat2]  

        radar = Radar(params, min_range, max_range,
                    lower_is_better=lower_is_better,
                    # whether to round any of the labels to integers instead of decimal places
                    round_int=[False]*len(params),
                    num_rings=3,  # the number of concentric circles (excluding center circle)
                    # if the ring_width is more than the center_circle_radius then
                    # the center circle radius will be wider than the width of the concentric circles
                    ring_width=1, center_circle_radius=1)



        fig, ax = radar.setup_axis()
        color44 = "black"
        color45= "#000544"
        color46 ='#4FC0E8'
        fig.set_facecolor("black")
        ax.patch.set_facecolor(color44)
        radar.setup_axis(ax=ax, facecolor='None')  # format axis as a radar
        rings_inner = radar.draw_circles(ax=ax, facecolor=color44, edgecolor=color46)
        radar_output = radar.draw_radar_compare(stat11, stat22, ax=ax,
                                                kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                                kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
        radar_poly, radar_poly2, vertices1, vertices2 = radar_output
        range_labels = radar.draw_range_labels(ax=ax, fontsize=15,color='white',
                                                font="STXihei",weight="bold")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=15,color='white',
                                            font="STXihei",weight="bold")

        ax.scatter(vertices1[:, 0], vertices1[:, 1],
                                    c="#00f2c1", edgecolors='#00f2c1', marker='o', s=30, zorder=2)
        ax.scatter(vertices2[:, 0], vertices2[:, 1],
                                c='#d80499', edgecolors='#d80499', marker='o', s=30, zorder=2)

        # add title
                # add title
        fig.text(
        0.515, 0.934, f"Attacking Midfielder Radar",
        path_effects=[path_effects.Stroke(linewidth=0.2, foreground="white"), path_effects.Normal()],
        ha="center", font = "Century Gothic",size =39,color="white",fontweight="bold"
        )



        # add credits
        notes = f'Stats compared only with\nplayers with minutes > {minutes} and age>= {age}'
        fig.text(
        0.895, 0.08, f"{notes}",
        font = "Century Gothic",size =13,color="white",
        ha="right"
        )

        # Define circle properties
        center = (0.38, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#00f2c1')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.40, 0.895, f"{Name}",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )

        # Define circle properties
        center = (0.6, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#d80499')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.62, 0.895, f"Average",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )
        plt.savefig("data/images/pizza.jpg",dpi =500, bbox_inches='tight')
        # st.pyplot(fig)
        
        defend.reset_index(drop=True, inplace=True)  # Reset the index after filtering

        # select stats
        x =player_name
        player_df = defend.loc[defend['Player']==x]

        # Define the desired order of metrics
        metrics = req_columns

        # Calculate percentile ranks for each metric
        percentile_ranks = {}
        for metric in metrics:
            percentile_ranks[metric] = rankdata(defend[metric], method='average') / len(defend) * 99

        # Define colors and create a colormap
        colors = ['red', 'orange', 'yellow', 'green']
        percentile_ranges = [ 25, 50, 75, 100]
        cmap = ListedColormap(colors)


        # Create a bar graph with black background and facecolor
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
        ax.set_facecolor('black')


        ax.axvline(x=25, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=50, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=75, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=100, linestyle='--', color='white', linewidth=0.8)

        for i, metric in enumerate(metrics):
            metric_value = player_df[metric].values[0]
            percentile_rank = percentile_ranks[metric][player_df.index[0]]

            color_index = next(idx for idx, pct in enumerate(percentile_ranges) if pct >= percentile_rank)
            
            # Assign color based on the index
            color = cmap(color_index)

        # Assign color based on percentile rank
            bar = ax.barh(i, percentile_rank, height=0.3, alpha=0.7, color=color,edgecolor ="white",linewidth =0.8 ,zorder=1)
            ax.text(
                105, bar[0].get_y() + bar[0].get_height() / 2, f'{metric_value:.2f}', va='center', ha='left',
                font='STXihei', size=15, color='white'
            )

        # ax.axvline(50, linestyle='--', label='50th Percentile Rank',color='white')
        # ax.axvline(25, linestyle='--', label='25th Percentile Rank', color='white')
        # ax.axvline(75, linestyle='--', label='75th Percentile Rank', color='white')

        new_labels = req_columns


        # Set the new labels for the y-axis ticks
        ax.set_yticklabels(new_labels,font='Tw Cen MT', color='white') 

        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_yticks(range(len(metrics)))

        max_percentile_rank = max([max(percentile_ranks[metric]) for metric in metrics])
        ax.set_xlim(0, max_percentile_rank + 10)

        ax.set_xlabel('Percentile Ranks', color='white',fontname = "Tw Cen MT",size=20)
        # ax.set_ylabel('Metrics', color='white',fontname = "Tw Cen MT",size=20)
        plt.xticks(fontname = "Tw Cen MT",color="white",size=12)
        ax.set_xticks([0, 25, 50,75, 100])

        plt.yticks(fontname = "Tw Cen MT",color="white",size=15)


        ax.legend().remove()


        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.spines['left'].set_visible(False)  # Hide the left spine

        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')

        top_boxes = [
            Rectangle((i * 20, 20), 20, 5, color=colors[i], edgecolor='black', linewidth=1) for i in range(len(colors))
        ]

        texts =["Bottom 25%","From 25-50%","From 50-75%","Top 25%"]
        for i, box in enumerate(top_boxes):
            ax.add_patch(box)
            ax.text(
                i * 24 + 15, -0.8, f'{texts[i]}',font="Tw Cen MT", va='center', ha='center', color=colors[i], fontweight='bold', fontsize=25
            )

        plt.subplots_adjust(left=0.2)

        # # add credits
        # notes = f'All Stats per 90'
        # fig.text(
        # 0.27, 0.067, f"{notes}",
        # font = "Century Gothic",size =11,color="white",
        # ha="right"
        # )


        plt.savefig("data/images/bar.jpg",dpi =500, bbox_inches='tight')

        fig = plt.figure(figsize = (10,8), dpi = 300)

        fig.set_facecolor('black')
        plt.rcParams['hatch.linewidth'] = .02

        # one-liner
        player_df = defend.loc[defend['Player']==player_name]


        time = float(player_df.iloc[0,5])
        Name = str(player_df.iloc[0,0])
        Team = str(player_df.iloc[0,1])
        age =player_df.iloc[0,3]

        if len(player_name) > 20:
            fn = 14
        elif len(player_name) > 15:
            fn = 17
        else : 
            fn = 22

        im1 = plt.imread(r"C:\Users\lolen\OneDrive\Documents\Coding\Neurotactic Essentials\Images\Neuro.png")
        ax_image = add_image( im1, fig, left=1.02, bottom=0.982, width=0.13, height=0.13 )   # these values might differ when you are plotting




        # NAME

        #Heading
        str_text = f'''Player Data Report'''

        fig_text(
            x = 0.55, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="white",
            bbox=dict(boxstyle='round', facecolor='none', edgecolor='#4A9BD4', linewidth=2)
        )


        #Heading
        str_text = f'''--------------------------------------------------'''

        fig_text(
            x = 0.15, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )


        #Heading
        str_text = f'''-----------------------------------------------'''

        fig_text(
            x = 0.775, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )



        Attacking_Mid_score = z_scores_df[z_scores_df['Player']==x]["Attacking_Mid"].values[0]
        attack_df = z_scores_df.sort_values(by='Attacking_Mid', ascending=False).reset_index(drop=True)
        Attacking_Mid_rank = attack_df.index[attack_df['Player'] == x].values[0] +1 


        fig.text(
        x = 1.14, y = 0.945, s = f"Attacking Mid Rating : {Attacking_Mid_score:.2f} | Rank : {Attacking_Mid_rank} / {len(defend)}",
        ha="right", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )

        # add radar
        im1 = plt.imread(r"data/images/pizza.jpg")
        ax_image = add_image(
                im1, fig, left=0.691, bottom=0.4, width=0.53, height=0.53
            )   # these values might differ when you are plotting


        im1 = plt.imread(r"data/images/bar.jpg")
        ax_image = add_image(
                im1, fig, left=0.14 ,bottom=0.4, width=0.6, height=0.6
            )   # these values might differ when you are plotting

        fig_text(
            x = 0.154, y = 1.04, 
            s = f'<{player_name}-{Team}>',
            highlight_textprops=[{"color":"#FFD230"}],
            fontname ="Century Gothic",path_effects=[path_effects.Stroke(linewidth=0.4, foreground="#BD8B00"), path_effects.Normal()],
            va = 'bottom', ha = 'left',
            fontsize = fn,  weight = 'bold',color="white"
        )

        fig.text(
        x = 0.15, y = 1.024, s = f" Minutes Played : {time:.0f} | Age : {age:.0f} | Season : 23-24",
        ha="left", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )
        st.pyplot(fig)
elif sec_pos == 'Central Mid':

    defend["Succ Dribbles per 90"] = defend['Dribbles per 90'] *defend['Successful dribbles, %'] * 0.01
    defend["Acc Passes per 90"] = defend['Passes per 90'] *defend['Accurate passes, %'] * 0.01
    defend["Possesion Lost per 90"] = (defend['Dribbles per 90'] -defend["Succ Dribbles per 90"]) + (defend['Passes per 90'] - defend["Acc Passes per 90"])
    defend["Acc Passes to final3rd per 90"] = defend['Passes to final third per 90'] *defend['Accurate passes to final third, %'] * 0.01
    defend["Acc Forward Passes per 90"] = defend['Forward passes per 90'] *defend['Accurate forward passes, %'] * 0.01
    defend["Acc Long Passes per 90"] = defend['Long passes per 90'] *defend['Accurate long passes, %'] * 0.01
    defend["Acc Passes PenArea per 90"] = defend['Passes to penalty area per 90'] *defend['Accurate passes to penalty area, %'] * 0.01
    defend["Acc Through Passes per 90"] = defend['Through passes per 90'] *defend['Accurate through passes, %'] * 0.01
    defend["Acc Crosses per 90"] = defend['Crosses per 90'] *defend['Accurate crosses, %'] * 0.01
    defend["Pass Prog Acc per 90"] =  defend['Progressive passes per 90'] * defend['Accurate progressive passes, %'] * 0.01
    defend["Def Duels Won"] = defend['Defensive duels per 90'] *defend['Defensive duels won, %'] * 0.01
    defend["Aerial Duels Won"] =   defend['Aerial duels per 90'] * defend['Aerial duels won, %'] * 0.01


    z_scores_df = defend.copy()  # Create a copy to preserve the original DataFrame
    numeric_columns = defend.select_dtypes(include='number').columns

    # Calculate Z-scores using the correct standard deviation for each column
    for col in numeric_columns:
        z_scores_df[col] = zscore(defend[col])

    # Min-Max scaling to scale Z-scores from 0 to 100
    z_scores_df = ((z_scores_df[numeric_columns] - z_scores_df[numeric_columns].min()) / 
                (z_scores_df[numeric_columns].max() - z_scores_df[numeric_columns].min())) * 100
    # Print the resulting DataFrame with Z-scores
    # Combine scaled values with non-numeric columns
    z_scores_df = pd.concat([defend[['Player','Team']], z_scores_df], axis=1)


        
    # 1. Play Making : xA,key passes,through balls.assists
    weights = [0.1,0.2,0.45,0.25]
    z_scores_df["Play_Making"] = (z_scores_df['Assists per 90'] * weights[0] +
                                z_scores_df['xA per 90'] * weights[1] +
                                z_scores_df['Key passes per 90'] * weights[2]+z_scores_df['Acc Through Passes per 90'] * weights[3]
                                )
    # 1.  Goal_Threat : 
    weights = [0.4,0.3,0.2,0.1]
    z_scores_df["Goal_Threat"] = (z_scores_df['Non-penalty goals per 90'] * weights[0] +
                                z_scores_df['xG per 90'] * weights[1] +
                                z_scores_df['Shots per 90'] * weights[2]+z_scores_df['Shots on target, %'] * weights[3])
    # 1.  Dribbling : 
    weights = [0.1,0.4,0.2,0.3]
    z_scores_df["Possession"] = (z_scores_df['Touches in box per 90'] * weights[0] +
                                z_scores_df['Succ Dribbles per 90'] * weights[1]+z_scores_df['Progressive runs per 90'] * weights[2] +
                                z_scores_df['Successful attacking actions per 90'] * weights[3] )

    # 1.  Passing : 
    weights = [0.2,0.1,0.3,0.2,0.2]
    z_scores_df["Passing"] = (z_scores_df['Acc Passes to final3rd per 90'] * weights[0] +
                                z_scores_df['Acc Long Passes per 90'] * weights[1]+z_scores_df['Acc Passes per 90'] * weights[2] +
                            z_scores_df["Acc Forward Passes per 90"] * weights[3]+    z_scores_df["Pass Prog Acc per 90"] * weights[3])


    # 1.  Game_Involvement : 
    weights = [0.4,0.2,0.2,0.2]
    z_scores_df["Defending"] = (z_scores_df['Def Duels Won'] * weights[0] +
                                z_scores_df['Sliding tackles per 90'] * weights[1] +
                                z_scores_df['Interceptions per 90'] * weights[2]+
                                z_scores_df['Aerial Duels Won'] * weights[3])



    weights = [0.3,0.1,0.1,0.4,0.1]
    z_scores_df["Central_Mid"] = (z_scores_df['Play_Making'] * weights[0] +
                                z_scores_df['Goal_Threat'] * weights[1] +
                                z_scores_df['Possession'] * weights[2]+z_scores_df['Passing'] * weights[3] +
                                z_scores_df['Defending'] * weights[4])


    # Min-Max scaling to scale values to a ranking between 50 and 100
    min_value = z_scores_df["Central_Mid"] .min()
    max_value = z_scores_df["Central_Mid"] .max()

    z_scores_df["Central_Mid"]  = 30 + ((z_scores_df["Central_Mid"]  - min_value) / (max_value - min_value)) * 70


    ratingss = z_scores_df.Central_Mid

    defend['Central_Mid'] = ratingss

    mean_values = defend.select_dtypes(include='number').mean().to_frame().T

    # Set a name for the index to distinguish the mean row
    mean_values.index = ['mean']

    # Concatenate the mean values as a new row
    defend = pd.concat([defend, mean_values], ignore_index=True)

    defend = defend.fillna("Avg")
    last_index = len(defend) - 1
    defend.iloc[last_index, 3] = 5000

    # defend.drop_duplicates(subset=['Player', 'Team','Position'])

    defend_p = defend.copy()
    # Function to calculate percentiles
    def calculate_percentiles(series):
        return series.rank(pct=True) * 100

    # Select all numerical columns
    numerical_columns = defend_p.select_dtypes(include='number').columns

    # Apply the percentile calculation to all numerical columns
    defend_p[numerical_columns] = defend_p[numerical_columns].apply(calculate_percentiles)


    filter_prem = defend_p.copy()
    # prem = prem[prem["Age"] < 30]
    x =player_name
    player_fil = filter_prem.loc[filter_prem['Player']==x]



    # Select all numerical columns
    numerical_columns = player_fil.select_dtypes(include='number').columns

    # Set percentile filter
    # min_percentile = st.slider("Set minimum percentile cutoff", 0, 99, format=None, key=None)
    min_percentile = 70
    player_fil = player_fil[player_fil[numerical_columns] > min_percentile].dropna(axis=1)


    req_columns = player_fil.columns


    columns = req_columns
    selected_columns = []

    st.markdown("""
        <style>
        .slider-label {
            font-family: Century Gothic;
            font-size: 22px;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<span class='slider-label'>Select Columns to plot</span>", unsafe_allow_html=True)

    for col in columns:
        selected = st.checkbox(col, value=False, key=col)
        if selected:
            selected_columns.append(col)

    # st.write("Selected columns:", selected_columns)
    status1 = [ "No","Yes"]
    status = st.radio("Show Plot",horizontal=True,options = status1)

    if status =="Yes":

        req_columns = selected_columns
                # Radar
        color_1 = "#F98866"
        color_2 = "#408EC6"
        x =	player_name
        prem = defend.copy()
        player = prem.loc[prem['Player']==x]
        player2 = prem.loc[prem['Player']== "Avg"]

        Name = str(player.iloc[0,0])

        kik = prem.copy()

        stat1 = []
        stat2 = []
        for x in req_columns:
            stat1.extend(player[x])
            stat2.extend(player2[x])

        params = req_columns


        lower_is_better = []
        # minimum range value and maximum range value for parameters
        min_range= []
        max_range =[]
        for x in req_columns:
            min_range.append(kik.loc[:,x].min())
            max_range.append(kik.loc[:,x].max())          
        stat11 = [ round(x, 2) for x in stat1]        
        stat22 = [ round(x, 2) for x in stat2]  

        radar = Radar(params, min_range, max_range,
                    lower_is_better=lower_is_better,
                    # whether to round any of the labels to integers instead of decimal places
                    round_int=[False]*len(params),
                    num_rings=3,  # the number of concentric circles (excluding center circle)
                    # if the ring_width is more than the center_circle_radius then
                    # the center circle radius will be wider than the width of the concentric circles
                    ring_width=1, center_circle_radius=1)



        fig, ax = radar.setup_axis()
        color44 = "black"
        color45= "#000544"
        color46 ='#4FC0E8'
        fig.set_facecolor("black")
        ax.patch.set_facecolor(color44)
        radar.setup_axis(ax=ax, facecolor='None')  # format axis as a radar
        rings_inner = radar.draw_circles(ax=ax, facecolor=color44, edgecolor=color46)
        radar_output = radar.draw_radar_compare(stat11, stat22, ax=ax,
                                                kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                                kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
        radar_poly, radar_poly2, vertices1, vertices2 = radar_output
        range_labels = radar.draw_range_labels(ax=ax, fontsize=15,color='white',
                                                font="STXihei",weight="bold")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=15,color='white',
                                            font="STXihei",weight="bold")

        ax.scatter(vertices1[:, 0], vertices1[:, 1],
                                    c="#00f2c1", edgecolors='#00f2c1', marker='o', s=30, zorder=2)
        ax.scatter(vertices2[:, 0], vertices2[:, 1],
                                c='#d80499', edgecolors='#d80499', marker='o', s=30, zorder=2)

        # add title
                # add title
        fig.text(
        0.515, 0.934, f"Central Midfielder Radar",
        path_effects=[path_effects.Stroke(linewidth=0.2, foreground="white"), path_effects.Normal()],
        ha="center", font = "Century Gothic",size =39,color="white",fontweight="bold"
        )



        # add credits
        notes = f'Stats compared only with\nplayers with minutes > {minutes} and age>= {age}'
        fig.text(
        0.895, 0.08, f"{notes}",
        font = "Century Gothic",size =13,color="white",
        ha="right"
        )

        # Define circle properties
        center = (0.38, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#00f2c1')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.40, 0.895, f"{Name}",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )

        # Define circle properties
        center = (0.6, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#d80499')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.62, 0.895, f"Average",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )
        plt.savefig("data/images/pizza.jpg",dpi =500, bbox_inches='tight')
        # st.pyplot(fig)
        
        defend.reset_index(drop=True, inplace=True)  # Reset the index after filtering

        # select stats
        x =player_name
        player_df = defend.loc[defend['Player']==x]

        # Define the desired order of metrics
        metrics = req_columns

        # Calculate percentile ranks for each metric
        percentile_ranks = {}
        for metric in metrics:
            percentile_ranks[metric] = rankdata(defend[metric], method='average') / len(defend) * 99

        # Define colors and create a colormap
        colors = ['red', 'orange', 'yellow', 'green']
        percentile_ranges = [ 25, 50, 75, 100]
        cmap = ListedColormap(colors)


        # Create a bar graph with black background and facecolor
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
        ax.set_facecolor('black')


        ax.axvline(x=25, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=50, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=75, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=100, linestyle='--', color='white', linewidth=0.8)

        for i, metric in enumerate(metrics):
            metric_value = player_df[metric].values[0]
            percentile_rank = percentile_ranks[metric][player_df.index[0]]

            color_index = next(idx for idx, pct in enumerate(percentile_ranges) if pct >= percentile_rank)
            
            # Assign color based on the index
            color = cmap(color_index)

        # Assign color based on percentile rank
            bar = ax.barh(i, percentile_rank, height=0.3, alpha=0.7, color=color,edgecolor ="white",linewidth =0.8 ,zorder=1)
            ax.text(
                105, bar[0].get_y() + bar[0].get_height() / 2, f'{metric_value:.2f}', va='center', ha='left',
                font='STXihei', size=15, color='white'
            )

        # ax.axvline(50, linestyle='--', label='50th Percentile Rank',color='white')
        # ax.axvline(25, linestyle='--', label='25th Percentile Rank', color='white')
        # ax.axvline(75, linestyle='--', label='75th Percentile Rank', color='white')

        new_labels = req_columns


        # Set the new labels for the y-axis ticks
        ax.set_yticklabels(new_labels,font='Tw Cen MT', color='white') 

        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_yticks(range(len(metrics)))

        max_percentile_rank = max([max(percentile_ranks[metric]) for metric in metrics])
        ax.set_xlim(0, max_percentile_rank + 10)

        ax.set_xlabel('Percentile Ranks', color='white',fontname = "Tw Cen MT",size=20)
        # ax.set_ylabel('Metrics', color='white',fontname = "Tw Cen MT",size=20)
        plt.xticks(fontname = "Tw Cen MT",color="white",size=12)
        ax.set_xticks([0, 25, 50,75, 100])

        plt.yticks(fontname = "Tw Cen MT",color="white",size=15)


        ax.legend().remove()


        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.spines['left'].set_visible(False)  # Hide the left spine

        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')

        top_boxes = [
            Rectangle((i * 20, 20), 20, 5, color=colors[i], edgecolor='black', linewidth=1) for i in range(len(colors))
        ]

        texts =["Bottom 25%","From 25-50%","From 50-75%","Top 25%"]
        for i, box in enumerate(top_boxes):
            ax.add_patch(box)
            ax.text(
                i * 24 + 15, -0.8, f'{texts[i]}',font="Tw Cen MT", va='center', ha='center', color=colors[i], fontweight='bold', fontsize=25
            )

        plt.subplots_adjust(left=0.2)

        # # add credits
        # notes = f'All Stats per 90'
        # fig.text(
        # 0.27, 0.067, f"{notes}",
        # font = "Century Gothic",size =11,color="white",
        # ha="right"
        # )


        plt.savefig("data/images/bar.jpg",dpi =500, bbox_inches='tight')

        fig = plt.figure(figsize = (10,8), dpi = 300)

        fig.set_facecolor('black')
        plt.rcParams['hatch.linewidth'] = .02

        # one-liner
        player_df = defend.loc[defend['Player']==player_name]


        time = float(player_df.iloc[0,5])
        Name = str(player_df.iloc[0,0])
        Team = str(player_df.iloc[0,1])
        age =player_df.iloc[0,3]

        if len(player_name) > 20:
            fn = 14
        elif len(player_name) > 15:
            fn = 17
        else : 
            fn = 22

        im1 = plt.imread(r"C:\Users\lolen\OneDrive\Documents\Coding\Neurotactic Essentials\Images\Neuro.png")
        ax_image = add_image( im1, fig, left=1.02, bottom=0.982, width=0.13, height=0.13 )   # these values might differ when you are plotting




        # NAME

        #Heading
        str_text = f'''Player Data Report'''

        fig_text(
            x = 0.55, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="white",
            bbox=dict(boxstyle='round', facecolor='none', edgecolor='#4A9BD4', linewidth=2)
        )


        #Heading
        str_text = f'''--------------------------------------------------'''

        fig_text(
            x = 0.15, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )


        #Heading
        str_text = f'''-----------------------------------------------'''

        fig_text(
            x = 0.775, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )



        Attacking_Mid_score = z_scores_df[z_scores_df['Player']==x]["Central_Mid"].values[0]
        attack_df = z_scores_df.sort_values(by='Central_Mid', ascending=False).reset_index(drop=True)
        Attacking_Mid_rank = attack_df.index[attack_df['Player'] == x].values[0] +1 


        fig.text(
        x = 1.14, y = 0.945, s = f"Central Midfielder Rating : {Attacking_Mid_score:.2f} | Rank : {Attacking_Mid_rank} / {len(defend)}",
        ha="right", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )

        # add radar
        im1 = plt.imread(r"data/images/pizza.jpg")
        ax_image = add_image(
                im1, fig, left=0.691, bottom=0.4, width=0.53, height=0.53
            )   # these values might differ when you are plotting


        im1 = plt.imread(r"data/images/bar.jpg")
        ax_image = add_image(
                im1, fig, left=0.14 ,bottom=0.4, width=0.6, height=0.6
            )   # these values might differ when you are plotting

        fig_text(
            x = 0.154, y = 1.04, 
            s = f'<{player_name}-{Team}>',
            highlight_textprops=[{"color":"#FFD230"}],
            fontname ="Century Gothic",path_effects=[path_effects.Stroke(linewidth=0.4, foreground="#BD8B00"), path_effects.Normal()],
            va = 'bottom', ha = 'left',
            fontsize = fn,  weight = 'bold',color="white"
        )

        fig.text(
        x = 0.15, y = 1.024, s = f" Minutes Played : {time:.0f} | Age : {age:.0f} | Season : 23-24",
        ha="left", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )
        st.pyplot(fig)

elif sec_pos == 'Defensive Midfielder':
    
    defend["Succ Dribbles per 90"] = defend['Dribbles per 90'] *defend['Successful dribbles, %'] * 0.01
    defend["Acc Passes per 90"] = defend['Passes per 90'] *defend['Accurate passes, %'] * 0.01
    defend["Possesion Lost per 90"] = (defend['Dribbles per 90'] -defend["Succ Dribbles per 90"]) + (defend['Passes per 90'] - defend["Acc Passes per 90"])
    defend["Acc Passes to final3rd per 90"] = defend['Passes to final third per 90'] *defend['Accurate passes to final third, %'] * 0.01
    defend["Acc Forward Passes per 90"] = defend['Forward passes per 90'] *defend['Accurate forward passes, %'] * 0.01
    defend["Acc Long Passes per 90"] = defend['Long passes per 90'] *defend['Accurate long passes, %'] * 0.01
    defend["Acc Passes PenArea per 90"] = defend['Passes to penalty area per 90'] *defend['Accurate passes to penalty area, %'] * 0.01
    defend["Acc Through Passes per 90"] = defend['Through passes per 90'] *defend['Accurate through passes, %'] * 0.01
    defend["Acc Crosses per 90"] = defend['Crosses per 90'] *defend['Accurate crosses, %'] * 0.01
    defend["Pass Prog Acc per 90"] =  defend['Progressive passes per 90'] * defend['Accurate progressive passes, %'] * 0.01
    defend["Def Duels Won"] = defend['Defensive duels per 90'] *defend['Defensive duels won, %'] * 0.01
    defend["Aerial Duels Won"] =   defend['Aerial duels per 90'] * defend['Aerial duels won, %'] * 0.01

    z_scores_df = defend.copy()  # Create a copy to preserve the original DataFrame
    numeric_columns = defend.select_dtypes(include='number').columns

    # Calculate Z-scores using the correct standard deviation for each column
    for col in numeric_columns:
        z_scores_df[col] = zscore(defend[col])

    # Min-Max scaling to scale Z-scores from 0 to 100
    z_scores_df = ((z_scores_df[numeric_columns] - z_scores_df[numeric_columns].min()) / 
                (z_scores_df[numeric_columns].max() - z_scores_df[numeric_columns].min())) * 100
    # Print the resulting DataFrame with Z-scores
    # Combine scaled values with non-numeric columns
    z_scores_df = pd.concat([defend[['Player','Team']], z_scores_df], axis=1)


        
    
    # 1. Play Making : xA,key passes,through balls.assists
    weights = [0.1,0.2,0.3,0.2,0.2]
    z_scores_df["Play_Making"] = (z_scores_df['Assists per 90'] * weights[0] +
                                z_scores_df['xA per 90'] * weights[1] +
                                z_scores_df['Key passes per 90'] * weights[2]+z_scores_df['Acc Through Passes per 90'] * weights[3]+
                                z_scores_df['Acc Passes PenArea per 90'] * weights[4]
                                )
    # 1.  Goal_Threat : 
    weights = [0.5,0.3,0.1,0.1]
    z_scores_df["Goal_Threat"] = (z_scores_df['Non-penalty goals per 90'] * weights[0] +
                                z_scores_df['xG per 90'] * weights[1] +
                                z_scores_df['Shots per 90'] * weights[2]+z_scores_df['Shots on target, %'] * weights[3])
    # 1.  Dribbling : 
    weights = [0.2,0.4,0.2,0.2]
    z_scores_df["Possession"] = (z_scores_df['Touches in box per 90'] * weights[0] +
                                z_scores_df['Succ Dribbles per 90'] * weights[1]+z_scores_df['Progressive runs per 90'] * weights[2] +
                                z_scores_df['Successful attacking actions per 90'] * weights[3])

    # 1.  Passing : 
    weights = [0.2,0.1,0.3,0.2,0.2]

    z_scores_df["Passing"] = (z_scores_df['Acc Passes to final3rd per 90'] * weights[0] +
                                z_scores_df['Acc Long Passes per 90'] * weights[1]+z_scores_df['Acc Passes per 90'] * weights[2] +
                            z_scores_df["Acc Forward Passes per 90"] * weights[3]+    z_scores_df["Pass Prog Acc per 90"] * weights[3])


    # 1.  Game_Involvement : 
    weights = [0.4,0.2,0.2,0.2]
    z_scores_df["Defending"] = (z_scores_df['Def Duels Won'] * weights[0] +
                                z_scores_df['Sliding tackles per 90'] * weights[1] +
                                z_scores_df['Interceptions per 90'] * weights[2]+
                                z_scores_df['Aerial Duels Won'] * weights[3])


    weights = [0.15,0.2,0.3,0.35]
    z_scores_df["Defensive_Mid"] = (z_scores_df['Play_Making'] * weights[0] +
                                +z_scores_df['Possession'] * weights[1]
                                +z_scores_df['Passing'] * weights[2]
                                +z_scores_df['Defending'] * weights[3])



    # Min-Max scaling to scale values to a ranking between 50 and 100
    min_value = z_scores_df["Defensive_Mid"] .min()
    max_value = z_scores_df["Defensive_Mid"] .max()

    z_scores_df["Defensive_Mid"]  = 30 + ((z_scores_df["Defensive_Mid"]  - min_value) / (max_value - min_value)) * 70


    ratingss = z_scores_df.Defensive_Mid

    defend['Defensive_Mid'] = ratingss


    mean_values = defend.select_dtypes(include='number').mean().to_frame().T

    # Set a name for the index to distinguish the mean row
    mean_values.index = ['mean']

    # Concatenate the mean values as a new row
    defend = pd.concat([defend, mean_values], ignore_index=True)

    defend = defend.fillna("Avg")
    last_index = len(defend) - 1
    defend.iloc[last_index, 3] = 5000

    # defend.drop_duplicates(subset=['Player', 'Team','Position'])

    defend_p = defend.copy()
    # Function to calculate percentiles
    def calculate_percentiles(series):
        return series.rank(pct=True) * 100

    # Select all numerical columns
    numerical_columns = defend_p.select_dtypes(include='number').columns

    # Apply the percentile calculation to all numerical columns
    defend_p[numerical_columns] = defend_p[numerical_columns].apply(calculate_percentiles)


    filter_prem = defend_p.copy()
    # prem = prem[prem["Age"] < 30]
    x =player_name
    player_fil = filter_prem.loc[filter_prem['Player']==x]



    # Select all numerical columns
    numerical_columns = player_fil.select_dtypes(include='number').columns

    # Set percentile filter
    # min_percentile = st.slider("Set minimum percentile cutoff", 0, 99, format=None, key=None)
    min_percentile = 70
    player_fil = player_fil[player_fil[numerical_columns] > min_percentile].dropna(axis=1)


    req_columns = player_fil.columns


    columns = req_columns
    selected_columns = []

    st.markdown("""
        <style>
        .slider-label {
            font-family: Century Gothic;
            font-size: 22px;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<span class='slider-label'>Select Columns to plot</span>", unsafe_allow_html=True)

    for col in columns:
        selected = st.checkbox(col, value=False, key=col)
        if selected:
            selected_columns.append(col)

    # st.write("Selected columns:", selected_columns)
    status1 = [ "No","Yes"]
    status = st.radio("Show Plot",horizontal=True,options = status1)

    if status =="Yes":

        req_columns = selected_columns
                # Radar
        color_1 = "#F98866"
        color_2 = "#408EC6"
        x =	player_name
        prem = defend.copy()
        player = prem.loc[prem['Player']==x]
        player2 = prem.loc[prem['Player']== "Avg"]

        Name = str(player.iloc[0,0])

        kik = prem.copy()

        stat1 = []
        stat2 = []
        for x in req_columns:
            stat1.extend(player[x])
            stat2.extend(player2[x])

        params = req_columns


        lower_is_better = []
        # minimum range value and maximum range value for parameters
        min_range= []
        max_range =[]
        for x in req_columns:
            min_range.append(kik.loc[:,x].min())
            max_range.append(kik.loc[:,x].max())          
        stat11 = [ round(x, 2) for x in stat1]        
        stat22 = [ round(x, 2) for x in stat2]  

        radar = Radar(params, min_range, max_range,
                    lower_is_better=lower_is_better,
                    # whether to round any of the labels to integers instead of decimal places
                    round_int=[False]*len(params),
                    num_rings=3,  # the number of concentric circles (excluding center circle)
                    # if the ring_width is more than the center_circle_radius then
                    # the center circle radius will be wider than the width of the concentric circles
                    ring_width=1, center_circle_radius=1)



        fig, ax = radar.setup_axis()
        color44 = "black"
        color45= "#000544"
        color46 ='#4FC0E8'
        fig.set_facecolor("black")
        ax.patch.set_facecolor(color44)
        radar.setup_axis(ax=ax, facecolor='None')  # format axis as a radar
        rings_inner = radar.draw_circles(ax=ax, facecolor=color44, edgecolor=color46)
        radar_output = radar.draw_radar_compare(stat11, stat22, ax=ax,
                                                kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                                kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
        radar_poly, radar_poly2, vertices1, vertices2 = radar_output
        range_labels = radar.draw_range_labels(ax=ax, fontsize=15,color='white',
                                                font="STXihei",weight="bold")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=15,color='white',
                                            font="STXihei",weight="bold")

        ax.scatter(vertices1[:, 0], vertices1[:, 1],
                                    c="#00f2c1", edgecolors='#00f2c1', marker='o', s=30, zorder=2)
        ax.scatter(vertices2[:, 0], vertices2[:, 1],
                                c='#d80499', edgecolors='#d80499', marker='o', s=30, zorder=2)

        # add title
                # add title
        fig.text(
        0.515, 0.934, f"Defensive Midfielder Radar",
        path_effects=[path_effects.Stroke(linewidth=0.2, foreground="white"), path_effects.Normal()],
        ha="center", font = "Century Gothic",size =39,color="white",fontweight="bold"
        )



        # add credits
        notes = f'Stats compared only with\nplayers with minutes > {minutes} and age>= {age}'
        fig.text(
        0.895, 0.08, f"{notes}",
        font = "Century Gothic",size =13,color="white",
        ha="right"
        )

        # Define circle properties
        center = (0.38, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#00f2c1')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.40, 0.895, f"{Name}",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )

        # Define circle properties
        center = (0.6, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#d80499')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.62, 0.895, f"Average",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )
        plt.savefig("data/images/pizza.jpg",dpi =500, bbox_inches='tight')
        # st.pyplot(fig)
        
        defend.reset_index(drop=True, inplace=True)  # Reset the index after filtering

        # select stats
        x =player_name
        player_df = defend.loc[defend['Player']==x]

        # Define the desired order of metrics
        metrics = req_columns

        # Calculate percentile ranks for each metric
        percentile_ranks = {}
        for metric in metrics:
            percentile_ranks[metric] = rankdata(defend[metric], method='average') / len(defend) * 99

        # Define colors and create a colormap
        colors = ['red', 'orange', 'yellow', 'green']
        percentile_ranges = [ 25, 50, 75, 100]
        cmap = ListedColormap(colors)


        # Create a bar graph with black background and facecolor
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
        ax.set_facecolor('black')


        ax.axvline(x=25, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=50, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=75, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=100, linestyle='--', color='white', linewidth=0.8)

        for i, metric in enumerate(metrics):
            metric_value = player_df[metric].values[0]
            percentile_rank = percentile_ranks[metric][player_df.index[0]]

            color_index = next(idx for idx, pct in enumerate(percentile_ranges) if pct >= percentile_rank)
            
            # Assign color based on the index
            color = cmap(color_index)

        # Assign color based on percentile rank
            bar = ax.barh(i, percentile_rank, height=0.3, alpha=0.7, color=color,edgecolor ="white",linewidth =0.8 ,zorder=1)
            ax.text(
                105, bar[0].get_y() + bar[0].get_height() / 2, f'{metric_value:.2f}', va='center', ha='left',
                font='STXihei', size=15, color='white'
            )

        # ax.axvline(50, linestyle='--', label='50th Percentile Rank',color='white')
        # ax.axvline(25, linestyle='--', label='25th Percentile Rank', color='white')
        # ax.axvline(75, linestyle='--', label='75th Percentile Rank', color='white')

        new_labels = req_columns


        # Set the new labels for the y-axis ticks
        ax.set_yticklabels(new_labels,font='Tw Cen MT', color='white') 

        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_yticks(range(len(metrics)))

        max_percentile_rank = max([max(percentile_ranks[metric]) for metric in metrics])
        ax.set_xlim(0, max_percentile_rank + 10)

        ax.set_xlabel('Percentile Ranks', color='white',fontname = "Tw Cen MT",size=20)
        # ax.set_ylabel('Metrics', color='white',fontname = "Tw Cen MT",size=20)
        plt.xticks(fontname = "Tw Cen MT",color="white",size=12)
        ax.set_xticks([0, 25, 50,75, 100])

        plt.yticks(fontname = "Tw Cen MT",color="white",size=15)


        ax.legend().remove()


        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.spines['left'].set_visible(False)  # Hide the left spine

        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')

        top_boxes = [
            Rectangle((i * 20, 20), 20, 5, color=colors[i], edgecolor='black', linewidth=1) for i in range(len(colors))
        ]

        texts =["Bottom 25%","From 25-50%","From 50-75%","Top 25%"]
        for i, box in enumerate(top_boxes):
            ax.add_patch(box)
            ax.text(
                i * 24 + 15, -0.8, f'{texts[i]}',font="Tw Cen MT", va='center', ha='center', color=colors[i], fontweight='bold', fontsize=25
            )

        plt.subplots_adjust(left=0.2)

        # # add credits
        # notes = f'All Stats per 90'
        # fig.text(
        # 0.27, 0.067, f"{notes}",
        # font = "Century Gothic",size =11,color="white",
        # ha="right"
        # )


        plt.savefig("data/images/bar.jpg",dpi =500, bbox_inches='tight')

        fig = plt.figure(figsize = (10,8), dpi = 300)

        fig.set_facecolor('black')
        plt.rcParams['hatch.linewidth'] = .02

        # one-liner
        player_df = defend.loc[defend['Player']==player_name]


        time = float(player_df.iloc[0,5])
        Name = str(player_df.iloc[0,0])
        Team = str(player_df.iloc[0,1])
        age =player_df.iloc[0,3]

        if len(player_name) > 20:
            fn = 14
        elif len(player_name) > 15:
            fn = 17
        else : 
            fn = 22

        im1 = plt.imread(r"C:\Users\lolen\OneDrive\Documents\Coding\Neurotactic Essentials\Images\Neuro.png")
        ax_image = add_image( im1, fig, left=1.02, bottom=0.982, width=0.13, height=0.13 )   # these values might differ when you are plotting




        # NAME

        #Heading
        str_text = f'''Player Data Report'''

        fig_text(
            x = 0.55, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="white",
            bbox=dict(boxstyle='round', facecolor='none', edgecolor='#4A9BD4', linewidth=2)
        )


        #Heading
        str_text = f'''--------------------------------------------------'''

        fig_text(
            x = 0.15, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )


        #Heading
        str_text = f'''-----------------------------------------------'''

        fig_text(
            x = 0.775, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )



        Attacking_Mid_score = z_scores_df[z_scores_df['Player']==x]["Defensive_Mid"].values[0]
        attack_df = z_scores_df.sort_values(by='Defensive_Mid', ascending=False).reset_index(drop=True)
        Attacking_Mid_rank = attack_df.index[attack_df['Player'] == x].values[0] +1 


        fig.text(
        x = 1.14, y = 0.945, s = f"Defensive Midfielder Rating : {Attacking_Mid_score:.2f} | Rank : {Attacking_Mid_rank} / {len(defend)}",
        ha="right", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )


        # add radar
        im1 = plt.imread(r"data/images/pizza.jpg")
        ax_image = add_image(
                im1, fig, left=0.691, bottom=0.4, width=0.53, height=0.53
            )   # these values might differ when you are plotting


        im1 = plt.imread(r"data/images/bar.jpg")
        ax_image = add_image(
                im1, fig, left=0.14 ,bottom=0.4, width=0.6, height=0.6
            )   # these values might differ when you are plotting

        fig_text(
            x = 0.154, y = 1.04, 
            s = f'<{player_name}-{Team}>',
            highlight_textprops=[{"color":"#FFD230"}],
            fontname ="Century Gothic",path_effects=[path_effects.Stroke(linewidth=0.4, foreground="#BD8B00"), path_effects.Normal()],
            va = 'bottom', ha = 'left',
            fontsize = fn,  weight = 'bold',color="white"
        )

        fig.text(
        x = 0.15, y = 1.024, s = f" Minutes Played : {time:.0f} | Age : {age:.0f} | Season : 23-24",
        ha="left", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )
        st.pyplot(fig)
        
elif sec_pos == 'Center Back':
    defend["Pass_Final_3rd_Acc"] = defend['Passes to final third per 90'] *defend['Accurate passes to final third, %'] * 0.01
                
    defend["Pass_Prog_Acc"] =  defend['Progressive passes per 90'] * defend['Accurate progressive passes, %'] * 0.01

    defend["Pass_Forward_Acc"] =  defend['Forward passes per 90'] *defend['Accurate forward passes, %'] *0.01 

    defend["Pass_Total_Acc"] =  defend['Passes per 90'] *  defend['Accurate passes, %'] * 0.01

    defend["Def_Duels_Acc"] = defend['Defensive duels per 90'] *defend['Defensive duels won, %'] * 0.01
                
    defend["Aerial_Duels_Acc"] =   defend['Aerial duels per 90'] * defend['Aerial duels won, %'] * 0.01

    z_scores_df = defend.copy()  # Create a copy to preserve the original DataFrame
    numeric_columns = defend.select_dtypes(include='number').columns

    # Calculate Z-scores using the correct standard deviation for each column
    for col in numeric_columns:
        z_scores_df[col] = zscore(defend[col])

    # Min-Max scaling to scale Z-scores from 0 to 100
    z_scores_df = ((z_scores_df[numeric_columns] - z_scores_df[numeric_columns].min()) / 
                (z_scores_df[numeric_columns].max() - z_scores_df[numeric_columns].min())) * 100
    # Print the resulting DataFrame with Z-scores
    # Combine scaled values with non-numeric columns
    z_scores_df = pd.concat([defend[['Player','Team']], z_scores_df], axis=1)


        
    # 1.  Game_Involvement : 
    weights = [0.1,0.2,0.3,0.4]

    z_scores_df["Passing"] = (z_scores_df['Pass_Final_3rd_Acc'] * weights[0] +
                                    z_scores_df['Pass_Prog_Acc'] * weights[1] +
                                    z_scores_df['Pass_Forward_Acc'] * weights[2]+
                                    z_scores_df['Pass_Total_Acc'] * weights[3])
    # 1.  Game_Involvement : 
    weights = [0.4,0.3,0.1,0.1,0.1]

    z_scores_df["Defending"] = (z_scores_df['Def_Duels_Acc'] * weights[0] +
                                z_scores_df['Aerial_Duels_Acc'] * weights[1] +
                                z_scores_df['Shots blocked per 90'] * weights[2]+
                                z_scores_df['PAdj Sliding tackles'] * weights[3]+
                                z_scores_df['PAdj Interceptions'] * weights[4])


    # z_scores_df.sort_values(by ="Passing",ascending=False )[:1]
    # Combined Stats

    weights = [0.7,0.3]
    z_scores_df["Centre_Back"] = (z_scores_df['Passing'] * weights[1]
                                +z_scores_df['Defending'] * weights[0])


    # Min-Max scaling to scale values to a ranking between 50 and 100
    min_value = z_scores_df["Centre_Back"] .min()
    max_value = z_scores_df["Centre_Back"] .max()

    z_scores_df["Centre_Back"]  = 30 + ((z_scores_df["Centre_Back"]  - min_value) / (max_value - min_value)) * 70

    ratingss = z_scores_df.Centre_Back

    defend['Centre_Back'] = ratingss

    mean_values = defend.select_dtypes(include='number').mean().to_frame().T

    # Set a name for the index to distinguish the mean row
    mean_values.index = ['mean']

    # Concatenate the mean values as a new row
    defend = pd.concat([defend, mean_values], ignore_index=True)

    defend = defend.fillna("Avg")
    last_index = len(defend) - 1
    defend.iloc[last_index, 3] = 5000

    # defend.drop_duplicates(subset=['Player', 'Team','Position'])

    defend_p = defend.copy()
    # Function to calculate percentiles
    def calculate_percentiles(series):
        return series.rank(pct=True) * 100

    # Select all numerical columns
    numerical_columns = defend_p.select_dtypes(include='number').columns

    # Apply the percentile calculation to all numerical columns
    defend_p[numerical_columns] = defend_p[numerical_columns].apply(calculate_percentiles)


    filter_prem = defend_p.copy()
    # prem = prem[prem["Age"] < 30]
    x =player_name
    player_fil = filter_prem.loc[filter_prem['Player']==x]



    # Select all numerical columns
    numerical_columns = player_fil.select_dtypes(include='number').columns

    # Set percentile filter
    # min_percentile = st.slider("Set minimum percentile cutoff", 0, 99, format=None, key=None)
    min_percentile = 70
    player_fil = player_fil[player_fil[numerical_columns] > min_percentile].dropna(axis=1)


    req_columns = player_fil.columns


    columns = req_columns
    selected_columns = []

    st.markdown("""
        <style>
        .slider-label {
            font-family: Century Gothic;
            font-size: 22px;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<span class='slider-label'>Select Columns to plot</span>", unsafe_allow_html=True)

    for col in columns:
        selected = st.checkbox(col, value=False, key=col)
        if selected:
            selected_columns.append(col)

    # st.write("Selected columns:", selected_columns)
    status1 = [ "No","Yes"]
    status = st.radio("Show Plot",horizontal=True,options = status1)

    if status =="Yes":

        req_columns = selected_columns
                # Radar
        color_1 = "#F98866"
        color_2 = "#408EC6"
        x =	player_name
        prem = defend.copy()
        player = prem.loc[prem['Player']==x]
        player2 = prem.loc[prem['Player']== "Avg"]

        Name = str(player.iloc[0,0])

        kik = prem.copy()

        stat1 = []
        stat2 = []
        for x in req_columns:
            stat1.extend(player[x])
            stat2.extend(player2[x])

        params = req_columns


        lower_is_better = []
        # minimum range value and maximum range value for parameters
        min_range= []
        max_range =[]
        for x in req_columns:
            min_range.append(kik.loc[:,x].min())
            max_range.append(kik.loc[:,x].max())          
        stat11 = [ round(x, 2) for x in stat1]        
        stat22 = [ round(x, 2) for x in stat2]  

        radar = Radar(params, min_range, max_range,
                    lower_is_better=lower_is_better,
                    # whether to round any of the labels to integers instead of decimal places
                    round_int=[False]*len(params),
                    num_rings=3,  # the number of concentric circles (excluding center circle)
                    # if the ring_width is more than the center_circle_radius then
                    # the center circle radius will be wider than the width of the concentric circles
                    ring_width=1, center_circle_radius=1)



        fig, ax = radar.setup_axis()
        color44 = "black"
        color45= "#000544"
        color46 ='#4FC0E8'
        fig.set_facecolor("black")
        ax.patch.set_facecolor(color44)
        radar.setup_axis(ax=ax, facecolor='None')  # format axis as a radar
        rings_inner = radar.draw_circles(ax=ax, facecolor=color44, edgecolor=color46)
        radar_output = radar.draw_radar_compare(stat11, stat22, ax=ax,
                                                kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                                kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
        radar_poly, radar_poly2, vertices1, vertices2 = radar_output
        range_labels = radar.draw_range_labels(ax=ax, fontsize=15,color='white',
                                                font="STXihei",weight="bold")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=15,color='white',
                                            font="STXihei",weight="bold")

        ax.scatter(vertices1[:, 0], vertices1[:, 1],
                                    c="#00f2c1", edgecolors='#00f2c1', marker='o', s=30, zorder=2)
        ax.scatter(vertices2[:, 0], vertices2[:, 1],
                                c='#d80499', edgecolors='#d80499', marker='o', s=30, zorder=2)

        # add title
                # add title
                # add title
        fig.text(
        0.515, 0.934, f"Center Back Radar",
        path_effects=[path_effects.Stroke(linewidth=0.2, foreground="white"), path_effects.Normal()],
        ha="center", font = "Century Gothic",size =39,color="white",fontweight="bold"
        )



        # add credits
        notes = f'Stats compared only with\nplayers with minutes > {minutes} and age>= {age}'
        fig.text(
        0.895, 0.08, f"{notes}",
        font = "Century Gothic",size =13,color="white",
        ha="right"
        )

        # Define circle properties
        center = (0.38, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#00f2c1')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.40, 0.895, f"{Name}",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )

        # Define circle properties
        center = (0.6, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#d80499')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.62, 0.895, f"Average",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )
        plt.savefig("data/images/pizza.jpg",dpi =500, bbox_inches='tight')
        # st.pyplot(fig)
        
        defend.reset_index(drop=True, inplace=True)  # Reset the index after filtering

        # select stats
        x =player_name
        player_df = defend.loc[defend['Player']==x]

        # Define the desired order of metrics
        metrics = req_columns

        # Calculate percentile ranks for each metric
        percentile_ranks = {}
        for metric in metrics:
            percentile_ranks[metric] = rankdata(defend[metric], method='average') / len(defend) * 99

        # Define colors and create a colormap
        colors = ['red', 'orange', 'yellow', 'green']
        percentile_ranges = [ 25, 50, 75, 100]
        cmap = ListedColormap(colors)


        # Create a bar graph with black background and facecolor
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
        ax.set_facecolor('black')


        ax.axvline(x=25, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=50, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=75, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=100, linestyle='--', color='white', linewidth=0.8)

        for i, metric in enumerate(metrics):
            metric_value = player_df[metric].values[0]
            percentile_rank = percentile_ranks[metric][player_df.index[0]]

            color_index = next(idx for idx, pct in enumerate(percentile_ranges) if pct >= percentile_rank)
            
            # Assign color based on the index
            color = cmap(color_index)

        # Assign color based on percentile rank
            bar = ax.barh(i, percentile_rank, height=0.3, alpha=0.7, color=color,edgecolor ="white",linewidth =0.8 ,zorder=1)
            ax.text(
                105, bar[0].get_y() + bar[0].get_height() / 2, f'{metric_value:.2f}', va='center', ha='left',
                font='STXihei', size=15, color='white'
            )

        # ax.axvline(50, linestyle='--', label='50th Percentile Rank',color='white')
        # ax.axvline(25, linestyle='--', label='25th Percentile Rank', color='white')
        # ax.axvline(75, linestyle='--', label='75th Percentile Rank', color='white')

        new_labels = req_columns


        # Set the new labels for the y-axis ticks
        ax.set_yticklabels(new_labels,font='Tw Cen MT', color='white') 

        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_yticks(range(len(metrics)))

        max_percentile_rank = max([max(percentile_ranks[metric]) for metric in metrics])
        ax.set_xlim(0, max_percentile_rank + 10)

        ax.set_xlabel('Percentile Ranks', color='white',fontname = "Tw Cen MT",size=20)
        # ax.set_ylabel('Metrics', color='white',fontname = "Tw Cen MT",size=20)
        plt.xticks(fontname = "Tw Cen MT",color="white",size=12)
        ax.set_xticks([0, 25, 50,75, 100])

        plt.yticks(fontname = "Tw Cen MT",color="white",size=15)


        ax.legend().remove()


        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.spines['left'].set_visible(False)  # Hide the left spine

        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')

        top_boxes = [
            Rectangle((i * 20, 20), 20, 5, color=colors[i], edgecolor='black', linewidth=1) for i in range(len(colors))
        ]

        texts =["Bottom 25%","From 25-50%","From 50-75%","Top 25%"]
        for i, box in enumerate(top_boxes):
            ax.add_patch(box)
            ax.text(
                i * 24 + 15, -0.8, f'{texts[i]}',font="Tw Cen MT", va='center', ha='center', color=colors[i], fontweight='bold', fontsize=25
            )

        plt.subplots_adjust(left=0.2)

        # # add credits
        # notes = f'All Stats per 90'
        # fig.text(
        # 0.27, 0.067, f"{notes}",
        # font = "Century Gothic",size =11,color="white",
        # ha="right"
        # )


        plt.savefig("data/images/bar.jpg",dpi =500, bbox_inches='tight')

        fig = plt.figure(figsize = (10,8), dpi = 300)

        fig.set_facecolor('black')
        plt.rcParams['hatch.linewidth'] = .02

        # one-liner
        player_df = defend.loc[defend['Player']==player_name]


        time = float(player_df.iloc[0,5])
        Name = str(player_df.iloc[0,0])
        Team = str(player_df.iloc[0,1])
        age =player_df.iloc[0,3]

        if len(player_name) > 20:
            fn = 14
        elif len(player_name) > 15:
            fn = 17
        else : 
            fn = 22

        im1 = plt.imread(r"C:\Users\lolen\OneDrive\Documents\Coding\Neurotactic Essentials\Images\Neuro.png")
        ax_image = add_image( im1, fig, left=1.02, bottom=0.982, width=0.13, height=0.13 )   # these values might differ when you are plotting




        # NAME

        #Heading
        str_text = f'''Player Data Report'''

        fig_text(
            x = 0.55, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="white",
            bbox=dict(boxstyle='round', facecolor='none', edgecolor='#4A9BD4', linewidth=2)
        )


        #Heading
        str_text = f'''--------------------------------------------------'''

        fig_text(
            x = 0.15, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )


        #Heading
        str_text = f'''-----------------------------------------------'''

        fig_text(
            x = 0.775, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )



        Attacking_Mid_score = z_scores_df[z_scores_df['Player']==x]["Centre_Back"].values[0]
        attack_df = z_scores_df.sort_values(by='Centre_Back', ascending=False).reset_index(drop=True)
        Attacking_Mid_rank = attack_df.index[attack_df['Player'] == x].values[0] +1 


        fig.text(
        x = 1.14, y = 0.945, s = f"Centre Back Rating : {Attacking_Mid_score:.2f} | Rank : {Attacking_Mid_rank}/{len(defend)}",
        ha="right", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )

        # add radar
        im1 = plt.imread(r"data/images/pizza.jpg")
        ax_image = add_image(
                im1, fig, left=0.691, bottom=0.4, width=0.53, height=0.53
            )   # these values might differ when you are plotting


        im1 = plt.imread(r"data/images/bar.jpg")
        ax_image = add_image(
                im1, fig, left=0.14 ,bottom=0.4, width=0.6, height=0.6
            )   # these values might differ when you are plotting

        fig_text(
            x = 0.154, y = 1.04, 
            s = f'<{player_name}-{Team}>',
            highlight_textprops=[{"color":"#FFD230"}],
            fontname ="Century Gothic",path_effects=[path_effects.Stroke(linewidth=0.4, foreground="#BD8B00"), path_effects.Normal()],
            va = 'bottom', ha = 'left',
            fontsize = fn,  weight = 'bold',color="white"
        )

        fig.text(
        x = 0.15, y = 1.024, s = f" Minutes Played : {time:.0f} | Age : {age:.0f} | Season : 23-24",
        ha="left", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )
        st.pyplot(fig)
elif sec_pos == 'Full Back':

    defend["Succ Dribbles per 90"] = defend['Dribbles per 90'] *defend['Successful dribbles, %'] * 0.01
    defend["Acc Passes per 90"] = defend['Passes per 90'] *defend['Accurate passes, %'] * 0.01
    defend["Possesion Lost per 90"] = (defend['Dribbles per 90'] -defend["Succ Dribbles per 90"]) + (defend['Passes per 90'] - defend["Acc Passes per 90"])
    defend["Acc Passes to final3rd per 90"] = defend['Passes to final third per 90'] *defend['Accurate passes to final third, %'] * 0.01
    defend["Acc Forward Passes per 90"] = defend['Forward passes per 90'] *defend['Accurate forward passes, %'] * 0.01
    defend["Acc Long Passes per 90"] = defend['Long passes per 90'] *defend['Accurate long passes, %'] * 0.01
    defend["Acc Passes PenArea per 90"] = defend['Passes to penalty area per 90'] *defend['Accurate passes to penalty area, %'] * 0.01
    defend["Acc Through Passes per 90"] = defend['Through passes per 90'] *defend['Accurate through passes, %'] * 0.01
    defend["Acc Crosses per 90"] = defend['Crosses per 90'] *defend['Accurate crosses, %'] * 0.01
    defend["Pass Prog Acc per 90"] =  defend['Progressive passes per 90'] * defend['Accurate progressive passes, %'] * 0.01
    defend["Def Duels Won"] = defend['Defensive duels per 90'] *defend['Defensive duels won, %'] * 0.01
    defend["Aerial Duels Won"] =   defend['Aerial duels per 90'] * defend['Aerial duels won, %'] * 0.01

    z_scores_df = defend.copy()  # Create a copy to preserve the original DataFrame
    numeric_columns = defend.select_dtypes(include='number').columns

    # Calculate Z-scores using the correct standard deviation for each column
    for col in numeric_columns:
        z_scores_df[col] = zscore(defend[col])

    # Min-Max scaling to scale Z-scores from 0 to 100
    z_scores_df = ((z_scores_df[numeric_columns] - z_scores_df[numeric_columns].min()) / 
                (z_scores_df[numeric_columns].max() - z_scores_df[numeric_columns].min())) * 100
    # Print the resulting DataFrame with Z-scores
    # Combine scaled values with non-numeric columns
    z_scores_df = pd.concat([defend[['Player','Team']], z_scores_df], axis=1)


    
    # 1. Play Making : xA,key passes,through balls.assists
    weights = [0.1,0.2,0.35,0.25,0.1]
    z_scores_df["Play_Making"] = (z_scores_df['Assists per 90'] * weights[0] +z_scores_df['Acc Crosses per 90'] * weights[4]+
                                z_scores_df['xA per 90'] * weights[1] +
                                z_scores_df['Key passes per 90'] * weights[2]+z_scores_df['Acc Through Passes per 90'] * weights[3]
                                )
    # 1.  Goal_Threat : 
    weights = [0.2,0.3,0.4,0.1]
    z_scores_df["Goal_Threat"] = (z_scores_df['Non-penalty goals per 90'] * weights[0] +
                                z_scores_df['xG per 90'] * weights[1] +
                                z_scores_df['Shots per 90'] * weights[2]+z_scores_df['Shots on target, %'] * weights[3])
    # 1.  Dribbling : 
    weights = [0.1,0.3,0.3,0.3]
    z_scores_df["Possession"] = (z_scores_df['Touches in box per 90'] * weights[0] +
                                z_scores_df['Succ Dribbles per 90'] * weights[1]+z_scores_df['Progressive runs per 90'] * weights[2] +
                                z_scores_df['Successful attacking actions per 90'] * weights[3] )

    # 1.  Passing : 
    weights = [0.3,0.1,0.2,0.2,0.2]
    z_scores_df["Passing"] = (z_scores_df['Acc Passes to final3rd per 90'] * weights[0] +
                                z_scores_df['Acc Long Passes per 90'] * weights[1]+z_scores_df['Acc Passes per 90'] * weights[2] +
                            z_scores_df["Acc Forward Passes per 90"] * weights[3]+    z_scores_df["Pass Prog Acc per 90"] * weights[3])

    # 1.  Game_Involvement : 
    weights = [0.4,0.2,0.2,0.2]
    z_scores_df["Defending"] = (z_scores_df['Def Duels Won'] * weights[0] +
                                z_scores_df['Sliding tackles per 90'] * weights[1] +
                                z_scores_df['Interceptions per 90'] * weights[2]+
                                z_scores_df['Aerial Duels Won'] * weights[3])

    # z_scores_df.sort_values(by ="Passing",ascending=False )[:1]
    # Combined Stats

    weights = [0.2,0.1,0.2,0.3,0.2]
    z_scores_df["Full_Back"] = (z_scores_df['Play_Making'] * weights[0] +
                                z_scores_df['Goal_Threat'] * weights[1] +
                                z_scores_df['Possession'] * weights[2]+z_scores_df['Passing'] * weights[3] +
                                z_scores_df['Defending'] * weights[4])


    # Min-Max scaling to scale values to a ranking between 50 and 100
    min_value = z_scores_df["Full_Back"] .min()
    max_value = z_scores_df["Full_Back"] .max()

    z_scores_df["Full_Back"]  = 30 + ((z_scores_df["Full_Back"]  - min_value) / (max_value - min_value)) * 70

    ratingss = z_scores_df.Full_Back

    defend['Full_Back'] = ratingss


    mean_values = defend.select_dtypes(include='number').mean().to_frame().T

    # Set a name for the index to distinguish the mean row
    mean_values.index = ['mean']

    # Concatenate the mean values as a new row
    defend = pd.concat([defend, mean_values], ignore_index=True)

    defend = defend.fillna("Avg")
    last_index = len(defend) - 1
    defend.iloc[last_index, 3] = 5000

    # defend.drop_duplicates(subset=['Player', 'Team','Position'])

    defend_p = defend.copy()
    # Function to calculate percentiles
    def calculate_percentiles(series):
        return series.rank(pct=True) * 100

    # Select all numerical columns
    numerical_columns = defend_p.select_dtypes(include='number').columns

    # Apply the percentile calculation to all numerical columns
    defend_p[numerical_columns] = defend_p[numerical_columns].apply(calculate_percentiles)


    filter_prem = defend_p.copy()
    # prem = prem[prem["Age"] < 30]
    x =player_name
    player_fil = filter_prem.loc[filter_prem['Player']==x]



    # Select all numerical columns
    numerical_columns = player_fil.select_dtypes(include='number').columns

    # Set percentile filter
    # min_percentile = st.slider("Set minimum percentile cutoff", 0, 99, format=None, key=None)
    min_percentile = 70
    player_fil = player_fil[player_fil[numerical_columns] > min_percentile].dropna(axis=1)


    req_columns = player_fil.columns


    columns = req_columns
    selected_columns = []

    st.markdown("""
        <style>
        .slider-label {
            font-family: Century Gothic;
            font-size: 22px;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<span class='slider-label'>Select Columns to plot</span>", unsafe_allow_html=True)

    for col in columns:
        selected = st.checkbox(col, value=False, key=col)
        if selected:
            selected_columns.append(col)

    # st.write("Selected columns:", selected_columns)
    status1 = [ "No","Yes"]
    status = st.radio("Show Plot",horizontal=True,options = status1)

    if status =="Yes":

        req_columns = selected_columns
                # Radar
        color_1 = "#F98866"
        color_2 = "#408EC6"
        x =	player_name
        prem = defend.copy()
        player = prem.loc[prem['Player']==x]
        player2 = prem.loc[prem['Player']== "Avg"]

        Name = str(player.iloc[0,0])

        kik = prem.copy()

        stat1 = []
        stat2 = []
        for x in req_columns:
            stat1.extend(player[x])
            stat2.extend(player2[x])

        params = req_columns


        lower_is_better = []
        # minimum range value and maximum range value for parameters
        min_range= []
        max_range =[]
        for x in req_columns:
            min_range.append(kik.loc[:,x].min())
            max_range.append(kik.loc[:,x].max())          
        stat11 = [ round(x, 2) for x in stat1]        
        stat22 = [ round(x, 2) for x in stat2]  

        radar = Radar(params, min_range, max_range,
                    lower_is_better=lower_is_better,
                    # whether to round any of the labels to integers instead of decimal places
                    round_int=[False]*len(params),
                    num_rings=3,  # the number of concentric circles (excluding center circle)
                    # if the ring_width is more than the center_circle_radius then
                    # the center circle radius will be wider than the width of the concentric circles
                    ring_width=1, center_circle_radius=1)



        fig, ax = radar.setup_axis()
        color44 = "black"
        color45= "#000544"
        color46 ='#4FC0E8'
        fig.set_facecolor("black")
        ax.patch.set_facecolor(color44)
        radar.setup_axis(ax=ax, facecolor='None')  # format axis as a radar
        rings_inner = radar.draw_circles(ax=ax, facecolor=color44, edgecolor=color46)
        radar_output = radar.draw_radar_compare(stat11, stat22, ax=ax,
                                                kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                                kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
        radar_poly, radar_poly2, vertices1, vertices2 = radar_output
        range_labels = radar.draw_range_labels(ax=ax, fontsize=15,color='white',
                                                font="STXihei",weight="bold")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=15,color='white',
                                            font="STXihei",weight="bold")

        ax.scatter(vertices1[:, 0], vertices1[:, 1],
                                    c="#00f2c1", edgecolors='#00f2c1', marker='o', s=30, zorder=2)
        ax.scatter(vertices2[:, 0], vertices2[:, 1],
                                c='#d80499', edgecolors='#d80499', marker='o', s=30, zorder=2)

        # add title
                # add title
                # add title
        fig.text(
        0.515, 0.934, f"Full Back Radar",
        path_effects=[path_effects.Stroke(linewidth=0.2, foreground="white"), path_effects.Normal()],
        ha="center", font = "Century Gothic",size =39,color="white",fontweight="bold"
        )




        # add credits
        notes = f'Stats compared only with\nplayers with minutes > {minutes} and age>= {age}'
        fig.text(
        0.895, 0.08, f"{notes}",
        font = "Century Gothic",size =13,color="white",
        ha="right"
        )

        # Define circle properties
        center = (0.38, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#00f2c1')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.40, 0.895, f"{Name}",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )

        # Define circle properties
        center = (0.6, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#d80499')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.62, 0.895, f"Average",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )
        plt.savefig("data/images/pizza.jpg",dpi =500, bbox_inches='tight')
        # st.pyplot(fig)
        
        defend.reset_index(drop=True, inplace=True)  # Reset the index after filtering

        # select stats
        x =player_name
        player_df = defend.loc[defend['Player']==x]

        # Define the desired order of metrics
        metrics = req_columns

        # Calculate percentile ranks for each metric
        percentile_ranks = {}
        for metric in metrics:
            percentile_ranks[metric] = rankdata(defend[metric], method='average') / len(defend) * 99

        # Define colors and create a colormap
        colors = ['red', 'orange', 'yellow', 'green']
        percentile_ranges = [ 25, 50, 75, 100]
        cmap = ListedColormap(colors)


        # Create a bar graph with black background and facecolor
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
        ax.set_facecolor('black')


        ax.axvline(x=25, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=50, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=75, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=100, linestyle='--', color='white', linewidth=0.8)

        for i, metric in enumerate(metrics):
            metric_value = player_df[metric].values[0]
            percentile_rank = percentile_ranks[metric][player_df.index[0]]

            color_index = next(idx for idx, pct in enumerate(percentile_ranges) if pct >= percentile_rank)
            
            # Assign color based on the index
            color = cmap(color_index)

        # Assign color based on percentile rank
            bar = ax.barh(i, percentile_rank, height=0.3, alpha=0.7, color=color,edgecolor ="white",linewidth =0.8 ,zorder=1)
            ax.text(
                105, bar[0].get_y() + bar[0].get_height() / 2, f'{metric_value:.2f}', va='center', ha='left',
                font='STXihei', size=15, color='white'
            )

        # ax.axvline(50, linestyle='--', label='50th Percentile Rank',color='white')
        # ax.axvline(25, linestyle='--', label='25th Percentile Rank', color='white')
        # ax.axvline(75, linestyle='--', label='75th Percentile Rank', color='white')

        new_labels = req_columns


        # Set the new labels for the y-axis ticks
        ax.set_yticklabels(new_labels,font='Tw Cen MT', color='white') 

        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_yticks(range(len(metrics)))

        max_percentile_rank = max([max(percentile_ranks[metric]) for metric in metrics])
        ax.set_xlim(0, max_percentile_rank + 10)

        ax.set_xlabel('Percentile Ranks', color='white',fontname = "Tw Cen MT",size=20)
        # ax.set_ylabel('Metrics', color='white',fontname = "Tw Cen MT",size=20)
        plt.xticks(fontname = "Tw Cen MT",color="white",size=12)
        ax.set_xticks([0, 25, 50,75, 100])

        plt.yticks(fontname = "Tw Cen MT",color="white",size=15)


        ax.legend().remove()


        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.spines['left'].set_visible(False)  # Hide the left spine

        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')

        top_boxes = [
            Rectangle((i * 20, 20), 20, 5, color=colors[i], edgecolor='black', linewidth=1) for i in range(len(colors))
        ]

        texts =["Bottom 25%","From 25-50%","From 50-75%","Top 25%"]
        for i, box in enumerate(top_boxes):
            ax.add_patch(box)
            ax.text(
                i * 24 + 15, -0.8, f'{texts[i]}',font="Tw Cen MT", va='center', ha='center', color=colors[i], fontweight='bold', fontsize=25
            )

        plt.subplots_adjust(left=0.2)

        # # add credits
        # notes = f'All Stats per 90'
        # fig.text(
        # 0.27, 0.067, f"{notes}",
        # font = "Century Gothic",size =11,color="white",
        # ha="right"
        # )


        plt.savefig("data/images/bar.jpg",dpi =500, bbox_inches='tight')

        fig = plt.figure(figsize = (10,8), dpi = 300)

        fig.set_facecolor('black')
        plt.rcParams['hatch.linewidth'] = .02

        # one-liner
        player_df = defend.loc[defend['Player']==player_name]


        time = float(player_df.iloc[0,5])
        Name = str(player_df.iloc[0,0])
        Team = str(player_df.iloc[0,1])
        age =player_df.iloc[0,3]

        if len(player_name) > 20:
            fn = 14
        elif len(player_name) > 15:
            fn = 17
        else : 
            fn = 22

        im1 = plt.imread(r"C:\Users\lolen\OneDrive\Documents\Coding\Neurotactic Essentials\Images\Neuro.png")
        ax_image = add_image( im1, fig, left=1.02, bottom=0.982, width=0.13, height=0.13 )   # these values might differ when you are plotting




        # NAME

        #Heading
        str_text = f'''Player Data Report'''

        fig_text(
            x = 0.55, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="white",
            bbox=dict(boxstyle='round', facecolor='none', edgecolor='#4A9BD4', linewidth=2)
        )


        #Heading
        str_text = f'''--------------------------------------------------'''

        fig_text(
            x = 0.15, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )


        #Heading
        str_text = f'''-----------------------------------------------'''

        fig_text(
            x = 0.775, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )



        Attacking_Mid_score = z_scores_df[z_scores_df['Player']==x]["Full_Back"].values[0]
        attack_df = z_scores_df.sort_values(by='Full_Back', ascending=False).reset_index(drop=True)
        Attacking_Mid_rank = attack_df.index[attack_df['Player'] == x].values[0] +1 


        fig.text(
        x = 1.14, y = 0.945, s = f"Full Back  Rating : {Attacking_Mid_score:.2f} | Rank : {Attacking_Mid_rank} / {len(defend)}",
        ha="right", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )

        # add radar
        im1 = plt.imread(r"data/images/pizza.jpg")
        ax_image = add_image(
                im1, fig, left=0.691, bottom=0.4, width=0.53, height=0.53
            )   # these values might differ when you are plotting


        im1 = plt.imread(r"data/images/bar.jpg")
        ax_image = add_image(
                im1, fig, left=0.14 ,bottom=0.4, width=0.6, height=0.6
            )   # these values might differ when you are plotting

        fig_text(
            x = 0.154, y = 1.04, 
            s = f'<{player_name}-{Team}>',
            highlight_textprops=[{"color":"#FFD230"}],
            fontname ="Century Gothic",path_effects=[path_effects.Stroke(linewidth=0.4, foreground="#BD8B00"), path_effects.Normal()],
            va = 'bottom', ha = 'left',
            fontsize = fn,  weight = 'bold',color="white"
        )

        fig.text(
        x = 0.15, y = 1.024, s = f" Minutes Played : {time:.0f} | Age : {age:.0f} | Season : 23-24",
        ha="left", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )
        st.pyplot(fig)
elif sec_pos == 'Goalkeeper':

    defend["Acc Passes per 90"] = defend['Passes per 90'] *defend['Accurate passes, %'] * 0.01
    defend["Acc Forward Passes per 90"] = defend['Forward passes per 90'] *defend['Accurate forward passes, %'] * 0.01
    defend["Acc Long Passes per 90"] = defend['Long passes per 90'] *defend['Accurate long passes, %'] * 0.01

    z_scores_df = defend.copy()  # Create a copy to preserve the original DataFrame
    numeric_columns = defend.select_dtypes(include='number').columns

    # Calculate Z-scores using the correct standard deviation for each column
    for col in numeric_columns:
        z_scores_df[col] = zscore(defend[col])

    # Min-Max scaling to scale Z-scores from 0 to 100
    z_scores_df = ((z_scores_df[numeric_columns] - z_scores_df[numeric_columns].min()) / 
                (z_scores_df[numeric_columns].max() - z_scores_df[numeric_columns].min())) * 100
    # Print the resulting DataFrame with Z-scores
    # Combine scaled values with non-numeric columns
    z_scores_df = pd.concat([defend[['Player','Team']], z_scores_df], axis=1)


        # 1.  Passing : 
    weights = [0.4,0.3,0.3]
    z_scores_df["Passing"] = (z_scores_df['Acc Passes per 90'] * weights[0] +
                                z_scores_df['Acc Long Passes per 90'] * weights[1]+
                            z_scores_df["Acc Forward Passes per 90"] * weights[2])


    # 1.  Game_Involvement : 
    weights = [0.2,0.5,0.2,0.1]
    z_scores_df["Goalkeeping"] = (-z_scores_df['Conceded goals per 90'] * weights[0] +
                                    z_scores_df['Save rate, %']*weights[1] +
                                z_scores_df['Prevented goals per 90'] * weights[2]+z_scores_df['Clean sheets'] * weights[3])



    # z_scores_df.sort_values(by ="Passing",ascending=False )[:1]
    # Combined Stats
    weights = [0.1,0.9]
    z_scores_df["Goalkeeper"] = (z_scores_df['Passing'] * weights[0] +
                                z_scores_df['Goalkeeping'] * weights[1])


    # Min-Max scaling to scale values to a ranking between 50 and 100
    min_value = z_scores_df["Goalkeeper"] .min()
    max_value = z_scores_df["Goalkeeper"] .max()

    z_scores_df["Goalkeeper"]  = 30 + ((z_scores_df["Goalkeeper"]  - min_value) / (max_value - min_value)) * 70

    ratingss = z_scores_df.Goalkeeper

    defend['Goalkeeper'] = ratingss


    mean_values = defend.select_dtypes(include='number').mean().to_frame().T

    # Set a name for the index to distinguish the mean row
    mean_values.index = ['mean']

    # Concatenate the mean values as a new row
    defend = pd.concat([defend, mean_values], ignore_index=True)

    defend = defend.fillna("Avg")
    last_index = len(defend) - 1
    defend.iloc[last_index, 3] = 5000

    # defend.drop_duplicates(subset=['Player', 'Team','Position'])

    defend_p = defend.copy()
    # Function to calculate percentiles
    def calculate_percentiles(series):
        return series.rank(pct=True) * 100

    # Select all numerical columns
    numerical_columns = defend_p.select_dtypes(include='number').columns

    # Apply the percentile calculation to all numerical columns
    defend_p[numerical_columns] = defend_p[numerical_columns].apply(calculate_percentiles)


    filter_prem = defend_p.copy()
    # prem = prem[prem["Age"] < 30]
    x =player_name
    player_fil = filter_prem.loc[filter_prem['Player']==x]



    # Select all numerical columns
    numerical_columns = player_fil.select_dtypes(include='number').columns

    # Set percentile filter
    # min_percentile = st.slider("Set minimum percentile cutoff", 0, 99, format=None, key=None)
    min_percentile = 70
    player_fil = player_fil[player_fil[numerical_columns] > min_percentile].dropna(axis=1)


    req_columns = player_fil.columns


    columns = req_columns
    selected_columns = []

    st.markdown("""
        <style>
        .slider-label {
            font-family: Century Gothic;
            font-size: 22px;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<span class='slider-label'>Select Columns to plot</span>", unsafe_allow_html=True)

    for col in columns:
        selected = st.checkbox(col, value=False, key=col)
        if selected:
            selected_columns.append(col)

    # st.write("Selected columns:", selected_columns)
    status1 = [ "No","Yes"]
    status = st.radio("Show Plot",horizontal=True,options = status1)

    if status =="Yes":

        req_columns = selected_columns
                # Radar
        color_1 = "#F98866"
        color_2 = "#408EC6"
        x =	player_name
        prem = defend.copy()
        player = prem.loc[prem['Player']==x]
        player2 = prem.loc[prem['Player']== "Avg"]

        Name = str(player.iloc[0,0])

        kik = prem.copy()

        stat1 = []
        stat2 = []
        for x in req_columns:
            stat1.extend(player[x])
            stat2.extend(player2[x])

        params = req_columns


        lower_is_better = []
        # minimum range value and maximum range value for parameters
        min_range= []
        max_range =[]
        for x in req_columns:
            min_range.append(kik.loc[:,x].min())
            max_range.append(kik.loc[:,x].max())          
        stat11 = [ round(x, 2) for x in stat1]        
        stat22 = [ round(x, 2) for x in stat2]  

        radar = Radar(params, min_range, max_range,
                    lower_is_better=lower_is_better,
                    # whether to round any of the labels to integers instead of decimal places
                    round_int=[False]*len(params),
                    num_rings=3,  # the number of concentric circles (excluding center circle)
                    # if the ring_width is more than the center_circle_radius then
                    # the center circle radius will be wider than the width of the concentric circles
                    ring_width=1, center_circle_radius=1)



        fig, ax = radar.setup_axis()
        color44 = "black"
        color45= "#000544"
        color46 ='#4FC0E8'
        fig.set_facecolor("black")
        ax.patch.set_facecolor(color44)
        radar.setup_axis(ax=ax, facecolor='None')  # format axis as a radar
        rings_inner = radar.draw_circles(ax=ax, facecolor=color44, edgecolor=color46)
        radar_output = radar.draw_radar_compare(stat11, stat22, ax=ax,
                                                kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                                kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
        radar_poly, radar_poly2, vertices1, vertices2 = radar_output
        range_labels = radar.draw_range_labels(ax=ax, fontsize=15,color='white',
                                                font="STXihei",weight="bold")
        param_labels = radar.draw_param_labels(ax=ax, fontsize=15,color='white',
                                            font="STXihei",weight="bold")

        ax.scatter(vertices1[:, 0], vertices1[:, 1],
                                    c="#00f2c1", edgecolors='#00f2c1', marker='o', s=30, zorder=2)
        ax.scatter(vertices2[:, 0], vertices2[:, 1],
                                c='#d80499', edgecolors='#d80499', marker='o', s=30, zorder=2)

        # add title
                # add title
                # add title
       # add title
        fig.text(
        0.515, 0.934, f"GoalKeeper Radar",
        path_effects=[path_effects.Stroke(linewidth=0.2, foreground="white"), path_effects.Normal()],
        ha="center", font = "Century Gothic",size =39,color="white",fontweight="bold"
        )


        # add credits
        notes = f'Stats compared only with\nplayers with minutes > {minutes} and age>= {age}'
        fig.text(
        0.895, 0.08, f"{notes}",
        font = "Century Gothic",size =13,color="white",
        ha="right"
        )

        # Define circle properties
        center = (0.38, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#00f2c1')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.40, 0.895, f"{Name}",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )

        # Define circle properties
        center = (0.6, 0.9)  # center coordinates (x, y)
        radius = 0.012        # circle radius

        # Create a green circle object
        circle = plt.Circle(xy=center, radius=radius, color='#d80499')

        # **Add the circle artist directly to the figure**
        fig.add_artist(circle)

        fig.text(
        0.62, 0.895, f"Average",
        font = "Century Gothic",size =16,color="white",fontweight="bold",
        ha="left"
        )
        plt.savefig("data/images/pizza.jpg",dpi =500, bbox_inches='tight')
        # st.pyplot(fig)
        
        defend.reset_index(drop=True, inplace=True)  # Reset the index after filtering

        # select stats
        x =player_name
        player_df = defend.loc[defend['Player']==x]

        # Define the desired order of metrics
        metrics = req_columns

        # Calculate percentile ranks for each metric
        percentile_ranks = {}
        for metric in metrics:
            percentile_ranks[metric] = rankdata(defend[metric], method='average') / len(defend) * 99

        # Define colors and create a colormap
        colors = ['red', 'orange', 'yellow', 'green']
        percentile_ranges = [ 25, 50, 75, 100]
        cmap = ListedColormap(colors)


        # Create a bar graph with black background and facecolor
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='black')
        ax.set_facecolor('black')


        ax.axvline(x=25, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=50, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=75, linestyle='--', color='white', linewidth=0.8)
        ax.axvline(x=100, linestyle='--', color='white', linewidth=0.8)

        for i, metric in enumerate(metrics):
            metric_value = player_df[metric].values[0]
            percentile_rank = percentile_ranks[metric][player_df.index[0]]

            color_index = next(idx for idx, pct in enumerate(percentile_ranges) if pct >= percentile_rank)
            
            # Assign color based on the index
            color = cmap(color_index)

        # Assign color based on percentile rank
            bar = ax.barh(i, percentile_rank, height=0.3, alpha=0.7, color=color,edgecolor ="white",linewidth =0.8 ,zorder=1)
            ax.text(
                105, bar[0].get_y() + bar[0].get_height() / 2, f'{metric_value:.2f}', va='center', ha='left',
                font='STXihei', size=15, color='white'
            )

        # ax.axvline(50, linestyle='--', label='50th Percentile Rank',color='white')
        # ax.axvline(25, linestyle='--', label='25th Percentile Rank', color='white')
        # ax.axvline(75, linestyle='--', label='75th Percentile Rank', color='white')

        new_labels = req_columns


        # Set the new labels for the y-axis ticks
        ax.set_yticklabels(new_labels,font='Tw Cen MT', color='white') 

        ax.set_ylim(len(metrics) - 0.5, -0.5)
        ax.set_yticks(range(len(metrics)))

        max_percentile_rank = max([max(percentile_ranks[metric]) for metric in metrics])
        ax.set_xlim(0, max_percentile_rank + 10)

        ax.set_xlabel('Percentile Ranks', color='white',fontname = "Tw Cen MT",size=20)
        # ax.set_ylabel('Metrics', color='white',fontname = "Tw Cen MT",size=20)
        plt.xticks(fontname = "Tw Cen MT",color="white",size=12)
        ax.set_xticks([0, 25, 50,75, 100])

        plt.yticks(fontname = "Tw Cen MT",color="white",size=15)


        ax.legend().remove()


        ax.spines['top'].set_visible(False)  # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['bottom'].set_visible(False)  # Hide the bottom spine
        ax.spines['left'].set_visible(False)  # Hide the left spine

        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')

        top_boxes = [
            Rectangle((i * 20, 20), 20, 5, color=colors[i], edgecolor='black', linewidth=1) for i in range(len(colors))
        ]

        texts =["Bottom 25%","From 25-50%","From 50-75%","Top 25%"]
        for i, box in enumerate(top_boxes):
            ax.add_patch(box)
            ax.text(
                i * 24 + 15, -0.8, f'{texts[i]}',font="Tw Cen MT", va='center', ha='center', color=colors[i], fontweight='bold', fontsize=25
            )

        plt.subplots_adjust(left=0.2)

        # # add credits
        # notes = f'All Stats per 90'
        # fig.text(
        # 0.27, 0.067, f"{notes}",
        # font = "Century Gothic",size =11,color="white",
        # ha="right"
        # )


        plt.savefig("data/images/bar.jpg",dpi =500, bbox_inches='tight')

        fig = plt.figure(figsize = (10,8), dpi = 300)

        fig.set_facecolor('black')
        plt.rcParams['hatch.linewidth'] = .02

        # one-liner
        player_df = defend.loc[defend['Player']==player_name]


        time = float(player_df.iloc[0,5])
        Name = str(player_df.iloc[0,0])
        Team = str(player_df.iloc[0,1])
        age =player_df.iloc[0,3]

        if len(player_name) > 20:
            fn = 14
        elif len(player_name) > 15:
            fn = 17
        else : 
            fn = 22

        im1 = plt.imread(r"C:\Users\lolen\OneDrive\Documents\Coding\Neurotactic Essentials\Images\Neuro.png")
        ax_image = add_image( im1, fig, left=1.02, bottom=0.982, width=0.13, height=0.13 )   # these values might differ when you are plotting




        # NAME

        #Heading
        str_text = f'''Player Data Report'''

        fig_text(
            x = 0.55, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="white",
            bbox=dict(boxstyle='round', facecolor='none', edgecolor='#4A9BD4', linewidth=2)
        )


        #Heading
        str_text = f'''--------------------------------------------------'''

        fig_text(
            x = 0.15, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )


        #Heading
        str_text = f'''-----------------------------------------------'''

        fig_text(
            x = 0.775, y = 1, 
            s = str_text,
            fontname ="STXihei",
            va = 'bottom', ha = 'left',
            fontsize = 17,  weight = 'bold',color="#4A9BD4"
        )



        Attacking_Mid_score = z_scores_df[z_scores_df['Player']==x]["Goalkeeper"].values[0]
        attack_df = z_scores_df.sort_values(by='Goalkeeper', ascending=False).reset_index(drop=True)
        Attacking_Mid_rank = attack_df.index[attack_df['Player'] == x].values[0] +1 


        fig.text(
        x = 1.14, y = 0.945, s = f"Goalkeeper Rating : {Attacking_Mid_score:.2f} | Rank : {Attacking_Mid_rank}/{len(defend)}",
        ha="right", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )

        # add radar
        im1 = plt.imread(r"data/images/pizza.jpg")
        ax_image = add_image(
                im1, fig, left=0.691, bottom=0.4, width=0.53, height=0.53
            )   # these values might differ when you are plotting


        im1 = plt.imread(r"data/images/bar.jpg")
        ax_image = add_image(
                im1, fig, left=0.14 ,bottom=0.4, width=0.6, height=0.6
            )   # these values might differ when you are plotting

        fig_text(
            x = 0.154, y = 1.04, 
            s = f'<{player_name}-{Team}>',
            highlight_textprops=[{"color":"#FFD230"}],
            fontname ="Century Gothic",path_effects=[path_effects.Stroke(linewidth=0.4, foreground="#BD8B00"), path_effects.Normal()],
            va = 'bottom', ha = 'left',
            fontsize = fn,  weight = 'bold',color="white"
        )

        fig.text(
        x = 0.15, y = 1.024, s = f" Minutes Played : {time:.0f} | Age : {age:.0f} | Season : 23-24",
        ha="left", font = "Century Gothic",size =10,color="white",fontweight="bold"
        )
        st.pyplot(fig)
