import streamlit as st
from PIL import Image


st.set_page_config(layout="wide")

st.markdown("""<p style="font-family:Century Gothic; color:white; font-size: 80px; font-weight: bold;">Neurotactic.Footy</p>""",unsafe_allow_html=True)

st.markdown(
    """
    <p style="font-family:STXihei; color:white; font-size: 20px;">
    Neurotactic.Footy is a user-friendly Streamlit web application designed for the Neurotactic team. Here, we as football data analysts can gain deeper insights into players' performance using the tools on this webpage. 
    This intuitive app offers 1 key feature for now that allow users to explore and analyze player statistics in an interactive and visually appealing manner.
    
    <p style="font-family:Century Gothic; color:#38B6FF; font-size: 20px; font-weight: bold;">1. Player Data Report</p>
    <p style="font-family:STXihei; color:white; font-size: 20px;">Visualize a player's data ratings in terms of a bar and pizza plot across different statistical categories using your own Data</p>
    
    <p style="font-family:Century Gothic; color:#38B6FF; font-size: 20px;">2.Coming Soon</p>

    
    </p>
    """,unsafe_allow_html=True
)

    # <p style="font-family:Gill Sans; color:white; font-size: 17px;">Compare the performance of two players using radar plots, also known as spider plots</p>
    
    # <p style="font-family:Futura; color:#38B6FF; font-size: 17px;">3. Player Match Reports</p>
    # <p style="font-family:Gill Sans; color:white; font-size: 17px;">Dive into detailed match reports for a selected player's recent game using event data</p>
    
    # <p style="font-family:Futura; color:#38B6FF; font-size: 17px;">4. Player Season Reports</p>
    # <p style="font-family:Gill Sans; color:white; font-size: 17px;">Dive into summarized season reports for a selected player for a particular season using event data</p>


# image1 = Image.open('data/images/pizza.jpeg')
# image2 = Image.open('data/images/Radar.jpeg')
# image3 = Image.open('data/images/matcha.jpeg')
# image4 = Image.open('data/images/Saka.png')

# st.image([image1,image2,image3,image4],caption=["Pizza Plot","Radar Plot","Match Report","Season Report"],width=260)

# st.markdown('<p style="font-family:Futura; color:#38B6FF; font-size: 15px;">Data: FBREF,Opta Made by : Ligandro</p>',unsafe_allow_html=True)
