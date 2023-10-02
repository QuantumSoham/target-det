import streamlit as st
import pandas as pd


# Create the Streamlit website
def website():
    st.set_page_config(page_title='Govt of India', page_icon='6.png', layout='wide')
    st.image('6.png', width=100)
    st.title('Govt of India')
    st.title(" ")
    st.title("HUMAN TARGET ACQUISITION SYSTEM ")
    st.title(" ")
    st.sidebar.header('User Input Parameters')

    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features



    st.image(['5.png'], width=100)
    st.write('Welcome to the Govt of India website!')
    st.image('3.png')
    st.title(" ")
    st.title("ARMY TABLE")


    # Use magic command to display an image
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Indian_Army_Special_Forces.jpg/800px-Indian_Army_Special_Forces.jpg",
        caption="Indian Army Special Forces")

    # Use magic command to display a video
    st.video("https://www.youtube.com/watch?v=5M1ZKPCLb4I")



if __name__ == '__main__':
    website()