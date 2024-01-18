import streamlit as st
import joblib

from custommodel import custommodel
from onemodel import onemodel

# Streamlit UI
def main():
    st.title('Fake Website Detection')

    # User input: enter a website URL
    website_url = st.text_input('Enter a website URL :')
    
    if st.button('Predict'):
        if not website_url:
            st.warning('Please enter a website URL.')
        else:
            # Preprocess the input text

            # Transform the new data using the loaded vectorizer
            processed_text_numeric_is_legit = custommodel(website=website_url)
            is_legit = processed_text_numeric_is_legit.is_legit
        
            
            if is_legit == True:
                st.success(f'The website is legitimate.')
            else:
                st.error(f'The website is "fake"')       
            

            # Display the prediction result
                

    st.title('Customer Care Detection')

    number = st.number_input('Enter the Number : ')

    if st.button('Predict Number'):
        if not number :
            st.warning('Please Enter the Number')
        else:
            processed_number_is_legit = onemodel(number=number)
            is_legit_number = processed_number_is_legit.is_legit_number


            if is_legit_number == True:
                st.success(f'The number is legitimate.')
            else:
                st.error(f'The number is "FAKE"')

if __name__ == '__main__':
    main()
