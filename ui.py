if delete_btn:
    st.session_state.tweet_text = ""
    st.experimental_rerun()  # This will refresh the app and update the text area
