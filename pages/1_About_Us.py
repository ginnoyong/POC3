import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App - About Us"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About Us")
st.subheader("subheader")
st.write("""As an academic staff member at a polytechnic in Singapore, I understand that our responsibilities extend beyond teaching. \
         One of the more challenging tasks is providing guidance during major admission exercises. The admission process can be complex, \
         and academic staff can be less than being conversant in its knowledge as it is not part of our core function.""") 
st.write("""With approximately 30,000 students enrolling in Polytechnics and ITE each year, there is a high demand for assistance from \
        secondary school leavers and their parents or guardians. During admission periods, the volume of inquiries can be overwhelming for \
        only the admission staff to manage alone.""")
st.write("""Additionally, some inquiries are course-specific, which can be challenging for foremost, the admission staff. \
         It can also be challenging even for the academic staff, if the question pertains to a course outside of their own department.""")
st.write("""This chatbot was designed to address these challenges. By providing accurate and timely information, \
         it aims to help prospects navigate the admission process more smoothly and efficiently. \
         It also helps to reduces the amount of preparation required, ultimately saving time for both admission and academic staff involved in advising duties.""")
