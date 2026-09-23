import streamlit as st


def show_home():
    st.title("CSC 8830 - Computer Vision Projects")
    st.write("**Student:** Jannati Chowdhury")
    st.write(
        "This web application contains my CSC 8830 Computer Vision module "
        "projects. Each module presents its own application and results."
    )
    st.write("Select a module from the sidebar to explore a project.")


st.set_page_config(
    page_title="CSC 8830 - Computer Vision Projects",
    layout="centered",
)

page = st.navigation(
    [
        st.Page(show_home, title="Home", default=True),
        st.Page(
            "Pages/app_module2.py",
            title="Module 2 - 2D Object Measurement",
            url_path="module-2",
        ),
        st.Page(
            "Pages/app_module3.py",
            title="Module 3 - Image Filtering",
            url_path="module-3",
        ),
    ]
)
page.run()
