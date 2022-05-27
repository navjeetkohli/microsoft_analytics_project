from imports import *

def contact():
    st.header(":mailbox: Get in Touch with Us!")

    #Building the contact form
    contact_form = """ 
        <form action="https://formsubmit.co/navjeetkohli02@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder = "Your Name" required>
            <input type="email" name="email" placeholder = "Your E-mail" required>
            <textarea name="message" placeholder="Your Message here"></textarea>
            <button type="submit">Send</button>
        </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)
    #Applying the CSS files
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style/style.css")