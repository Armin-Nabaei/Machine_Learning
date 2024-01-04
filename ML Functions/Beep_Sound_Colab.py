from IPython.display import Audio, display

def beep():
    display(Audio(url='https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg', autoplay=True))

beep()
