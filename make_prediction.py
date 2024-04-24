import os
import time
from tkinter import *
import glob
from tkinter.filedialog import askopenfilename
from train_model import predict_song_genre
'''
def submit():
    entry.delete(0,END)
    entry.config(state="disabled")
    print(predict_song_genre(file))
    #time.sleep(5)
    #entry.config(state="normal")
    #entry.insert(0,"testing")
'''


def get_prediction(song_file):
    mfile = os.path.abspath("models/TEST1.pkl")
    return predict_song_genre(file, mfile, testing=False)


if __name__ == "__main__":
    file = askopenfilename()
    print(get_prediction(file))

"""window = Tk()
entry = Entry()
submit = Button(window, text="Submit",command=submit)
submit.pack(side="bottom")
entry.config(bg="Black")
entry.config(fg="Green")
entry.config(font=("Times", 40))
entry.pack()
window.mainloop()"""
#print(os.path.abspath("features_to_csv.py"))

