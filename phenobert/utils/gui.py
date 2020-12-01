from tkinter import *
# from api import annotate_text
import hashlib
import time

LOG_LINE_NUM = 0

class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name


    def set_init_window(self):
        self.init_window_name.title("PhenoBERT")
        self.init_window_name.geometry('1068x681+10+10')

        self.init_data_label = Label(self.init_window_name, text="Enter your free text here")
        self.init_data_label.grid(row=4, column=0)
        self.result_data_label = Label(self.init_window_name, text="Annotation results")
        self.result_data_label.grid(row=0, column=13)
        #文本框
        self.init_data_Text = Text(self.init_window_name, width=60, height=25)
        self.init_data_Text.grid(row=10, column=0, rowspan=10, columnspan=10)
        self.result_data_Text = Text(self.init_window_name, width=40, height=30)
        self.result_data_Text.grid(row=1, column=13, rowspan=15, columnspan=10)
        #按钮
        self.annotate_button = Button(self.init_window_name, text="Annotate", bg="lightblue", width=10,command=self.annotate())
        self.annotate_button.grid(row=1, column=10)

        self.clear_button = Button(self.init_window_name, text="Clear", bg="lightblue", width=10,
                                      command=self.clear())
        self.clear_button.grid(row=4, column=10)

        photo = PhotoImage(file="../img/logo.jpg")
        label = Label(image=photo)
        label.image = photo
        label.grid(row=0, column=2)


    def annotate(self):
        text = self.init_data_Text.get(1.0,END)
        print(text)
        if text:
            try:
                res = "annotate_text(text)"
                self.result_data_Text.insert(1.0, text)
            except:
                self.result_data_Text.delete(1.0,END)
                self.result_data_Text.insert(1.0,"Annotate fail")
        else:
            self.result_data_Text.clipboard_clear()
            self.result_data_Text.insert(1.0, "Please enter free text")

    def clear(self):
        pass

def gui_start():
    init_window = Tk()
    ZMJ_PORTAL = MY_GUI(init_window)

    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()


gui_start()