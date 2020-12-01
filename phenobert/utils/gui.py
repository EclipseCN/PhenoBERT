from tkinter import *
from api import annotate_text


class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name


    def set_init_window(self):
        self.init_window_name.title("PhenoBERT")
        # self.init_window_name.geometry('800x800')

        photo = PhotoImage(file="../img/logo_gui.gif")
        label = Label(self.init_window_name, image=photo)
        label.image = photo
        label.grid(row=0, column=0, rowspan=3, sticky =W)

        self.init_data_label = Label(self.init_window_name, text="Enter your free text here")
        self.init_data_label.grid(row=3, column=0, sticky=W)
        self.result_data_label = Label(self.init_window_name, text="Annotation results")
        self.result_data_label.grid(row=5, column=0, sticky=W)
        name = Label(self.init_window_name, text="Author: NeoFengyh").grid(row=0, column=1, sticky=W)
        mail = Label(self.init_window_name, text="E-mail: 18210700100@fudan.edu.cn").grid(row=1, column=1, sticky=W)
        mail = Label(self.init_window_name, text="Page: https://github.com/EclipseCN/PhenoBERT").grid(row=2, column=1, sticky=W)
        self.log_data_Text = Text(self.init_window_name, width=10, height=2, bg="yellow")
        self.log_data_Text.grid(row=0, column=3)
        #文本框
        self.init_data_Text = Text(self.init_window_name, height=15)
        self.init_data_Text.grid(row=4, column=0, columnspan=4)
        self.result_data_Text = Text(self.init_window_name, height=30)
        self.result_data_Text.grid(row=6, column=0, columnspan=4)
        #按钮
        self.annotate_button = Button(self.init_window_name, text="Annotate", bg="lightblue",command=self.annotate)
        self.annotate_button.grid(row=1, column=3)

        self.clear_button = Button(self.init_window_name, text="Clear", bg="lightblue",
                                      command=self.clear)
        self.clear_button.grid(row=2, column=3)




    def annotate(self):
        self.result_data_Text.delete(1.0, END)
        self.log_data_Text.delete(1.0, END)
        text = self.init_data_Text.get(1.0, END).strip().replace("\n", " ")
        if text:
            try:
                res = annotate_text(str(text))
                if res:
                    self.result_data_Text.insert(1.0, res)
                    self.log_data_Text.insert(1.0, "Success")
                else:
                    self.log_data_Text.insert(1.0, "None")
            except:
                self.log_data_Text.insert(1.0, "Annotate fail")
        else:
            self.log_data_Text.insert(1.0, "Please enter valid text")
    def clear(self):
        self.init_data_Text.delete(1.0, END)
        self.result_data_Text.delete(1.0, END)
        self.log_data_Text.delete(1.0, END)


def gui_start():
    init_window = Tk()
    gui = MY_GUI(init_window)

    gui.set_init_window()

    init_window.mainloop()


gui_start()