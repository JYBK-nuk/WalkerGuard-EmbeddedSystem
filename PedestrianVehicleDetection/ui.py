import tkinter as tk


class UI(tk.Frame):
    signal_mode: str = "行人綠燈"

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.signal_mode = tk.BooleanVar()
        self.signal_mode.set(True)  # 預設為有號誌模式

        # 有號誌模式和無號誌模式切換按鈕
        self.mode_button = tk.Checkbutton(
            self, text="號誌模式", variable=self.signal_mode, command=self.toggle_mode
        )
        self.mode_button.pack(side="top")

        # 行人紅燈綠燈切換按鈕
        self.red_button = tk.Button(self, text="紅燈", command=self.red_light)
        self.red_button.pack(side="left")

        self.green_button = tk.Button(self, text="綠燈", command=self.green_light)
        self.green_button.pack(side="right")

    def toggle_mode(self):
        if self.signal_mode.get():  # 如果是有號誌模式
            self.red_button.config(state="normal")
            self.green_button.config(state="normal")
        else:  # 如果是無號誌模式
            self.red_button.config(state="disabled")
            self.green_button.config(state="disabled")

    def red_light(self):
        self.signal_mode = "行人紅燈"
        print("行人紅燈")

    def green_light(self):
        self.signal_mode = "行人綠燈"
        print("行人綠燈")


