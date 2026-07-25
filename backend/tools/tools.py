import os
import webbrowser
from datetime import datetime

def open_notepad():
    os.system("notepad")
    return "Opening Notepad"

def open_calculator():
    os.system("calc")
    return "Opening Calculator"

def open_vscode():
    os.system("code")
    return "Opening VS Code"

def open_chrome():
    webbrowser.open("https://www.google.com")
    return "Opening Chrome"

def open_explorer():
    os.system("explorer")
    return "Opening File Explorer"

def get_time():
    return datetime.now().strftime("%I:%M %p")