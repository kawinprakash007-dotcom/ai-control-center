from datetime import datetime


def log(module, message):

    print(

        f"[{datetime.now().strftime('%H:%M:%S')}] "

        f"[{module}] "

        f"{message}"

    )