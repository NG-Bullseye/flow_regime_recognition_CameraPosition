import telegram_send

def senden(message="Ready"):
    telegram_send.send(messages=[message])