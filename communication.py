#
#
# helper functions for sending notifications after
# training and validation.
# (c) Neil Nie, All Rights Reserved.
#

import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText


def notify_training_completion(save_path):

    fromaddr = "notification.nie@gmail.com"
    toaddr = "ynie19@deerfield.edu"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Steering Training Completed"

    body = "Please do validation. Saved to : " + save_path
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "notification2018")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()


def notify_validation_completion(score, model_name):

    fromaddr = "notification.nie@gmail.com"
    toaddr = "ynie19@deerfield.edu"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Steering Validation Completed"

    body = model_name + " Score : " + str(score)
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "notification2018")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()