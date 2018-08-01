import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText


def notify_training_completion():

    fromaddr = "notification.nie@gmail.com"
    toaddr = "ynie19@deerfield.edu"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Training Completed"

    body = "Please do validation"
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "notification2018")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()