import smtplib
import ssl
from email.message import EmailMessage


class SMTPMail:
    def __init__(self, settings):
        self.settings = settings

    def send(self, email):
        settings = self.settings
        message = EmailMessage()
        # Recipient is configuration-owned; never route mail using model/user output.
        message["From"] = settings.smtp_from
        message["To"] = settings.support_email
        message["Subject"] = email["subject"]
        message["Message-ID"] = email["message_id"]
        message.set_content(email["body"])
        context = ssl.create_default_context()
        kwargs = {"timeout": settings.integration_timeout_seconds}
        if settings.smtp_security == "ssl":
            connection = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, context=context, **kwargs)
        else:
            connection = smtplib.SMTP(settings.smtp_host, settings.smtp_port, **kwargs)
        with connection as smtp:
            if settings.smtp_security == "starttls":
                smtp.ehlo()
                smtp.starttls(context=context)
                smtp.ehlo()
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password.get_secret_value())
            refused = smtp.send_message(
                message, from_addr=settings.smtp_from, to_addrs=[settings.support_email]
            )
            if refused:
                raise RuntimeError("SMTP recipient refused")
        return "SENT"
