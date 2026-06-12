import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_notification(subject, body, is_html=False):
    """
    Envoie une notification par email en utilisant les paramètres de config.
    """
    try:
        from config import EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO, SMTP_SERVER, SMTP_PORT
    except ImportError:
        print("⚠️ Configuration email manquante dans config.py")
        return False

    try:
        msg = MIMEMultipart("alternative") if is_html else MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO

        if is_html:
            msg.attach(MIMEText(body, "html", "utf-8"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"❌ Erreur SMTP : {e}")
        return False

def format_html_bets(bets, sport):
    """Génère le corps HTML pour les pronostics."""
    now = datetime.now().strftime('%d/%m/%Y')
    html = f"<h2>🔍 Value Bets - {now}</h2><p>Sport: {sport}</p><table border='1' style='border-collapse:collapse; width:100%'>"
    html += "<tr style='background:#4CAF50;color:white'><th>Match</th><th>Cote</th><th>EV</th><th>Mise</th></tr>"
    for b in bets:
        color = "green" if b['expected_value'] > 0 else "red"
        html += f"<tr><td>{b['match']}</td><td>{b['cote']:.2f}</td><td style='color:{color}'>{b['expected_value']:.1%}</td><td>{b['mise_conseillee']:.2f}€</td></tr>"
    html += "</table>"
    return html