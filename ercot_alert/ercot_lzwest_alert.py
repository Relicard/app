import time
import re
import requests
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pytz

# ----------------------------------------------------------
# CONFIGURAZIONE GENERALE
# ----------------------------------------------------------

# URL ufficiale ERCOT con i prezzi real-time per Load Zones e Hubs
ERCOT_URL = "https://www.ercot.com/content/cdr/html/hb_lz.html"

# Soglie in $/MWh
HIGH_THRESHOLD = 100  # manda mail quando LZ_WEST sale sopra questa soglia
LOW_THRESHOLD = 80    # manda mail quando LZ_WEST scende sotto questa soglia

# Ogni quanto controllare (in secondi)
CHECK_INTERVAL_SECONDS = 300  # 5 minuti

# ----------------------------------------------------------
# CONFIGURAZIONE EMAIL (ESEMPIO GMAIL)
# ----------------------------------------------------------
# ATTENZIONE: Usa preferibilmente una "App Password" di Google, non la password normale.

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # SSL

import os

SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_FROM = os.environ.get("EMAIL_FROM")
EMAIL_TO = os.environ.get("EMAIL_TO")
  

# ----------------------------------------------------------
# FUNZIONI
# ----------------------------------------------------------

def get_lz_west_price():
    """
    Scarica la pagina ERCOT e ritorna il prezzo LZ_WEST come float (in $/MWh).

    Strategia:
    - scarico l'HTML
    - rimuovo i tag
    - cerco la stringa LZ_WEST e il primo numero dopo
    """
    resp = requests.get(ERCOT_URL, timeout=10)
    resp.raise_for_status()
    html = resp.text

    # 1) togliamo tutti i TAG HTML per lavorare solo su testo
    #    <qualcosa> -> rimosso
    text = re.sub(r"<[^>]+>", " ", html)

    # 2) comprimiamo spazi multipli
    text = re.sub(r"\s+", " ", text)

    # 3) cerchiamo "LZ_WEST" seguito da un numero (anche con segno e decimali)
    #    es: "LZ_WEST 47.70" oppure "LZ_WEST -12.34"
    match = re.search(r"LZ[_ ]WEST\s+(-?\d+\.\d+)", text)

    if not match:
        # DEBUG opzionale: stampa un pezzo di testo per capire
        snippet = text[:1000]
        print("DEBUG: primi 1000 caratteri del testo pulito:")
        print(snippet)
        raise ValueError("Non trovo LZ_WEST nella pagina ERCOT (dopo pulizia HTML)")

    price = float(match.group(1))
    return price



def send_email(subject: str, body: str):
    """
    Invia una email con subject e body al destinatario EMAIL_TO.
    """
    if not (SMTP_USER and SMTP_PASSWORD and EMAIL_FROM and EMAIL_TO):
        raise RuntimeError("Config email incompleta: controlla SMTP_USER/PASSWORD/EMAIL_FROM/EMAIL_TO.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.login(SMTP_USER, SMTP_PASSWORD)
        smtp.send_message(msg)

    print(f"[MAIL] Inviata a {EMAIL_TO} con subject: {subject}")


def send_high_alert(price: float):
    """
    Mail quando il prezzo sale sopra HIGH_THRESHOLD.
    """
    subject = f"[ERCOT] LZ_WEST sopra {HIGH_THRESHOLD} $/MWh (ora {price:.2f})"
    body = (
        f"Ciao,\n\n"
        f"Il prezzo ERCOT LZ_WEST ha superato la soglia alta.\n\n"
        f"Prezzo attuale: {price:.2f} $/MWh\n"
        f"Soglia alta:   {HIGH_THRESHOLD:.2f} $/MWh\n\n"
        f"Fonte: {ERCOT_URL}\n\n"
        f"-- Alert automatico DESMO\n"
    )
    send_email(subject, body)


def send_low_alert(price: float):
    """
    Mail quando il prezzo scende sotto LOW_THRESHOLD.
    """
    subject = f"[ERCOT] LZ_WEST sotto {LOW_THRESHOLD} $/MWh (ora {price:.2f})"
    body = (
        f"Ciao,\n\n"
        f"Il prezzo ERCOT LZ_WEST è sceso sotto la soglia bassa.\n\n"
        f"Prezzo attuale: {price:.2f} $/MWh\n"
        f"Soglia bassa:  {LOW_THRESHOLD:.2f} $/MWh\n\n"
        f"Fonte: {ERCOT_URL}\n\n"
        f"-- Alert automatico DESMO\n"
    )
    send_email(subject, body)


def main():
    print("=== Avvio monitor ERCOT LZ_WEST ===")
    print(f"URL: {ERCOT_URL}")
    print(f"Soglia ALTA: {HIGH_THRESHOLD} $/MWh")
    print(f"Soglia BASSA: {LOW_THRESHOLD} $/MWh")
    print(f"Intervallo controllo: {CHECK_INTERVAL_SECONDS} secondi\n")

    # last_state può essere:
    #   "HIGH"  -> ultimo stato conosciuto: sopra 100
    #   "LOW"   -> ultimo stato conosciuto: sotto 80
    #   "MID"   -> fra 80 e 100
    #   None    -> sconosciuto (all'avvio)
    last_state = None

    while True:
        try:
            price = get_lz_west_price()
            # --- Data e ora in Texas (Midland) ---
            tz_tx = pytz.timezone("America/Chicago")
            now_tx = datetime.now(tz_tx).strftime("%Y-%m-%d %H:%M:%S")

            print("\n----------------------------------------")
            print(f"[CHECK] {now_tx} (Midland TX) → Prezzo ERCOT LZ_WEST: {price:.2f} $/MWh")
            print("----------------------------------------")

            # Logica a isteresi:
            # - se prezzo >= HIGH_THRESHOLD e non eravamo già "HIGH" -> manda mail high, imposta stato HIGH
            # - se prezzo <= LOW_THRESHOLD  e non eravamo già "LOW"  -> manda mail low, imposta stato LOW
            # - se è in mezzo -> stato MID, nessuna mail

            if price >= HIGH_THRESHOLD:
                if last_state != "HIGH":
                    print("[EVENTO] Prezzo sopra soglia alta, invio mail HIGH...")
                    send_high_alert(price)
                    last_state = "HIGH"
                else:
                    print("[STATE] Rimane in stato HIGH, nessuna nuova mail.")
            elif price <= LOW_THRESHOLD:
                if last_state != "LOW":
                    print("[EVENTO] Prezzo sotto soglia bassa, invio mail LOW...")
                    send_low_alert(price)
                    last_state = "LOW"
                else:
                    print("[STATE] Rimane in stato LOW, nessuna nuova mail.")
            else:
                # in fascia intermedia
                if last_state != "MID":
                    print("[STATE] Prezzo in fascia 80–100, entro in stato MID.")
                else:
                    print("[STATE] Resto in stato MID.")
                last_state = "MID"

        except Exception as e:
            print(f"[ERRORE] {e}")

        # Attendi prima del prossimo controllo
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
