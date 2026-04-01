import csv
import os
import pathlib
import pytesseract
import cv2
from PIL import Image
import difflib
from datetime import datetime
import getpass
import pandas as pd
from sklearn.ensemble import IsolationForest
from termcolor import colored
import pyotp
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
import yagmail
from crypto_utils import decrypt_text, load_key
import random
import numpy as np
import shutil

from pathlib import Path
import sys
import bcrypt
from collections import defaultdict
from datetime import timedelta
import pyotp
import qrcode
# ====== PDF Statement Export (ReportLab) ======
import os, csv, json
from datetime import datetime, timedelta, date
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
try:
    ROOT  = Path(__file__).resolve().parent  # base folder of intell.py
except NameError:
    ROOT  = Path.cwd()

STATEMENTS_DIR = ROOT / "statements"
TXN_CSV = ROOT / "transactions.csv"
TOTP_DIR = "totp_secrets"
os.makedirs(TOTP_DIR, exist_ok=True)

def get_totp_secret(username):
    """
    Load or create a TOTP secret for a given user.
    If new, prints the otpauth:// URI and shows a QR code.
    """
    path = os.path.join(TOTP_DIR, f"{username}.txt")

    if not os.path.exists(path):
        # generate a new secret
        secret = pyotp.random_base32()
        with open(path, "w") as f:
            f.write(secret)

        uri = f"otpauth://totp/ColonyBank:{username}?secret={secret}&issuer=ColonyBank"
        print(f"🔐 New TOTP secret for {username}: {secret}")
        print(f"👉 Add manually or scan QR with Google Authenticator:")
        print(uri)

        # optional QR code image
        img = qrcode.make(uri)
        qr_path = os.path.join(TOTP_DIR, f"{username}_qr.png")
        img.save(qr_path)
        print(f"📱 QR code saved to {qr_path}")

    else:
        with open(path) as f:
            secret = f.read().strip()

    return secret

# in-memory counters for this run
FAILED_LOGINS = defaultdict(int)   # username -> count
LOCKED_UNTIL = {}                  # username -> datetime


CREDENTIALS_FILE = "admin_credentials.txt" # Automatically create admin_images folder if it doesn't exist if not os.path.exists("admin_images"): os.makedirs("admin_images")
# --- Model paths (absolute, robust) ---
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PROTOTXT   = MODEL_DIR / "deploy.prototxt"
CAFFEMODEL = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# OpenFace file can be named either of these; pick whichever exists
OPENFACE_1 = MODEL_DIR / "openface_nn4.small2.v1.t7"
OPENFACE_2 = MODEL_DIR / "nn4.small2.v1.t7"

def _pick_openface():
    if OPENFACE_1.exists():
        return OPENFACE_1
    if OPENFACE_2.exists():
        return OPENFACE_2
    raise FileNotFoundError(
        "OpenFace .t7 model not found. Expected one of:\n"
        f" - {OPENFACE_1}\n - {OPENFACE_2}"
    )

import cv2
import os

import cv2, os, sys
from pathlib import Path

def load_face_models():
    # Always resolve relative to THIS file, not the shell's cwd
    script_dir = Path(__file__).resolve().parent
    base_path  = script_dir / "models"

    prototxt    = base_path / "deploy.prototxt"
    caffemodel  = base_path / "res10_300x300_ssd_iter_140000.caffemodel"
    embedderpth = base_path / "nn4.small2.v1.t7"

    missing = [p for p in [prototxt, caffemodel, embedderpth] if not p.exists()]
    if missing:
        # Helpful diagnostics so you can see what's going on
        try:
            contents = "\n".join(f" - {p.name}" for p in sorted(base_path.iterdir()))
        except Exception:
            contents = "(could not list directory)"
        raise FileNotFoundError(
            "Missing model files:\n"
            + "\n".join(f" - {p}" for p in missing)
            + f"\n\nLooked in: {base_path}\nContents there were:\n{contents}"
        )

    # Load models
    face_detector = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    embedder      = cv2.dnn.readNetFromTorch(str(embedderpth))
    return face_detector, embedder

# ---------- FACE RECOGNITION UTILS ----------


def get_face_embedding(image, face_detector, embedder, conf_thresh=0.5):
    """
    Returns a 128D embedding for the most confident detected face, else None.
    """
    if image is None or image.size == 0:
        return None

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    if detections is None or detections.ndim < 3 or detections.shape[2] == 0:
        return None

    # Pick most confident box above threshold
    best = (-1.0, None)  # (conf, (x1,y1,x2,y2))
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h], dtype=float)
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                # too tiny; likely a false detection
                continue

            if conf > best[0]:
                best = (conf, (x1, y1, x2, y2))

    if best[1] is None:
        return None

    x1, y1, x2, y2 = best[1]
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face_blob = cv2.dnn.blobFromImage(
        cv2.resize(face, (96, 96)),
        1.0 / 255.0, (96, 96), (0, 0, 0),
        swapRB=True, crop=False
    )
    embedder.setInput(face_blob)
    vec = embedder.forward()
    return vec.flatten() if vec is not None else None

def verify_face(ref_img_path):
    face_detector = cv2.dnn.readNetFromCaffe("models/deploy.prototxt",
                                             "models/res10_300x300_ssd_iter_140000.caffemodel")
    embedder = cv2.dnn.readNetFromTorch("models/openface_nn4.small2.v1.t7")

    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        print("❌ Could not load reference image.")
        return False

    ref_embedding = get_face_embedding(ref_img, face_detector, embedder)
    if ref_embedding is None:
        print("❌ No face found in reference image.")
        return False

    print("📸 Please look at the camera. Verifying face... (press 'q' to cancel)")

    cap = cv2.VideoCapture(0)
    matched = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_embedding = get_face_embedding(frame, face_detector, embedder)

        if frame_embedding is not None:
            distance = np.linalg.norm(ref_embedding - frame_embedding)
            print(f"🔍 Similarity Score: {distance:.2f}")

            if distance < 0.6:
                print("✅ Face match successful!")
                matched = True
                break
            else:
                print("❌ Face mismatch. Try again or press 'q' to quit.")

        cv2.imshow("🔍 Face Verification", frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            print("❌ Face verification aborted.")
            matched = False
            break

    cap.release()
    cv2.destroyAllWindows()
    return matched

# ---------- Admin Registration & Login ----------
# ------------ REGISTER ------------
from crypto_utils import encrypt_text, load_key
def register():
    print("🆕 Register New Admin Credentials")
    username = input("Create a Username: ").strip()
    password = input("Create a Password: ").strip()

    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt).decode()

    key = load_key()
    encrypted = encrypt_text(f"{username},{hashed}", key)

    if not os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'w') as f:
            pass

    with open(CREDENTIALS_FILE, 'a') as f:
        f.write(encrypted + "\n")

    print("✅ Registration Complete (Face verification will be done during account creation)\n")


def login():
    # --- tiny helpers kept inside the function so you can paste this as-is ---
    def _is_bcrypt_hash(s: str) -> bool:
        return isinstance(s, str) and (s.startswith("$2a$") or s.startswith("$2b$") or s.startswith("$2y$"))

    def _rewrite_user_to_bcrypt(username: str, new_bcrypt_hash: str):
        """
        Rewrites the line for `username` in CREDENTIALS_FILE to store bcrypt hash.
        Keeps other users intact. Uses your existing encrypt_text/decrypt_text.
        """
        key_local = load_key()
        # read all encrypted lines
        with open(CREDENTIALS_FILE, "r") as fin:
            enc_lines = [ln.rstrip("\n") for ln in fin.readlines()]

        new_lines = []
        replaced = False
        for enc in enc_lines:
            if not enc:
                new_lines.append(enc)
                continue
            try:
                dec = decrypt_text(enc, key_local)
                parts = [p.strip() for p in dec.split(",")]
                # support extended formats like username,secret,role,...
                if len(parts) >= 2:
                    u = parts[0]
                    if u == username and not replaced:
                        # preserve any trailing fields beyond secret (e.g., role)
                        trailing = parts[2:] if len(parts) > 2 else []
                        dec_new = ",".join([username, new_bcrypt_hash] + trailing)
                        enc_new = encrypt_text(dec_new, key_local)
                        new_lines.append(enc_new)
                        replaced = True
                    else:
                        new_lines.append(enc)
                else:
                    new_lines.append(enc)
            except Exception:
                # if a line can't be decrypted, keep it as-is
                new_lines.append(enc)

        if not replaced:
            # if the user didn't exist (unexpected), append a new line
            enc_new = encrypt_text(f"{username},{new_bcrypt_hash}", key_local)
            new_lines.append(enc_new)

        with open(CREDENTIALS_FILE, "w") as fout:
            for ln in new_lines:
                fout.write(ln + "\n")

    # --- ensure lockout globals exist (safe even if already defined) ---
    global FAILED_LOGINS, LOCKED_UNTIL
    try:
        FAILED_LOGINS
    except NameError:
        from collections import defaultdict
        FAILED_LOGINS = defaultdict(int)
    try:
        LOCKED_UNTIL
    except NameError:
        LOCKED_UNTIL = {}
    from datetime import timedelta

    if not os.path.exists(CREDENTIALS_FILE):
        print("⚠ No credentials file found. Please run registration first.\n")
        return False

    with open(CREDENTIALS_FILE, 'r') as f:
        if len(f.readlines()) == 0:
            print("⚠ No admin registered yet. Please register first.\n")
            return False

    key = load_key()

    while True:  # 🔁 Full retry loop
        print("🔐 Login with Existing Admin Credentials")
        input_username = input("Username: ").strip()
        input_password = input("Password: ").strip()

        # ⛔ lockout window
        if input_username in LOCKED_UNTIL and datetime.now() < LOCKED_UNTIL[input_username]:
            until = LOCKED_UNTIL[input_username].strftime("%H:%M:%S")
            print(colored(f"⛔ Account locked. Try again after {until}", "red"))
            continue

        match_found = False
        password_ok = False
        stored_line_had_plaintext = False  # detect legacy format for upgrade

        # Re-open to iterate lines fresh
        with open(CREDENTIALS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    decrypted = decrypt_text(line, key)
                    parts = [p.strip() for p in decrypted.split(',')]
                    if len(parts) < 2:
                        continue

                    stored_username = parts[0]
                    stored_secret   = parts[1]   # bcrypt hash OR legacy plaintext
                    # optional extra fields (role, etc.) -> parts[2:]

                    if input_username != stored_username:
                        continue

                    # we found the user record
                    match_found = True

                    if _is_bcrypt_hash(stored_secret):
                        # new scheme
                        try:
                            import bcrypt  # ensure available
                            password_ok = bcrypt.checkpw(input_password.encode(), stored_secret.encode())
                        except Exception:
                            password_ok = False
                    else:
                        # legacy plaintext compare
                        password_ok = (input_password == stored_secret)
                        stored_line_had_plaintext = True

                    if password_ok:
                        break  # stop scanning lines for this attempt
                except Exception:
                    # skip undecodable lines silently
                    continue

        if not match_found or not password_ok:
            # count only password failures (not OTP failures)
            FAILED_LOGINS[input_username] += 1
            print("❌ Login failed.")
            if FAILED_LOGINS[input_username] >= 5:
                LOCKED_UNTIL[input_username] = datetime.now() + timedelta(minutes=5)
                print(colored("⛔ Too many attempts. Locked for 5 minutes.", "red"))
            else:
                left = 5 - FAILED_LOGINS[input_username]
                print(colored(f"Attempts left: {left}", "yellow"))
            continue

        # Auto-upgrade legacy plaintext to bcrypt after successful password
        if stored_line_had_plaintext:
            try:
                import bcrypt
                salt = bcrypt.gensalt()
                new_hash = bcrypt.hashpw(input_password.encode(), salt).decode()
                _rewrite_user_to_bcrypt(input_username, new_hash)
                print(colored("🔄 Upgraded your password storage to bcrypt.", "cyan"))
            except Exception as e:
                print(colored(f"⚠ Could not upgrade to bcrypt automatically: {e}", "yellow"))

        print(colored("🟢 Password correct!", "green"))

        # === OTP VERIFICATION ===
        print("\n🔐 Choose OTP Delivery Method:")
        print("1. Phone Number (SMS via Twilio)")
        print("2. Email (SMTP)")
        print("3. Show on screen (Test Mode)")
        print("4. Authenticator App (TOTP)")  # will work only if you added get_totp_secret()
        otp_method = input("Enter choice (1/2/3/4): ").strip()

        if otp_method == '4':
            # TOTP path: only if you implemented get_totp_secret() and have pyotp
            try:
                import pyotp  # requires pyotp installed
                secret = get_totp_secret(input_username)  # you must have defined this elsewhere
                totp = pyotp.TOTP(secret)
                entered_otp = input("🔑 Enter 6-digit code from your Authenticator: ").strip()
                if totp.verify(entered_otp, valid_window=1):
                    print(colored("✅ Successfully Logged In. Welcome to Colony Bank!\n", "green"))
                    FAILED_LOGINS[input_username] = 0
                    return True
                else:
                    print(colored("❌ Invalid TOTP. Restarting login...\n", "red"))
                    continue
            except Exception as e:
                print(colored(f"⚠ TOTP not available: {e}", "yellow"))
                print("Falling back to random OTP methods.\n")

        # Random OTP (1/2/3)
        otp = str(random.randint(100000, 999999))

        if otp_method == '1':
            phone = input("Enter phone number with country code (e.g. +91XXXXXXXXXX): ").strip()
            send_sms_otp(phone, otp)
        elif otp_method == '2':
            email = input("Enter your email address: ").strip()
            send_email_otp(email, otp)
        else:
            print(colored(f"📲 Your OTP is: {otp}", "yellow"))

        entered_otp = input("🔑 Enter the OTP: ").strip()

        if entered_otp == otp:
            print(colored("✅ Successfully Logged In. Welcome to Colony Bank!\n", "green"))
            FAILED_LOGINS[input_username] = 0
            return True
        else:
            print(colored("❌ Invalid OTP. Restarting login...\n", "red"))
            # Note: we do NOT count OTP failure towards lockout; only password failures.
            continue

# ------------ MAIN CALL ------------
def register_and_login():
    print("🔐 Admin Access Required")
    while True:
        choice = input("Do you want to (1) Login or (2) Register new admin? Enter 1 or 2: ").strip()
        if choice == '1':
            return login()
        elif choice == '2':
            register()
            return login()
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")

# ------------ SMS OTP ------------

def send_sms_otp(phone_number, otp):
    import os

    account_sid = os.getenv("TWILIO_SID")
    auth_token = os.getenv("TWILIO_AUTH")     # Your Twilio Auth Token
    from_number = '+1 920 709 9126'                       # Your Twilio verified sender number

    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=f"Your Colony Bank OTP is: {otp}",
            from_=from_number,
            to=phone_number
        )
        print(colored(f"📨 OTP sent to {phone_number}. SID: {message.sid}", "cyan"))
    except Exception as e:
        print(colored(f"❌ Failed to send SMS: {e}", "red"))
# ------------ EMAIL OTP ------------
def send_email_otp(receiver_email, otp):
    try:
        yag = yagmail.SMTP(user='akshitamoda2006@gmail.com', password='bddb qvnw qobx qemb')
        yag.send(to=receiver_email, subject="Colony Bank OTP Verification", contents=f"Your OTP is: {otp}")
        print(colored(f"📧 OTP sent to {receiver_email}", "cyan"))
    except Exception as e:
        print(colored(f"❌ Email send failed: {e}", "red"))

# ---------- Account Class ----------
class Account:
    def __init__(self):
        self.accNo = 0
        self.name = ''
        self.deposit = 0
        self.type = ''

    def extract_name_from_id(self, image_path):
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            print("\n🧾 Extracted Text from ID:\n")
            print(text)
            return text
        except Exception as e:
            print("OCR failed:", e)
            return ""

    def createAccount(self):
        try:
            self.accNo = int(input("Enter the account number: "))
            id_path = input("Enter path to your ID image: ").strip()
            id_text = self.extract_name_from_id(id_path)

            self.name = input("Enter the account holder's name (must match ID): ").strip()

            # === OCR Name Matching ===
            lines = [line.strip() for line in id_text.splitlines() if line.strip()]
            name_candidates = [line for line in lines if any(w[0].isupper() for w in line.split())]

            best_match = None
            best_score = 0.0
            for candidate in name_candidates:
                score = difflib.SequenceMatcher(None, self.name.lower(), candidate.lower()).ratio()
                if score > best_score:
                   best_score = score
                   best_match = candidate

            if best_score < 0.6:
                print("❌ Name does not match any name-like text from ID.")
                print(f"Best Match Found: '{best_match}' (Similarity: {best_score:.2f})")
                print("📝 OCR-extracted name-like candidates:")
                for cand in name_candidates:
                    print("   -", cand)
                self.name = ''
                return
            else:
                print(f"✅ Name matched with: '{best_match}' (Similarity: {best_score:.2f})")

            # ======= Save Aadhaar card image =======
            aadhaar_folder = "aadhaarcards"
            os.makedirs(aadhaar_folder, exist_ok=True)
            aadhaar_save_path = os.path.join(aadhaar_folder, f"{self.name.replace(' ', '_')}_aadhaar.jpg")
            shutil.copy(id_path, aadhaar_save_path)
            print(f"🧾 Aadhaar card saved to: {aadhaar_save_path}")

             # ======= FACE RECOGNITION STEP 1: Upload Passport Photo =======
            print("\n🖼 Upload a passport-size photo for face verification.")
            passport_path = input("Enter path to your passport-size photo: ").strip()
     
            # Save passport photo
            passport_folder = "passport_size_photos"
            os.makedirs(passport_folder, exist_ok=True)
            passport_save_path = os.path.join(passport_folder, f"{self.name.replace(' ', '_')}_passport.jpg")
            shutil.copy(passport_path, passport_save_path)
            print(f"🖼 Passport photo saved to: {passport_save_path}")

            # Load models
            # Load models (robust absolute paths + sanity checks)
            face_detector, embedder = load_face_models()

# IMPORTANT: read the file we just saved (ensures consistent path)
            passport_img = cv2.imread(passport_save_path)
            passport_embedding = get_face_embedding(passport_img, face_detector, embedder)
            if passport_embedding is None:
                print("❌ No face detected in passport image.")
                return

            # Create window and move to top-left
            #import pygetwindow as gw
            #import time
            #import win32gui
            #import win32con
            # Create OpenCV window and move it
            print("\n📸 Now looking for a live face match... (Press 'q' to quit)")

# Windows tends to behave better with CAP_DSHOW
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)

# Warm-up a few frames 
            for _ in range(5):
                cap.read()

            match_found = False

# Try to create & move the preview window
            try:
                import time
                cv2.namedWindow("Live Face Verification", cv2.WINDOW_NORMAL)
                cv2.moveWindow("Live Face Verification", 0, 0)
                time.sleep(0.2)  # let it initialize
                try:
                    import pygetwindow as gw, win32gui, win32con
                    win = gw.getWindowsWithTitle("Live Face Verification")[0]
                    hwnd = win._hWnd
                    win32gui.SetWindowPos(
                        hwnd, win32con.HWND_TOPMOST, 0, 0, 640, 480,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
                    )
                except Exception as e:
                    print("⚠️ Could not force window to top:", e)
            except Exception:
                pass  # if no GUI libs available, continue without window adjustment

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_embedding = get_face_embedding(frame, face_detector, embedder)
                if frame_embedding is not None:
                    distance = np.linalg.norm(passport_embedding - frame_embedding)
                    cv2.putText(frame, f"Distance: {distance:.2f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if distance < 0.6:
                        print("✅ Live face verification successful!")
                        match_found = True
                       # Save snapshot
                        if not os.path.exists("admin_images"):
                           os.makedirs("admin_images")
                        sanitized_name = self.name.replace(" ", "_")
                        snapshot_path = os.path.join("admin_images", f"{sanitized_name}.jpg")
                        cv2.imwrite(snapshot_path, frame)
                        print(f"📸 Snapshot saved at: {snapshot_path}")
                        break

                cv2.imshow("Live Face Verification", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("❌ Verification cancelled.")
                    break

            cap.release()
            cv2.destroyAllWindows()

            if not match_found:
                print("❌ Live face does not match passport photo.")
                return
        # ======= ACCOUNT TYPE AND DEPOSIT =======
            while True:
                self.type = input("Enter the account type [C/S]: ").strip().upper()
                if self.type in ['C', 'S']:
                   break
                print("Invalid type. Enter 'C' for Current or 'S' for Saving.")

            while True:
                self.deposit = int(input("Enter initial amount (>=500 for Saving, >=1000 for Current): "))
                if (self.type == 'S' and self.deposit >= 500) or (self.type == 'C' and self.deposit >= 1000):
                    break
                print("Insufficient initial amount. Try again.")

            print("\n✅ Account Created Successfully!\n")

        except ValueError:
            print("❌ Invalid input. Try again.")


    def showAccount(self):
        print(f"Account Number     : {self.accNo}")
        print(f"Account Holder     : {self.name}")
        print(f"Account Type       : {self.type}")
        print(f"Account Balance    : ₹{self.deposit}")

    def modifyAccount(self):
        print("✏ Modify Account:")
        self.name = input("New Account Holder Name: ").strip()
        while True:
            self.type = input("New Account Type [C/S]: ").strip().upper()
            if self.type in ['C', 'S']:
                break
            print("Invalid account type.")
        self.deposit = int(input("New Balance: "))
        print("✅ Account modified.")

    def depositAmount(self, amount):
        self.deposit += amount

    def withdrawAmount(self, amount):
        if amount <= self.deposit:
            self.deposit -= amount
            return True
        return False

    def report(self):
        print(f"{self.accNo:<10} {self.name:<20} {self.type:<10} ₹{self.deposit:<10}")

    def getAccountNo(self):
        return self.accNo


# ---------- Transactions ----------
def log_transaction(account_no, txn_type, amount):
    with open("transactions.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            account_no,
            txn_type,
            amount
        ])


def initialize_transaction_log():
    if not os.path.exists("transactions.csv"):
        with open("transactions.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "AccountNo", "TransactionType", "Amount"])


def displayTransactions():
    if not os.path.exists("transactions.csv"):
        print("📂 No transaction history found.")
        return

    with open("transactions.csv", 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) <= 1:
        print("📂 No transactions logged yet.")
        return

    print("\n💳 Transaction History:")
    print(f"{'Timestamp':<20} {'Acc No':<10} {'Type':<10} {'Amount':<10}")
    print("-" * 60)
    for row in rows[1:]:
        timestamp, acc_no, txn_type, amount = row
        print(f"{timestamp:<20} {acc_no:<10} {txn_type:<10} ₹{amount:<10}")

#---Train Anamoly Model---
def train_anomaly_model():
    if not os.path.exists("transactions.csv"):
        print("⚠ transactions.csv not found.")
        return None

    df = pd.read_csv("transactions.csv")
    if df.shape[0] < 10:
        print("⚠ Not enough data to train model.")
        return None

    df['Amount'] = df['Amount'].astype(float)
    df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    df['is_withdraw'] = df['TransactionType'].apply(lambda x: 1 if x == 'withdraw' else 0)

    X = df[['Amount', 'hour', 'is_withdraw']]

    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X)

    df['anomaly'] = model.predict(X)
    df.to_csv("transactions_flagged.csv", index=False)
    print("✅ Anomaly detection complete. Results saved to transactions_flagged.csv")

#----View Flagged Transaction----
def view_flagged_transactions():
    if not os.path.exists("transactions_flagged.csv"):
        print("⚠ No flagged transaction file found. Train the model first.")
        return

    df = pd.read_csv("transactions_flagged.csv")

    print("\n🚨 Flagged Transactions (Anomalies in Red):")
    print(f"{'Timestamp':<20} {'Acc No':<10} {'Type':<10} {'Amount':<10} {'Flag':<10}")
    print("-" * 70)

    for _, row in df.iterrows():
        output = f"{row['Timestamp']:<20} {row['AccountNo']:<10} {row['TransactionType']:<10} ₹{row['Amount']:<10}"
        if row['anomaly'] == -1:
            print(colored(output + " 🚨", 'red'))
        else:
            print(output)

# ---------- File Handling ----------
import pathlib

def readAccountsCSV():
    accounts = []
    if not pathlib.Path("accounts.csv").exists():
        return accounts

    key = load_key()

    with open("accounts.csv", 'r') as csvfile:
        for line in csvfile:
            line = line.strip()
            if not line:
                continue
            try:
                decrypted = decrypt_text(line, key)
                accNo, name, acc_type, deposit = decrypted.split(',')

                acc = Account()
                acc.accNo = int(accNo)
                acc.name = name
                acc.type = acc_type
                acc.deposit = int(deposit)
                accounts.append(acc)
            except Exception as e:
                print("❌ Failed to decrypt account line:", e)

    return accounts

from crypto_utils import encrypt_text, load_key

def writeAccountsCSV(accounts):
    key = load_key()
    with open("accounts.csv", 'w', newline='') as csvfile:
        for acc in accounts:
            row_data = f"{acc.accNo},{acc.name},{acc.type},{acc.deposit}"
            encrypted = encrypt_text(row_data, key)
            csvfile.write(encrypted + "\n")



# ---------- Functional Operations ----------
def displayAll():
    accounts = readAccountsCSV()
    if accounts:
        print("\n📄 All Account Holders:")
        print(f"{'Acc No':<10} {'Name':<20} {'Type':<10} {'Balance':<10}")
        print("-" * 50)
        for acc in accounts:
            acc.report()
    else:
        print("📂 No records found.")


def displaySp(num):
    accounts = readAccountsCSV()
    for acc in accounts:
        if acc.getAccountNo() == num:
            print("🔎 Account Found:")
            acc.showAccount()
            return
    print("❌ No account found with that number.")


def searchByName(name_query):
    accounts = readAccountsCSV()
    matches = [acc for acc in accounts if name_query.lower() in acc.name.lower()]
    if matches:
        print(f"\n🔍 Found {len(matches)} matching result(s):")
        for acc in matches:
            acc.showAccount()
            print("-" * 40)
    else:
        print("❌ No matching names found.")


def depositOrWithdraw(num, mode):
    accounts = readAccountsCSV()
    for acc in accounts:
        if acc.getAccountNo() == num:
            if mode == "deposit":
                amount = int(input("Enter amount to deposit: "))
                acc.depositAmount(amount)
                log_transaction(acc.getAccountNo(), "deposit", amount)
                print("✅ Amount Deposited.")
            elif mode == "withdraw":
                amount = int(input("Enter amount to withdraw: "))
                if acc.withdrawAmount(amount):
                    log_transaction(acc.getAccountNo(), "withdraw", amount)
                    print("✅ Amount Withdrawn.")
                else:
                    print("❌ Insufficient balance.")
            writeAccountsCSV(accounts)
            return
    print("❌ Account not found.")


def deleteAccount(num):
    accounts = readAccountsCSV()
    accounts = [acc for acc in accounts if acc.getAccountNo() != num]
    writeAccountsCSV(accounts)
    print("🗑 Account deleted (if it existed).")


def modifyAccount(num):
    accounts = readAccountsCSV()
    for acc in accounts:
        if acc.getAccountNo() == num:
            acc.modifyAccount()
            writeAccountsCSV(accounts)
            return
    print("❌ Account not found.")

def displayAccountTransactions(acc_no):
    if not os.path.exists("transactions.csv"):
        print("📂 No transaction history found.")
        return

    with open("transactions.csv", 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) <= 1:
        print("📂 No transactions logged yet.")
        return

    print(f"\n📜 Transaction History for Account {acc_no}:")
    print(f"{'Timestamp':<20} {'Type':<10} {'Amount':<10}")
    print("-" * 50)

    for row in rows[1:]:
        timestamp, acc, txn_type, amount = row
        if int(acc) == acc_no:
            print(f"{timestamp:<20} {txn_type:<10} ₹{amount:<10}")

#def clear_transaction_log():
 #   with open("transactions.csv", 'w', newline='') as f:
  #      writer = csv.writer(f)
   #     writer.writerow(["Timestamp", "AccountNo", "TransactionType", "Amount"])
   # print("🧹 Transaction log cleared.")

#--Export accounts in excel--
import pandas as pd
import xlsxwriter
from crypto_utils import decrypt_text, load_key

import pandas as pd
import os
import win32com.client
from crypto_utils import decrypt_text, load_key

def encrypt_excel_file(original_file_path, password, output_file_path):
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    abs_input = os.path.abspath(original_file_path)
    abs_output = os.path.abspath(output_file_path)

    workbook = excel.Workbooks.Open(abs_input)

    # Save as new file with OPEN password protection
    workbook.SaveAs(
        abs_output,
        FileFormat=51,       # xlOpenXMLWorkbook (.xlsx)
        Password=password,   # 🔐 Require password to open
        AccessMode=1         # xlNoChange
    )

    workbook.Close(False)
    excel.Quit()

def exportAccountsToExcel():
    input_file = "accounts.csv"
    temp_file = "decrypted_accounts.xlsx"
    final_file = "decrypted_accounts_protected.xlsx"
    protection_password = input("🔑 Set a password to protect the Excel file (required to open it): ").strip()

    if not os.path.exists(input_file):
        print("❌ accounts.csv not found.")
        return

    key = load_key()
    data = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                decrypted = decrypt_text(line, key)
                accNo, name, acc_type, deposit = decrypted.split(',')
                data.append({
                    'AccountNo': int(accNo),
                    'Name': name,
                    'Type': acc_type,
                    'Balance': int(deposit)
                })
            except Exception as e:
                print("⚠ Failed to decrypt a row:", e)

    if not data:
        print("⚠ No valid account records to export.")
        return

    # Export to unprotected Excel file
    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(temp_file, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Accounts')
    writer.close()

    print("🔐 Encrypting Excel file...")

    try:
        encrypt_excel_file(temp_file, protection_password, final_file)
        os.remove(temp_file)  # Delete the unprotected version
        print(f"✅ Encrypted Excel file saved as: {final_file}")
        print("📂 File now requires password to open.")
    except Exception as e:
        print("❌ Failed to apply file-level password protection:", e)

# ---------- Main ----------
def intro():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("\t\t✨ Welcome to Colony Bank ✨")
    print("\t    Secure. Simple. Smart Banking.")
    print("=" * 60)

# ====== PDF Statement Export helpers ======
def _register_unicode_font():
    try:
        for candidate in [
            ROOT / "assets" / "fonts" / "DejaVuSans.ttf",
            ROOT / "assets" / "fonts" / "NotoSans-Regular.ttf",
        ]:
            if candidate.exists():
                pdfmetrics.registerFont(TTFont("AppSans", str(candidate)))
                return "AppSans"
    except Exception:
        pass
    return "Helvetica"

_PDF_FONT = _register_unicode_font()

def _parse_date(d: str) -> datetime:
    d = d.strip()
    if len(d) == 7:  # YYYY-MM
        return datetime.strptime(d + "-01", "%Y-%m-%d")
    return datetime.strptime(d, "%Y-%m-%d")

def _month_range(year: int, month: int):
    start = datetime(year, month, 1)
    if month == 12:
        nextm = datetime(year + 1, 1, 1)
    else:
        nextm = datetime(year, month + 1, 1)
    end = nextm - timedelta(seconds=1)
    return start, end

def _get_account_meta(acc_no: int):
    try:
        accts = readAccountsCSV()
        for a in accts:
            if a.accNo == acc_no:
                return a.name, ("Savings" if a.type.upper() == "S" else "Current")
    except Exception:
        pass
    return "Unknown", "Unknown"

def _load_txn_df() -> pd.DataFrame:
    if not TXN_CSV.exists():
        return pd.DataFrame(columns=["Timestamp", "AccountNo", "TransactionType", "Amount"])
    df = pd.read_csv(TXN_CSV)
    if df.empty:
        return df
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["AccountNo"] = df["AccountNo"].astype(int)
    df["TransactionType"] = df["TransactionType"].astype(str)
    df["Amount"] = df["Amount"].astype(float)
    return df

def _opening_balance(acc_no: int, start_dt: datetime) -> float:
    df = _load_txn_df()
    if df.empty:
        return 0.0
    df = df[df["AccountNo"] == acc_no]
    if df.empty:
        return 0.0
    before = df[df["Timestamp"] < start_dt].sort_values("Timestamp")
    bal = 0.0
    for _, r in before.iterrows():
        t = r["TransactionType"].lower()
        amt = float(r["Amount"])
        if t in ("create", "deposit"):
            bal += amt
        elif t in ("withdraw",):
            bal -= amt
    return bal

def _format_money(x: float) -> str:
    sign = "₹" if _PDF_FONT != "Helvetica" else "Rs "
    return f"{sign}{x:,.2f}"

def export_statement_pdf(
    acc_no: int,
    start_date: str,
    end_date: str | None = None,
    out_dir: str | Path | None = None
) -> str:
    # Resolve and create absolute output dir
    base_out_dir = Path(out_dir) if out_dir else STATEMENTS_DIR
    if not base_out_dir.is_absolute():
        base_out_dir = ROOT / base_out_dir
    base_out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = _parse_date(start_date)
    if end_date is None and len(start_date.strip()) == 7:
        y, m = start_dt.year, start_dt.month
        start_dt, end_dt = _month_range(y, m)
    else:
        end_dt = _parse_date(end_date) if end_date else start_dt
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

    df = _load_txn_df()
    df_acc = df[df["AccountNo"] == acc_no].copy()

    holder_name, acc_type_human = _get_account_meta(acc_no)
    opening = _opening_balance(acc_no, start_dt)

    period = df_acc[(df_acc["Timestamp"] >= start_dt) & (df_acc["Timestamp"] <= end_dt)]
    period = period.sort_values("Timestamp")

    rows = []
    running = opening
    total_debit = 0.0
    total_credit = 0.0

    for i, r in enumerate(period.itertuples(index=False), start=1):
        ttype = r.TransactionType.lower()
        ts_str = r.Timestamp.strftime("%Y-%m-%d %H:%M")
        amt = float(r.Amount)

        debit = credit = ""
        if ttype in ("withdraw",):
            running -= amt
            debit = _format_money(amt)
            total_debit += amt
        elif ttype in ("deposit", "create"):
            running += amt
            credit = _format_money(amt)
            total_credit += amt

        rows.append([i, ts_str, ttype.capitalize(), debit, credit, _format_money(running)])

    closing = running

    fname = f"{acc_no}_{start_dt.strftime('%Y%m%d')}_to_{end_dt.strftime('%Y%m%d')}.pdf"
    out_path = base_out_dir / fname

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18*mm,
        rightMargin=18*mm,
        topMargin=16*mm,
        bottomMargin=16*mm
    )
    styles = getSampleStyleSheet()
    H = ParagraphStyle(name="H", parent=styles["Heading1"], fontName=_PDF_FONT, fontSize=16, leading=20, spaceAfter=6)
    B = ParagraphStyle(name="B", parent=styles["BodyText"], fontName=_PDF_FONT, fontSize=10, leading=13)
    S = ParagraphStyle(name="S", parent=styles["BodyText"], fontName=_PDF_FONT, fontSize=9, textColor=colors.grey)

    story = []
    story.append(Paragraph("Colony Bank — Account Statement", H))
    story.append(Paragraph(f"Account No: <b>{acc_no}</b> &nbsp;&nbsp; Holder: <b>{holder_name}</b> &nbsp;&nbsp; Type: <b>{acc_type_human}</b>", B))
    story.append(Paragraph(f"Period: <b>{start_dt.strftime('%Y-%m-%d')}</b> to <b>{end_dt.strftime('%Y-%m-%d')}</b>", B))
    story.append(Spacer(1, 8))

    story.append(Paragraph(f"Opening Balance: <b>{_format_money(opening)}</b>", B))
    if period.empty:
        story.append(Paragraph("No transactions found in this period.", B))
    story.append(Spacer(1, 6))

    data = [["#", "Date/Time", "Type", "Debit", "Credit", "Balance"]]
    data.extend(rows if rows else [["—", "—", "—", "—", "—", _format_money(opening)]])

    table = Table(data, colWidths=[14*mm, 34*mm, 22*mm, 26*mm, 26*mm, 28*mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), _PDF_FONT),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8EEF7")),
        ("ALIGN", (0,0), (0,-1), "RIGHT"),
        ("ALIGN", (3,1), (5,-1), "RIGHT"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(f"Total Debits: <b>{_format_money(total_debit)}</b> &nbsp;&nbsp; "
                           f"Total Credits: <b>{_format_money(total_credit)}</b>", B))
    story.append(Paragraph(f"Closing Balance: <b>{_format_money(closing)}</b>", B))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", S))

    doc.build(story)

    try:
        print(f"[PDF] Saved statement -> {out_path}")
    except Exception:
        pass
    return str(out_path)



def main():
    intro()
    if not register_and_login():
        #print("❌ Exiting...")
        return

    initialize_transaction_log()

    while True:
        print("\n📋 MAIN MENU")
        print("1. Open New Account")
        print("2. Deposit Amount")
        print("3. Withdraw Amount")
        print("4. Balance Enquiry")
        print("5. List All Accounts")
        print("6. Close Account")
        print("7. Modify Account")
        print("8. Exit")
        print("9. View Transaction History")
        print("10. Search Account by Name")
        print("11. Train Anomaly Detection Model")
        print("12. View Flagged Transactions")
        print("13. View Only a Specific Account’s Transactions")
        print("14. Export All Decrypted Accounts to Excel")
        print("15. Export PDF Statement")   # <-- add this line

        choice = input("Select your option (1-15): ")

        if choice == '1':
            acc = Account()
            acc.createAccount()
            if acc.name != '':
                accounts = readAccountsCSV()
                accounts.append(acc)
                writeAccountsCSV(accounts)
        elif choice == '2':
            num = int(input("Enter account number: "))
            depositOrWithdraw(num, "deposit")
        elif choice == '3':
            num = int(input("Enter account number: "))
            depositOrWithdraw(num, "withdraw")
        elif choice == '4':
            num = int(input("Enter account number: "))
            displaySp(num)
        elif choice == '5':
            displayAll()
        elif choice == '6':
            num = int(input("Enter account number to delete: "))
            deleteAccount(num)
        elif choice == '7':
            num = int(input("Enter account number to modify: "))
            modifyAccount(num)
        elif choice == '8':
            print("\n🙏 Thank you for banking with us. Goodbye!")
            break
        elif choice == '9':
            displayTransactions()
        elif choice == '10':
            name_query = input("Enter full or partial name to search: ").strip()
            searchByName(name_query)
        elif choice == '11':
            train_anomaly_model()
        elif choice == '12':
            view_flagged_transactions()
        elif choice == '13':
            acc = int(input("Enter account number to view transactions: "))
            displayAccountTransactions(acc)
        elif choice == '14':
            exportAccountsToExcel()
        elif choice == '15':
            try:
                acc = int(input("Enter account number: ").strip())
                period = input("Enter period (YYYY-MM or 'YYYY-MM-DD to YYYY-MM-DD'): ").strip()

                # Decide month vs custom range
                if "to" in period:
                    start, end = [p.strip() for p in period.split("to", 1)]
                    # Call the backend or local function depending on where you put it
                    out = export_statement_pdf(acc, start, end)   # if imported from intell.py
                    # out = backend.export_statement_pdf(acc, start, end)  # if you did: import intell as backend
                    # out = export_statement_pdf(acc, start, end)          # if function is in this same file
                else:
                    out = export_statement_pdf(acc, period)        # monthly 'YYYY-MM' or single 'YYYY-MM-DD'

                print(f"✅ Statement saved: {out}")
            except Exception as e:
                print(f"❌ Failed to export PDF statement: {e}")

        else:
            print("❌ Invalid option. Please choose between 1–15.")


    #with open(file, 'w', newline='') as f:
    #   writer = csv.writer(f)
     #   writer.writerow(["Timestamp", "AccountNo", "TransactionType", "Amount"])
      #  base_time = datetime.now()

    #    for _ in range(num):
     #       time = base_time - timedelta(minutes=random.randint(0, 10000))
      #      acc = random.choice([101, 102, 103])
       #     txn_type = random.choice(["deposit", "withdraw"])
        #    amount = random.randint(
         #       100, 50000 if txn_type == "withdraw" else 20000
          #  )

           # writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), acc, txn_type, amount])

 #print(f"✅ {num} sample transactions written to {file}")
 

if __name__ == '__main__':
    #  clear_transaction_log()  
    main()
