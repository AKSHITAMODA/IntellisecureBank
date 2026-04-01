# IntelliSecureBank 🏦🔐

A **secure, intelligent banking system** that integrates **OCR KYC, Face Verification, Encrypted Storage, and Anomaly Detection**.  Built for the **Samsung PRISM GenAI Hackathon 2025**, it ensures **robust security, user privacy, and real-time monitoring**. #SamsungPRISMGenIHackathon2025
---
## Submissions:
https://drive.google.com/file/d/1mnNFxrsImLMr9FQHi0EsuNqjoDOuNuxk/view?usp=sharing

---

## ✨ Key Features

- **🔑 Multi-Factor Admin Authentication**
  - Passwords stored with **bcrypt hashing**
  - **TOTP (Google Authenticator/Authy)** support
  - OTP delivery via **Twilio SMS** or **Email**

- **🧾 KYC Verification**
  - OCR (Tesseract) to extract name & details from Aadhaar/ID
  - Automated text matching for name validation

- **🖼 Face Verification**
  - Passport photo upload + **OpenCV DNN** live webcam matching
  - Snapshot storage for audit/compliance

- **💳 Banking Operations**
  - Create, modify, delete accounts
  - Deposit/Withdraw with validation
  - Balance enquiry & search by account/name
  - Transaction history logging in `transactions.csv`

- **🚨 Anomaly Detection**
  - Train **IsolationForest model** on transaction history
  - Flag suspicious transactions (amount/time anomalies)
  - Save flagged results in `transactions_flagged.csv`

- **📊 Data Export & Security**
  - Encrypted account storage (`accounts.csv` with AES)
  - **Excel export** with password-protection
  - Audit logging for sensitive operations

- **⚙️ Extras**
  - Role-based admin management (Admin, Teller, Auditor)
  - Tamper-evident audit log (hash chain)
  - CLI menu-driven interface (extendable to GUI)

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+  
- **Core Libraries**: OpenCV, Tesseract OCR, bcrypt, pyotp, pandas, scikit-learn, xlsxwriter, yagmail, twilio  
- **Database**: SQLite (`admins.db`, accounts/transactions)  
- **Encryption**: AES/Fernet (custom `crypto_utils.py`)

---

## 🛠 Setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python db.py create-admin admin StrongPass!123 --role Admin
python intell.py

```
## 📂 Project Structure

```text
INTELLISECUREBANK/
│── intell.py                 # Main CLI entry point
│── db.py                     # Admins DB (bcrypt + TOTP)
│── crypto_utils.py           # AES/Fernet utilities
│── accounts.csv              # Encrypted account storage
│── transactions.csv          # Transaction log
│── admins.db                 # Admin auth DB (SQLite)
│── TeamName.pdf              # Supplementary file
│── requirements.txt          # Python dependencies
│── README.md                 # This file
│
├── models/                   # Face recognition models
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── nn4.small2.v1.t7
│
├── aadhaarcards/             # Stored KYC documents
├── passport_size_photos/     # Stored passport photos
├── admin_images/             # Snapshots after verification
├── backups/                  # Auto backups (encrypted zips)
└── logs/                     # Security & transaction logs





