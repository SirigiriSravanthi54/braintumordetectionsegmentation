import sqlite3
from passlib.hash import pbkdf2_sha256
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, 
                  password TEXT, 
                  reset_token TEXT,
                  verification_code TEXT,
                  is_verified INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

def generate_verification_code():
    return ''.join([str(secrets.randbelow(10)) for _ in range(6)])

def send_email(to_email, subject, body):
    sender_email = "sirigirisravanthi10@gmail.com"
    sender_password = "jltt cqwz cmck lhmc"  # Your App Password
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = to_email
    message["Subject"] = subject
    
    message.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
            return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"Authentication failed: {e}")
        return False
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def add_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        # Check if email already exists
        c.execute("SELECT email FROM users WHERE email=?", (email,))
        if c.fetchone() is not None:
            conn.close()
            return False, "Email already exists"
        
        # If email doesn't exist, proceed with user creation
        verification_code = generate_verification_code()
        hashed_password = pbkdf2_sha256.hash(password)
        
        # Try to send verification email first
        subject = "Verify Your Email - Brain Tumor MRI Classification App"
        body = f"""
        Thank you for registering!
        Your verification code is: {verification_code}
        
        Please enter this code in the app to verify your email address.
        """
        
        # Attempt to send email
        if not send_email(email, subject, body):
            conn.close()
            return False, "Error sending verification email"
        
        # If email sent successfully, add user to database
        c.execute("""INSERT INTO users 
                    (email, password, verification_code, is_verified) 
                    VALUES (?, ?, ?, ?)""", 
                 (email, hashed_password, verification_code, 0))
        conn.commit()
        return True, verification_code
        
    except sqlite3.IntegrityError as e:
        print(f"Database error: {e}")
        return False, "Database error occurred"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False, f"An unexpected error occurred: {str(e)}"
    finally:
        conn.close()

def verify_email(email, code):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("SELECT verification_code FROM users WHERE email=?", (email,))
        result = c.fetchone()
        
        if result and result[0] == code:
            c.execute("UPDATE users SET is_verified=1 WHERE email=?", (email,))
            conn.commit()
            return True
        return False
    except Exception as e:
        print(f"Error verifying email: {e}")
        return False
    finally:
        conn.close()

def verify_user(email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("SELECT password, is_verified FROM users WHERE email=?", (email,))
        result = c.fetchone()
        
        if result:
            is_password_valid = pbkdf2_sha256.verify(password, result[0])
            is_verified = result[1]
            return is_password_valid, is_verified
        return False, False
    except Exception as e:
        print(f"Error verifying user: {e}")
        return False, False
    finally:
        conn.close()

def store_reset_token(email):
    # Generate a shorter, more user-friendly code (8 characters)
    token = ''.join(secrets.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(8))
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET reset_token=? WHERE email=?", (token, email))
        conn.commit()
        return token
    except Exception as e:
        print(f"Error storing reset token: {e}")
        return None
    finally:
        conn.close()

def verify_reset_token(email, token):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute("SELECT reset_token FROM users WHERE email=?", (email,))
        result = c.fetchone()
        if result and result[0] == token:
            return True
        return False
    except Exception as e:
        print(f"Error verifying reset token: {e}")
        return False
    finally:
        conn.close()

def update_password(email, new_password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        hashed_password = pbkdf2_sha256.hash(new_password)
        c.execute("UPDATE users SET password=?, reset_token=NULL WHERE email=?", 
                 (hashed_password, email))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating password: {e}")
        return False
    finally:
        conn.close()

def send_reset_email(email, token):
    try:
        subject = "Password Reset Request"
        body = f"""
        You have requested to reset your password.
        Your password reset code is: {token}
        
        Please enter this code to reset your password.
        
        If you didn't request this, please ignore this email.
        """
        
        return send_email(email, subject, body)
            
    except Exception as e:
        print(f"Error sending reset email: {e}")
        return False