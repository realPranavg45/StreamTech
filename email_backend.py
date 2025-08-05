# email_backend.py
#
# MAIN BACKEND FILE FOR INTEGRATION
# This file is the main backend for user registration, authentication, and email sending.
# All credentials (Supabase, system email, etc.) are loaded from the .env file.
# Other files (e.g., stmail.py, test_auth.py) are for testing/demo purposes only.

import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional, Union, Any
from supabase import create_client, Client
import json
from datetime import datetime, timedelta
import traceback
from dotenv import load_dotenv
import hashlib
import secrets
import re
import bcrypt

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize EmailService with Supabase connection
        
        Args:
            supabase_url (str): Supabase project URL
            supabase_key (str): Supabase service key
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self._setup_database_schema()
        
    def _setup_database_schema(self):
        """Automatically create necessary database tables and columns"""
        try:
            # Create users table if it doesn't exist
            self._create_users_table()
            
            # Create OTP table if it doesn't exist
            self._create_otp_table()
            
            # Create email_logs table if it doesn't exist
            self._create_email_logs_table()
            
            logger.info("Database schema setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup database schema: {str(e)}")
    
    def _create_users_table(self):
        """Create users table with all required fields"""
        try:
            # Check if users table exists by trying to select from it
            result = self.supabase.table("users").select("id").limit(1).execute()
            logger.info("Users table already exists")
        except Exception:
            # Table doesn't exist, create it
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS users (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                first_name VARCHAR(100) NOT NULL,
                last_name VARCHAR(100) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                phone_number VARCHAR(20),
                password_hash VARCHAR(255) NOT NULL,
                is_verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            
            # Execute the SQL (this would need to be done through Supabase's SQL editor)
            # For now, we'll handle this through the application logic
            logger.info("Users table creation SQL prepared")
    
    def _create_otp_table(self):
        """Create OTP table for email verification"""
        try:
            # Check if otp table exists
            result = self.supabase.table("otp_codes").select("id").limit(1).execute()
            logger.info("OTP table already exists")
        except Exception:
            create_otp_sql = """
            CREATE TABLE IF NOT EXISTS otp_codes (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                otp_code VARCHAR(6) NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                is_used BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
            logger.info("OTP table creation SQL prepared")
    
    def _create_email_logs_table(self):
        """Create email_logs table if it doesn't exist"""
        try:
            # Check if email_logs table exists
            result = self.supabase.table("email_logs").select("id").limit(1).execute()
            logger.info("Email logs table already exists")
        except Exception:
            create_logs_sql = """
            CREATE TABLE IF NOT EXISTS email_logs (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                sender VARCHAR(255) NOT NULL,
                recipients TEXT[] NOT NULL,
                subject TEXT NOT NULL,
                status VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                error_message TEXT
            );
            """
            logger.info("Email logs table creation SQL prepared")
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """
        Real-time password validation
        
        Args:
            password (str): Password to validate
            
        Returns:
            Dict: Validation result with details
        """
        errors = []
        warnings = []
        
        # Minimum length check
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        elif len(password) < 12:
            warnings.append("Consider using a longer password (12+ characters)")
        
        # Character type checks
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Common password check
        common_passwords = ['password', '123456', 'qwerty', 'admin', 'letmein']
        if password.lower() in common_passwords:
            errors.append("Password is too common")
        
        # Sequential characters check
        if re.search(r'(.)\1{2,}', password):
            warnings.append("Avoid repeated characters")
        
        # Sequential numbers/letters check
        if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password.lower()):
            warnings.append("Avoid sequential letters")
        
        if re.search(r'(123|234|345|456|567|678|789|012)', password):
            warnings.append("Avoid sequential numbers")
        
        is_valid = len(errors) == 0
        strength_score = self._calculate_password_strength(password)
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "strength_score": strength_score,
            "strength_level": self._get_strength_level(strength_score)
        }
    
    def _calculate_password_strength(self, password: str) -> int:
        """Calculate password strength score (0-100)"""
        score = 0
        
        # Length contribution
        score += min(len(password) * 4, 40)
        
        # Character variety contribution
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'\d', password):
            score += 10
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 10
        
        # Bonus for longer passwords
        if len(password) > 12:
            score += 10
        if len(password) > 16:
            score += 10
        
        return min(score, 100)
    
    def _get_strength_level(self, score: int) -> str:
        """Get password strength level based on score"""
        if score >= 80:
            return "Very Strong"
        elif score >= 60:
            return "Strong"
        elif score >= 40:
            return "Moderate"
        elif score >= 20:
            return "Weak"
        else:
            return "Very Weak"
    
    def validate_registration_data(self, data: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate user registration data
        
        Args:
            data (Dict): Registration data with keys: first_name, last_name, email, phone_number, password, confirm_password
            
        Returns:
            Dict: Validation result
        """
        errors = {}
        warnings = {}
        
        # First Name validation
        if not data.get('first_name', '').strip():
            errors['first_name'] = "First name is required"
        elif len(data['first_name'].strip()) < 2:
            errors['first_name'] = "First name must be at least 2 characters"
        elif not re.match(r'^[a-zA-Z\s]+$', data['first_name'].strip()):
            errors['first_name'] = "First name should only contain letters and spaces"
        
        # Last Name validation
        if not data.get('last_name', '').strip():
            errors['last_name'] = "Last name is required"
        elif len(data['last_name'].strip()) < 2:
            errors['last_name'] = "Last name must be at least 2 characters"
        elif not re.match(r'^[a-zA-Z\s]+$', data['last_name'].strip()):
            errors['last_name'] = "Last name should only contain letters and spaces"
        
        # Email validation
        email = data.get('email', '').strip().lower()
        if not email:
            errors['email'] = "Email is required"
        elif not self._is_valid_email(email):
            errors['email'] = "Please enter a valid email address"
        else:
            # Check if email already exists
            try:
                existing_user = self.supabase.table("users").select("email").eq("email", email).execute()
                if existing_user.data:
                    errors['email'] = "Email address is already registered"
            except Exception as e:
                logger.error(f"Error checking email existence: {str(e)}")
        
        # Phone number validation (optional)
        phone = data.get('phone_number', '').strip()
        if phone:
            # Remove all non-digit characters for validation
            phone_digits = re.sub(r'\D', '', phone)
            if len(phone_digits) < 10:
                errors['phone_number'] = "Phone number must have at least 10 digits"
            elif len(phone_digits) > 15:
                errors['phone_number'] = "Phone number is too long"
        
        # Password validation
        password = data.get('password', '')
        password_validation = self.validate_password(password)
        if not password_validation['is_valid']:
            errors['password'] = password_validation['errors']
        elif password_validation['warnings']:
            warnings['password'] = password_validation['warnings']
        
        # Confirm password validation
        confirm_password = data.get('confirm_password', '')
        if not confirm_password:
            errors['confirm_password'] = "Please confirm your password"
        elif password != confirm_password:
            errors['confirm_password'] = "Passwords do not match"
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "password_strength": password_validation.get('strength_level', 'Unknown') if password else None
        }
    
    def register_user(self, registration_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Register a new user
        
        Args:
            registration_data (Dict): User registration data
            
        Returns:
            Dict: Registration result
        """
        try:
            # Validate registration data
            validation_result = self.validate_registration_data(registration_data)
            
            if not validation_result['is_valid']:
                return {
                    "success": False,
                    "errors": validation_result['errors'],
                    "warnings": validation_result['warnings'],
                    "popup_toast": "Please fix the validation errors"
                }
            
            # Hash password
            password_hash = bcrypt.hashpw(
                registration_data['password'].encode('utf-8'),
                bcrypt.gensalt()
            ).decode('utf-8')
            
            # Prepare user data
            user_data = {
                "first_name": registration_data['first_name'].strip(),
                "last_name": registration_data['last_name'].strip(),
                "email": registration_data['email'].strip().lower(),
                "phone_number": registration_data.get('phone_number', '').strip(),
                "password_hash": password_hash,
                "is_verified": False,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Insert user into database
            result = self.supabase.table("users").insert(user_data).execute()
            
            if result.data:
                # Generate and send OTP
                otp_result = self._generate_and_send_otp(user_data['email'])
                
                if otp_result['success']:
                    return {
                        "success": True,
                        "user_id": result.data[0]['id'],
                        "email": user_data['email'],
                        "popup_toast": "Registration successful! Please check your email for verification.",
                        "warnings": validation_result.get('warnings', {})
                    }
                else:
                    # User created but OTP failed
                    return {
                        "success": False,
                        "error": "User registered but verification email failed to send",
                        "popup_toast": "Registration completed but verification email failed. Please contact support."
                    }
            else:
                return {
                    "success": False,
                    "error": "Failed to create user account",
                    "popup_toast": "Registration failed. Please try again."
                }
                
        except Exception as e:
            error_msg = f"Registration error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg,
                "popup_toast": "An unexpected error occurred during registration."
            }
    
    def _generate_and_send_otp(self, email: str) -> Dict[str, Any]:
        """
        Generate OTP and send verification email
        
        Args:
            email (str): User's email address
            
        Returns:
            Dict: OTP generation and sending result
        """
        try:
            # Generate 6-digit OTP
            otp_code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
            
            # Store OTP in database with expiration (15 minutes)
            otp_data = {
                "email": email,
                "otp_code": otp_code,
                "expires_at": (datetime.now() + timedelta(minutes=15)).isoformat(),
                "is_used": False
            }
            
            result = self.supabase.table("otp_codes").insert(otp_data).execute()
            
            if not result.data:
                return {
                    "success": False,
                    "error": "Failed to store OTP"
                }
            
            # Send OTP email
            subject = "Email Verification - SecureMail Pro"
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #667eea;">🔐 Email Verification</h2>
                    <p>Thank you for registering with SecureMail Pro!</p>
                    <p>Your verification code is:</p>
                    <div style="background: #f8f9fa; padding: 20px; text-align: center; border-radius: 8px; margin: 20px 0;">
                        <h1 style="color: #667eea; font-size: 32px; letter-spacing: 5px; margin: 0;">{otp_code}</h1>
                    </div>
                    <p><strong>Important:</strong></p>
                    <ul>
                        <li>This code will expire in 15 minutes</li>
                        <li>If you didn't request this verification, please ignore this email</li>
                        <li>For security, never share this code with anyone</li>
                    </ul>
                    <p>Best regards,<br><strong>SecureMail Pro Team</strong></p>
                </div>
            </body>
            </html>
            """
            
            # Use system email service to send OTP
            # Note: This requires system email credentials to be configured
            system_email_creds = {
                "email": os.getenv("SYSTEM_EMAIL"),
                "password": os.getenv("SYSTEM_EMAIL_PASSWORD"),
                "smtp_server": os.getenv("SYSTEM_SMTP_SERVER", "smtp.gmail.com"),
                "smtp_port": os.getenv("SYSTEM_SMTP_PORT", "587")
            }
            
            if all(system_email_creds.values()):
                email_result = self.send_email(
                    login_credentials=system_email_creds,
                    to=[email],
                    cc=[],
                    bcc=[],
                    subject=subject,
                    body=body
                )
                
                if email_result['success']:
                    return {
                        "success": True,
                        "otp_code": otp_code  # For testing purposes only
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to send OTP email: {email_result.get('error', 'Unknown error')}"
                    }
            else:
                # For development/testing, return OTP without sending email
                return {
                    "success": True,
                    "otp_code": otp_code,
                    "warning": "System email not configured - OTP returned for testing"
                }
                
        except Exception as e:
            error_msg = f"OTP generation error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def verify_otp(self, email: str, otp_code: str) -> Dict[str, Any]:
        """
        Verify OTP and activate user account
        
        Args:
            email (str): User's email address
            otp_code (str): OTP code to verify
            
        Returns:
            Dict: Verification result
        """
        try:
            # Find valid OTP
            result = self.supabase.table("otp_codes").select("*").eq("email", email).eq("otp_code", otp_code).eq("is_used", False).execute()
            
            if not result.data:
                return {
                    "success": False,
                    "error": "Invalid or expired OTP code",
                    "popup_toast": "Invalid verification code. Please check and try again."
                }
            
            otp_record = result.data[0]
            expires_at = datetime.fromisoformat(otp_record['expires_at'].replace('Z', '+00:00'))
            
            # Check if OTP is expired
            if datetime.now(expires_at.tzinfo) > expires_at:
                return {
                    "success": False,
                    "error": "OTP code has expired",
                    "popup_toast": "Verification code has expired. Please request a new one."
                }
            
            # Mark OTP as used
            self.supabase.table("otp_codes").update({"is_used": True}).eq("id", otp_record['id']).execute()
            
            # Activate user account
            user_result = self.supabase.table("users").update({
                "is_verified": True,
                "updated_at": datetime.now().isoformat()
            }).eq("email", email).execute()
            
            if user_result.data:
                return {
                    "success": True,
                    "popup_toast": "Email verified successfully! You can now log in to your account."
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to activate user account",
                    "popup_toast": "Verification failed. Please contact support."
                }
                
        except Exception as e:
            error_msg = f"OTP verification error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg,
                "popup_toast": "An unexpected error occurred during verification."
            }
    
    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user login
        
        Args:
            email (str): User's email address
            password (str): User's password
            
        Returns:
            Dict: Login result
        """
        try:
            # Find user by email
            result = self.supabase.table("users").select("*").eq("email", email.lower()).execute()
            
            if not result.data:
                return {
                    "success": False,
                    "error": "Invalid email or password",
                    "popup_toast": "Invalid email or password. Please try again."
                }
            
            user = result.data[0]
            
            # Check if account is verified
            if not user.get('is_verified', False):
                return {
                    "success": False,
                    "error": "Account not verified",
                    "popup_toast": "Please verify your email address before logging in.",
                    "needs_verification": True,
                    "email": email
                }
            
            # Verify password
            if bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                return {
                    "success": True,
                    "user": {
                        "id": user['id'],
                        "email": user['email'],
                        "first_name": user['first_name'],
                        "last_name": user['last_name'],
                        "phone_number": user.get('phone_number', ''),
                        "is_verified": user['is_verified']
                    },
                    "popup_toast": f"Welcome back, {user['first_name']}!"
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid email or password",
                    "popup_toast": "Invalid email or password. Please try again."
                }
                
        except Exception as e:
            error_msg = f"Login error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg,
                "popup_toast": "An unexpected error occurred during login."
            }
    
    def resend_otp(self, email: str) -> Dict[str, Any]:
        """
        Resend OTP for email verification
        
        Args:
            email (str): User's email address
            
        Returns:
            Dict: Resend result
        """
        try:
            # Check if user exists and is not verified
            user_result = self.supabase.table("users").select("is_verified").eq("email", email.lower()).execute()
            
            if not user_result.data:
                return {
                    "success": False,
                    "error": "Email address not found",
                    "popup_toast": "Email address not found in our records."
                }
            
            if user_result.data[0]['is_verified']:
                return {
                    "success": False,
                    "error": "Account already verified",
                    "popup_toast": "Your account is already verified. You can log in directly."
                }
            
            # Generate and send new OTP
            return self._generate_and_send_otp(email)
            
        except Exception as e:
            error_msg = f"Resend OTP error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": error_msg,
                "popup_toast": "An unexpected error occurred while resending verification code."
            }
    
    def send_email(
        self,
        login_credentials: Dict[str, str],
        to: List[str],
        cc: List[str],
        bcc: List[str],
        subject: str,
        body: str,
        attachments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send email function matching the specified parameters
        
        Args:
            login_credentials (Dict): Dictionary containing email login details
                - Required keys: 'email', 'password', 'smtp_server', 'smtp_port'
            to (List[str]): List of recipient email addresses (1 or more)
            cc (List[str]): List of CC email addresses (1 or more)
            bcc (List[str]): List of BCC email addresses (1 or more)
            subject (str): Email subject (HTML format supported)
            body (str): Email body (HTML format supported)
            attachments (List[str], optional): List of file paths to attach
            
        Returns:
            Dict: Success response with popup/toast message or error details
        """
        try:
            # Validate required parameters
            if not self._validate_parameters(login_credentials, to, cc, bcc, subject, body):
                return {
                    "success": False,
                    "error": "Invalid parameters provided",
                    "popup_toast": "Error: Missing required parameters"
                }
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = login_credentials['email']
            msg['To'] = ', '.join(to)
            
            if cc:
                msg['Cc'] = ', '.join(cc)
            
            msg['Subject'] = subject
            
            # Add body to email (HTML format)
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachments if provided
            if attachments:
                for file_path in attachments:
                    if os.path.isfile(file_path):
                        self._attach_file(msg, file_path)
                    else:
                        logger.warning(f"Attachment file not found: {file_path}")
            
            # Combine all recipients
            all_recipients = to + cc + bcc
            
            # Send email
            with smtplib.SMTP(login_credentials['smtp_server'], int(login_credentials['smtp_port'])) as server:
                server.starttls()
                server.login(login_credentials['email'], login_credentials['password'])
                text = msg.as_string()
                server.sendmail(login_credentials['email'], all_recipients, text)
            
            # Log successful email send to database
            self._log_email_activity(
                sender=login_credentials['email'],
                recipients=all_recipients,
                subject=subject,
                status="success"
            )
            
            logger.info(f"Email sent successfully to {len(all_recipients)} recipients")
            
            return {
                "success": True,
                "popup_toast": f"Email sent successfully to {len(all_recipients)} recipients!",
                "recipients_count": len(all_recipients),
                "timestamp": datetime.now().isoformat()
            }
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP Authentication failed: {str(e)}"
            logger.error(error_msg)
            self._log_email_activity(
                sender=login_credentials.get('email', 'unknown'),
                recipients=to + cc + bcc,
                subject=subject,
                status="failed",
                error_message=error_msg
            )
            return {
                "success": False,
                "error": error_msg,
                "popup_toast": "Authentication failed. Please check email credentials."
            }
            
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error occurred: {str(e)}"
            logger.error(error_msg)
            self._log_email_activity(
                sender=login_credentials.get('email', 'unknown'),
                recipients=to + cc + bcc,
                subject=subject,
                status="failed",
                error_message=error_msg
            )
            return {
                "success": False,
                "error": error_msg,
                "popup_toast": "Failed to send email. Please try again."
            }
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self._log_email_activity(
                sender=login_credentials.get('email', 'unknown'),
                recipients=to + cc + bcc,
                subject=subject,
                status="failed",
                error_message=error_msg
            )
            return {
                "success": False,
                "error": error_msg,
                "popup_toast": "An unexpected error occurred."
            }
    
    def _validate_parameters(
        self,
        login_credentials: Dict[str, str],
        to: List[str],
        cc: List[str],
        bcc: List[str],
        subject: str,
        body: str
    ) -> bool:
        """Validate input parameters"""
        
        # Check login credentials
        required_creds = ['email', 'password', 'smtp_server', 'smtp_port']
        if not all(key in login_credentials for key in required_creds):
            logger.error("Missing required login credentials")
            return False
        
        # Check that at least one recipient exists (to, cc, or bcc must have 1 or more)
        if not (to or cc or bcc):
            logger.error("At least one recipient (to, cc, or bcc) is required")
            return False
        
        # Validate email addresses
        all_emails = to + cc + bcc
        for email in all_emails:
            if not self._is_valid_email(email):
                logger.error(f"Invalid email address: {email}")
                return False
        
        # Check subject and body
        if not subject or not body:
            logger.error("Subject and body are required")
            return False
        
        return True
    
    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _attach_file(self, msg: MIMEMultipart, file_path: str):
        """Attach file to email message"""
        try:
            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(file_path)}'
            )
            msg.attach(part)
            logger.info(f"Attached file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to attach file {file_path}: {str(e)}")
    
    def _log_email_activity(
        self,
        sender: str,
        recipients: List[str],
        subject: str,
        status: str,
        error_message: str = None
    ):
        """Log email activity to Supabase database"""
        try:
            log_data = {
                "sender": sender,
                "recipients": recipients,
                "subject": subject,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "error_message": error_message
            }
            
            # Insert into email_logs table (you need to create this table in Supabase)
            result = self.supabase.table("email_logs").insert(log_data).execute()
            logger.info("Email activity logged to database")
            
        except Exception as e:
            logger.error(f"Failed to log email activity: {str(e)}")

# Initialize EmailService using environment variables
def get_email_service():
    """Factory function to create EmailService instance"""
    supabase_url = "----" 
    supabase_key = "---"
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")
    
    return EmailService(supabase_url, supabase_key)

# Main function for testing
if __name__ == "__main__":
    # Example usage
    email_service = get_email_service()
    
    # Example parameters - UPDATE THESE WITH YOUR ACTUAL CREDENTIALS
    login_creds = {
        "email": "123349@gmail.com",  # Replace with your Gmail
        "password": "----",  # Replace with App Password (no spaces)
        "smtp_server": "smtp.gmail.com",
        "smtp_port": "587"
    }
    
    result = email_service.send_email(
        login_credentials=login_creds,
        to=[],  # Replace with your email for testing
        cc=[],  # Empty list is fine
        bcc=[],  # Empty list is fine
        subject="<h2>Test Subject</h2>",
        body="<html><body><h1>Hello World!</h1><p>This is a test email.</p></body></html>",
        attachments=None  # Remove the file that doesn't exist
    )
    
    print(json.dumps(result, indent=2))