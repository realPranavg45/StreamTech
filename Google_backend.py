# backend_pure.py

from pydantic import BaseModel, EmailStr, ValidationError
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import requests
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from sqlalchemy import desc
import supabase
import traceback
import calendar

load_dotenv()

# ------------------- Models -------------------
class MeetingData(BaseModel):
    subject: str
    start_datetime: str
    end_datetime: str
    description: Optional[str] = ""
    attendees: Optional[List[EmailStr]] = []
    user_email: EmailStr

class MeetingDisplay(BaseModel):
    id: str
    subject: str
    start_datetime: str
    end_datetime: str
    description: str
    meet_link: Optional[str] = ""
    attendees: List[str]
    status: str

class CalendarEvent(BaseModel):
    id: str
    title: str
    start: str
    end: str
    description: str
    meet_link: Optional[str] = ""
    attendees: List[str]

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    due_datetime: Optional[str] = None  # ISO format
    user_email: EmailStr

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    due_datetime: Optional[str] = None
    status: Optional[str] = None

class TaskDisplay(BaseModel):
    id: int
    title: str
    description: str
    due_datetime: Optional[str]
    user_email: EmailStr
    status: str

class QuickCreateData(BaseModel):
    date: str  # YYYY-MM-DD format
    type: str  # "meeting" or "task"
    title: str
    time: Optional[str] = None  # HH:MM format for meetings
    duration: Optional[int] = 60  # minutes for meetings
    user_email: EmailStr

class DateEventsResponse(BaseModel):
    date: str
    meetings: List[MeetingDisplay]
    tasks: List[TaskDisplay]

# ------------------- Configuration -------------------
class BackendConfig:
    def __init__(self):
        self.GOOGLE_CLIENT_ID = "---"
        self.GOOGLE_CLIENT_SECRET = "---"
        self.REDIRECT_URI = "---"
        self.SUPABASE_API_URL = os.getenv("SUPABASE_API_URL")
        self.SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
        self.SMTP_SERVER = os.getenv("SMTP_SERVER")
        self.SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
        self.SMTP_USERNAME = os.getenv("SMTP_USERNAME")
        self.SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
        self.EMAIL_FROM = os.getenv("EMAIL_FROM")
        
        # Initialize Supabase client
        try:
            self.supabase_client = supabase.create_client(self.SUPABASE_API_URL, self.SUPABASE_API_KEY)
            print("✅ Supabase client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Supabase client: {e}")
            self.supabase_client = None

# Global config instance
config = BackendConfig()

# ------------------- Custom Exceptions -------------------
class BackendError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)

# ------------------- Database Setup -------------------
def create_tables():
    """Create necessary database tables - Run these SQL commands in your Supabase SQL editor"""
    
    sql_commands = """
    -- Main meetings table
    CREATE TABLE IF NOT EXISTS meetings (
        id SERIAL PRIMARY KEY,
        subject VARCHAR(255) NOT NULL,
        start_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
        end_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
        description TEXT DEFAULT '',
        user_email VARCHAR(255) NOT NULL,
        google_event_id VARCHAR(255) UNIQUE,
        status VARCHAR(50) DEFAULT 'scheduled',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Meeting links table
    CREATE TABLE IF NOT EXISTS meeting_links (
        id SERIAL PRIMARY KEY,
        meeting_id INTEGER REFERENCES meetings(id) ON DELETE CASCADE,
        meet_link VARCHAR(500) NOT NULL,
        link_type VARCHAR(50) DEFAULT 'google_meet',
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Meeting attendees table
    CREATE TABLE IF NOT EXISTS meeting_attendees (
        id SERIAL PRIMARY KEY,
        meeting_id INTEGER REFERENCES meetings(id) ON DELETE CASCADE,
        email VARCHAR(255) NOT NULL,
        response_status VARCHAR(50) DEFAULT 'needsAction',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(meeting_id, email)
    );

    -- Tasks table (FIXED SCHEMA)
    CREATE TABLE IF NOT EXISTS tasks (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        description TEXT DEFAULT '',
        due_datetime TIMESTAMP WITH TIME ZONE,
        user_email VARCHAR(255) NOT NULL,
        status VARCHAR(50) DEFAULT 'pending',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Meeting update notifications log
    CREATE TABLE IF NOT EXISTS meeting_notifications (
        id SERIAL PRIMARY KEY,
        meeting_id INTEGER REFERENCES meetings(id) ON DELETE CASCADE,
        notification_type VARCHAR(50) NOT NULL,
        sent_to TEXT[],
        sent_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        changes_summary TEXT
    );

    -- Add indexes for better performance
    CREATE INDEX IF NOT EXISTS idx_meetings_user_email ON meetings(user_email);
    CREATE INDEX IF NOT EXISTS idx_meetings_start_datetime ON meetings(start_datetime);
    CREATE INDEX IF NOT EXISTS idx_tasks_user_email ON tasks(user_email);
    CREATE INDEX IF NOT EXISTS idx_tasks_due_datetime ON tasks(due_datetime);
    CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
    """
    
    print("📋 SQL Commands to run in Supabase SQL Editor:")
    print("=" * 50)
    print(sql_commands)
    print("=" * 50)
    return True

# ------------------- Helper Functions -------------------
def get_google_access_token(user_id: str) -> str:
    if user_id == "demo-user-id":
        return os.getenv("GOOGLE_TEST_ACCESS_TOKEN")  # For demo/testing
    else:
        # In production, fetch from database
        raise BackendError(status_code=404, detail="Access token not found for user")

def create_google_calendar_event(meeting: MeetingData, access_token: str):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    body = {
        "summary": meeting.subject,
        "description": meeting.description,
        "start": {
            "dateTime": meeting.start_datetime,
            "timeZone": "UTC"
        },
        "end": {
            "dateTime": meeting.end_datetime,
            "timeZone": "UTC"
        },
        "attendees": [{"email": str(email)} for email in meeting.attendees] if meeting.attendees else [],
        "conferenceData": {
            "createRequest": {
                "requestId": f"meet-{datetime.utcnow().timestamp()}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"}
            }
        }
    }

    url = "https://www.googleapis.com/calendar/v3/calendars/primary/events?conferenceDataVersion=1"
    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code not in (200, 201):
        raise BackendError(status_code=500, detail=f"Google Calendar error: {response.text}")

    return response.json()

def send_email(to_emails: List[str], subject: str, body: str, is_html: bool = False):
    try:
        msg = MIMEMultipart()
        msg['From'] = config.EMAIL_FROM
        msg['To'] = ", ".join(to_emails)
        msg['Subject'] = subject

        if is_html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
            server.send_message(msg)
            
    except Exception as e:
        raise BackendError(status_code=500, detail=f"Email sending failed: {str(e)}")

def save_to_supabase(meeting_data: dict, event_data: dict):
    try:
        if not config.supabase_client:
            raise Exception("Supabase client not initialized")
            
        # Insert main meeting record
        meeting_record = {
            "subject": meeting_data["subject"],
            "start_datetime": meeting_data["start_datetime"],
            "end_datetime": meeting_data["end_datetime"],
            "description": meeting_data.get("description", ""),
            "user_email": meeting_data["user_email"],
            "google_event_id": event_data.get("id"),
            "status": "scheduled",
            "created_at": datetime.utcnow().isoformat()
        }

        meeting_response = config.supabase_client.table("meetings").insert(meeting_record).execute()
        
        if not meeting_response.data:
            raise Exception("Failed to create meeting record")
            
        meeting_id = meeting_response.data[0]["id"]
        
        # Save meeting link if exists
        meet_link = event_data.get("hangoutLink") or event_data.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri")
        if meet_link:
            link_record = {
                "meeting_id": meeting_id,
                "meet_link": meet_link,
                "link_type": "google_meet",
                "is_active": True
            }
            config.supabase_client.table("meeting_links").insert(link_record).execute()
        
        # Save attendees
        if meeting_data.get("attendees"):
            attendee_records = [
                {
                    "meeting_id": meeting_id,
                    "email": str(email),
                    "response_status": "needsAction"
                }
                for email in meeting_data["attendees"]
            ]
            config.supabase_client.table("meeting_attendees").insert(attendee_records).execute()

        return meeting_response.data[0]
        
    except Exception as e:
        print(f"❌ Database error in save_to_supabase: {str(e)}")
        raise BackendError(status_code=500, detail=f"Database error: {str(e)}")

def update_google_calendar_event(google_event_id: str, meeting: MeetingData, access_token: str):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    body = {
        "summary": meeting.subject,
        "description": meeting.description,
        "start": {
            "dateTime": meeting.start_datetime,
            "timeZone": "UTC"
        },
        "end": {
            "dateTime": meeting.end_datetime,
            "timeZone": "UTC"
        },
        "attendees": [{"email": str(email)} for email in meeting.attendees] if meeting.attendees else []
    }

    url = f"https://www.googleapis.com/calendar/v3/calendars/primary/events/{google_event_id}?conferenceDataVersion=1"
    response = requests.patch(url, headers=headers, data=json.dumps(body))

    if response.status_code not in (200, 201):
        raise BackendError(status_code=500, detail=f"Google Calendar update error: {response.text}")

    return response.json()

def create_meeting_invite_email(meeting: dict, event_data: dict):
    start_time = datetime.fromisoformat(meeting["start_datetime"]).strftime("%d-%m-%y %I:%M %p")
    end_time = datetime.fromisoformat(meeting["end_datetime"]).strftime("%I:%M %p")
    
    meet_link = event_data.get("hangoutLink") or event_data.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri", "")
    
    html_content = f"""
    <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: #4285f4; color: white; padding: 20px; text-align: center;">
                <h2>📅 Meeting Invitation: {meeting['subject']}</h2>
            </div>
            <div style="padding: 20px; background-color: #f9f9f9;">
                <div style="background-color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h3 style="color: #333; margin-top: 0;">Meeting Details</h3>
                    <p><strong>📅 When:</strong> {start_time} - {end_time}</p>
                    <p><strong>📝 Description:</strong> {meeting.get('description', 'No description provided')}</p>
                    
                    {f'''
                    <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <p><strong>🎥 Join Google Meet:</strong></p>
                        <a href="{meet_link}" style="background-color: #4285f4; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">Join Meeting</a>
                        <p style="font-size: 12px; color: #666; margin-top: 10px;">Link: {meet_link}</p>
                    </div>
                    ''' if meet_link else ''}
                </div>
                <p style="text-align: center; color: #666; font-size: 12px;">
                    Please add this event to your calendar and join on time.
                </p>
            </div>
        </body>
    </html>
    """
    return html_content

def create_meeting_update_email(old_meeting: dict, new_meeting: dict, changes_summary: str, meet_link: str = ""):
    """Create email template for meeting update notifications"""
    start_time = datetime.fromisoformat(new_meeting["start_datetime"]).strftime("%d-%m-%y %I:%M %p")
    end_time = datetime.fromisoformat(new_meeting["end_datetime"]).strftime("%I:%M %p")
    
    html_content = f"""
    <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: #ff9800; color: white; padding: 20px; text-align: center;">
                <h2>🔄 Meeting Updated: {new_meeting['subject']}</h2>
            </div>
            <div style="padding: 20px; background-color: #f9f9f9;">
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ff9800; margin-bottom: 20px;">
                    <h4 style="margin-top: 0; color: #856404;">⚠️ Important: Meeting Details Have Changed</h4>
                    <p style="color: #856404; margin-bottom: 0;">This meeting has been updated. Please review the new details below.</p>
                </div>
                
                <div style="background-color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h3 style="color: #333; margin-top: 0;">Updated Meeting Details</h3>
                    <p><strong>📅 When:</strong> {start_time} - {end_time}</p>
                    <p><strong>📝 Description:</strong> {new_meeting.get('description', 'No description provided')}</p>
                    
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0;">
                        <h4 style="color: #495057; margin-top: 0;">📋 What Changed:</h4>
                        <p style="color: #6c757d; margin-bottom: 0;">{changes_summary}</p>
                    </div>
                    
                    {f'''
                    <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <p><strong>🎥 Join Google Meet:</strong></p>
                        <a href="{meet_link}" style="background-color: #4285f4; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">Join Meeting</a>
                        <p style="font-size: 12px; color: #666; margin-top: 10px;">Link: {meet_link}</p>
                    </div>
                    ''' if meet_link else ''}
                </div>
                
                <div style="background-color: white; padding: 20px; border-radius: 8px; text-align: center;">
                    <p style="color: #666; margin-bottom: 10px;">Please update your calendar with the new meeting details.</p>
                    <p style="color: #666; font-size: 12px; margin-bottom: 0;">
                        If you cannot attend with the new schedule, please notify the meeting organizer.
                    </p>
                </div>
            </div>
        </body>
    </html>
    """
    return html_content

def generate_changes_summary(old_meeting: dict, new_meeting: dict) -> str:
    """Generate a summary of what changed in the meeting"""
    changes = []
    
    # Check subject change
    if old_meeting.get('subject') != new_meeting.get('subject'):
        changes.append(f"Subject changed from '{old_meeting.get('subject')}' to '{new_meeting.get('subject')}'")
    
    # Check time changes
    old_start = datetime.fromisoformat(old_meeting['start_datetime'].replace('Z', '+00:00'))
    new_start = datetime.fromisoformat(new_meeting['start_datetime'])
    if old_start != new_start:
        changes.append(f"Start time changed from {old_start.strftime('%d-%m-%y %I:%M %p')} to {new_start.strftime('%d-%m-%y %I:%M %p')}")
    
    old_end = datetime.fromisoformat(old_meeting['end_datetime'].replace('Z', '+00:00'))
    new_end = datetime.fromisoformat(new_meeting['end_datetime'])
    if old_end != new_end:
        changes.append(f"End time changed from {old_end.strftime('%d-%m-%y %I:%M %p')} to {new_end.strftime('%d-%m-%y %I:%M %p')}")
    
    # Check description change
    if old_meeting.get('description', '') != new_meeting.get('description', ''):
        changes.append("Meeting description was updated")
    
    # Check attendees change
    old_attendees = set(old_meeting.get('attendees', []))
    new_attendees = set(str(email) for email in new_meeting.get('attendees', []))
    if old_attendees != new_attendees:
        added = new_attendees - old_attendees
        removed = old_attendees - new_attendees
        if added:
            changes.append(f"Added attendees: {', '.join(added)}")
        if removed:
            changes.append(f"Removed attendees: {', '.join(removed)}")
    
    return "; ".join(changes) if changes else "Minor updates were made to the meeting"

def log_meeting_notification(meeting_id: int, notification_type: str, sent_to: List[str], changes_summary: str = ""):
    """Log meeting notification to database"""
    try:
        if not config.supabase_client:
            print("⚠️ Cannot log notification - Supabase client not available")
            return
            
        notification_record = {
            "meeting_id": meeting_id,
            "notification_type": notification_type,
            "sent_to": sent_to,
            "changes_summary": changes_summary,
            "sent_at": datetime.utcnow().isoformat()
        }
        
        config.supabase_client.table("meeting_notifications").insert(notification_record).execute()
        print(f"✅ Logged notification for meeting {meeting_id}")
    except Exception as e:
        print(f"❌ Error logging notification: {e}")

# ------------------- Task Helper Functions -------------------
def validate_task_data(task_data: dict) -> dict:
    """Validate and clean task data before database operations"""
    cleaned_data = {}
    
    # Required fields
    cleaned_data["title"] = str(task_data.get("title", "")).strip()
    if not cleaned_data["title"]:
        raise ValueError("Task title is required")
    
    cleaned_data["user_email"] = str(task_data.get("user_email", "")).strip()
    if not cleaned_data["user_email"]:
        raise ValueError("User email is required")
    
    # Optional fields with defaults
    cleaned_data["description"] = str(task_data.get("description", "")).strip()
    cleaned_data["status"] = task_data.get("status", "pending")
    
    # Handle due_datetime
    due_datetime = task_data.get("due_datetime")
    if due_datetime:
        try:
            # Validate datetime format
            datetime.fromisoformat(due_datetime.replace('Z', '+00:00'))
            cleaned_data["due_datetime"] = due_datetime
        except ValueError:
            raise ValueError("Invalid due_datetime format. Use ISO format.")
    else:
        cleaned_data["due_datetime"] = None
    
    # Add timestamps
    cleaned_data["updated_at"] = datetime.utcnow().isoformat()
    
    return cleaned_data

# ------------------- Backend Service Functions -------------------

def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "supabase": "connected" if config.supabase_client else "disconnected",
            "environment_vars": {
                "SUPABASE_API_URL": bool(config.SUPABASE_API_URL),
                "SUPABASE_API_KEY": bool(config.SUPABASE_API_KEY),
                "SMTP_SERVER": bool(config.SMTP_SERVER)
            }
        }
    }
    
    if not config.supabase_client:
        health_status["status"] = "degraded"
        health_status["warnings"] = ["Supabase client not initialized"]
    
    return health_status

# ------------------- Task Service Functions -------------------

def create_task(task: TaskCreate) -> TaskDisplay:
    """Create a new task"""
    try:
        if not config.supabase_client:
            raise BackendError(status_code=503, detail="Database service unavailable")
        
        # Validate and clean task data
        task_data = validate_task_data(task.dict())
        
        # Remove updated_at for creation
        task_data.pop("updated_at", None)
        task_data["created_at"] = datetime.utcnow().isoformat()
        
        print(f"📝 Creating task with data: {task_data}")
        
        # Insert into database
        response = config.supabase_client.table("tasks").insert(task_data).execute()
        
        if not response.data:
            raise Exception("No data returned from database insert")
        
        created_task = response.data[0]
        print(f"✅ Task created successfully: {created_task['id']}")
        
        return TaskDisplay(**created_task)
        
    except ValueError as ve:
        print(f"❌ Validation error: {ve}")
        raise BackendError(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"❌ Task creation error: {str(e)}")
        raise BackendError(status_code=500, detail=f"Task creation error: {str(e)}")

def update_task(task_id: int, task: TaskUpdate) -> TaskDisplay:
    """Update an existing task"""
    try:
        if not config.supabase_client:
            raise BackendError(status_code=503, detail="Database service unavailable")
        
        # Get non-null update fields
        update_data = {k: v for k, v in task.dict().items() if v is not None}
        
        if not update_data:
            raise BackendError(status_code=400, detail="No fields to update")
        
        # Validate update data
        if update_data:
            update_data = validate_task_data({**update_data, "title": update_data.get("title", "temp"), "user_email": "temp@temp.com"})
            # Remove the temp values we added for validation
            if task.title is None:
                update_data.pop("title", None)
            update_data.pop("user_email", None)
            update_data.pop("created_at", None)
        
        print(f"📝 Updating task {task_id} with data: {update_data}")
        
        # Update in database
        response = config.supabase_client.table("tasks").update(update_data).eq("id", task_id).execute()
        
        if not response.data:
            raise BackendError(status_code=404, detail="Task not found")
        
        updated_task = response.data[0]
        print(f"✅ Task updated successfully: {task_id}")
        
        return TaskDisplay(**updated_task)
        
    except ValueError as ve:
        print(f"❌ Validation error: {ve}")
        raise BackendError(status_code=400, detail=str(ve))
    except BackendError:
        raise
    except Exception as e:
        print(f"❌ Task update error: {str(e)}")
        raise BackendError(status_code=500, detail=f"Task update error: {str(e)}")

def delete_task(task_id: int) -> dict:
    """Delete a task"""
    try:
        if not config.supabase_client:
            raise BackendError(status_code=503, detail="Database service unavailable")
        
        print(f"🗑️ Deleting task: {task_id}")
        
        # Delete from database
        response = config.supabase_client.table("tasks").delete().eq("id", task_id).execute()
        
        if not response.data:
            raise BackendError(status_code=404, detail="Task not found")
        
        print(f"✅ Task deleted successfully: {task_id}")
        return {"message": "Task deleted successfully", "task_id": task_id}
        
    except BackendError:
        raise
    except Exception as e:
        print(f"❌ Task deletion error: {str(e)}")
        raise BackendError(status_code=500, detail=f"Task deletion error: {str(e)}")

def list_tasks(user_email: str, date: Optional[str] = None, status: Optional[str] = None) -> List[TaskDisplay]:
    """List tasks with optional filtering"""
    try:
        if not config.supabase_client:
            raise BackendError(status_code=503, detail="Database service unavailable")
        
        print(f"📋 Fetching tasks for user: {user_email}, date: {date}, status: {status}")
        
        # Build query
        query = config.supabase_client.table("tasks").select("*").eq("user_email", user_email)
        
        # Apply date filter if provided
        if date:
            # For date filtering, we want all tasks due on that specific date
            start_of_day = f"{date}T00:00:00"
            end_of_day = f"{date}T23:59:59"
            query = query.gte("due_datetime", start_of_day).lte("due_datetime", end_of_day)
        
        # Apply status filter if provided
        if status:
            query = query.eq("status", status)
        
        # Execute query with ordering
        response = query.order("due_datetime", desc=True).execute()
        
        tasks = response.data or []
        print(f"✅ Found {len(tasks)} tasks")
        
        return [TaskDisplay(**task) for task in tasks]
        
    except Exception as e:
        print(f"❌ Failed to fetch tasks: {str(e)}")
        raise BackendError(status_code=500, detail=f"Failed to fetch tasks: {str(e)}")

# ------------------- Meeting Service Functions -------------------

def create_meeting(data: MeetingData, user_id: str = "demo-user-id") -> dict:
    """Create a new meeting"""
    try:
        access_token = get_google_access_token(user_id)
        event_response = create_google_calendar_event(data, access_token)
        
        # Save to database
        meeting_dict = data.dict()
        db_record = save_to_supabase(meeting_dict, event_response)
        
        # Send emails if attendees exist
        if data.attendees:
            email_content = create_meeting_invite_email(meeting_dict, event_response)
            attendee_emails = [str(email) for email in data.attendees]
            send_email(
                to_emails=attendee_emails,
                subject=f"Meeting Invitation: {data.subject}",
                body=email_content,
                is_html=True
            )
            
            # Log notification
            log_meeting_notification(
                meeting_id=db_record["id"],
                notification_type="created",
                sent_to=attendee_emails
            )
        
        # Return response with meet link
        meet_link = event_response.get("hangoutLink") or event_response.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri", "")
        
        return {
            "message": "Meeting created and invites sent successfully",
            "event": event_response,
            "db_record": db_record,
            "meet_link": meet_link
        }

    except BackendError as he:
        raise he
    except Exception as e:
        raise BackendError(status_code=500, detail=f"Error creating meeting: {str(e)}")

def update_meeting(meeting_id: int, data: MeetingData, user_id: str = "demo-user-id") -> dict:
    """Update an existing meeting"""
    try:
        # Get existing meeting from Supabase with attendees
        meeting_record = config.supabase_client.table("meetings").select("*").eq("id", meeting_id).single().execute()
        if not meeting_record.data:
            raise BackendError(status_code=404, detail="Meeting not found")

        # Get current attendees
        current_attendees_response = config.supabase_client.table("meeting_attendees").select("email").eq("meeting_id", meeting_id).execute()
        current_attendees = [att["email"] for att in current_attendees_response.data] if current_attendees_response.data else []

        # Get current meeting link
        current_link_response = config.supabase_client.table("meeting_links").select("meet_link").eq("meeting_id", meeting_id).eq("is_active", True).execute()
        current_meet_link = current_link_response.data[0]["meet_link"] if current_link_response.data else ""

        old_meeting_data = {
            **meeting_record.data,
            "attendees": current_attendees
        }

        google_event_id = meeting_record.data.get("google_event_id")
        if not google_event_id:
            raise BackendError(status_code=404, detail="Google event ID not found")

        access_token = get_google_access_token(user_id)
        updated_event = update_google_calendar_event(google_event_id, data, access_token)

        # Update Supabase record
        update_fields = {
            "subject": data.subject,
            "start_datetime": data.start_datetime,
            "end_datetime": data.end_datetime,
            "description": data.description,
            "updated_at": datetime.utcnow().isoformat()
        }

        config.supabase_client.table("meetings").update(update_fields).eq("id", meeting_id).execute()

        # Update attendees
        if data.attendees:
            # Delete existing attendees
            config.supabase_client.table("meeting_attendees").delete().eq("meeting_id", meeting_id).execute()
            
            # Insert new attendees
            attendee_records = [
                {
                    "meeting_id": meeting_id,
                    "email": str(email),
                    "response_status": "needsAction"
                }
                for email in data.attendees
            ]
            config.supabase_client.table("meeting_attendees").insert(attendee_records).execute()

        # Generate changes summary
        new_meeting_data = data.dict()
        changes_summary = generate_changes_summary(old_meeting_data, new_meeting_data)

        # Send update notifications to all attendees (both old and new)
        all_attendees = set(current_attendees + [str(email) for email in (data.attendees or [])])
        
        if all_attendees:
            email_content = create_meeting_update_email(
                old_meeting=old_meeting_data,
                new_meeting=new_meeting_data,
                changes_summary=changes_summary,
                meet_link=current_meet_link
            )
            
            attendee_emails = list(all_attendees)
            send_email(
                to_emails=attendee_emails,
                subject=f"Meeting Updated: {data.subject}",
                body=email_content,
                is_html=True
            )
            
            # Log notification
            log_meeting_notification(
                meeting_id=meeting_id,
                notification_type="updated",
                sent_to=attendee_emails,
                changes_summary=changes_summary
            )

        return {
            "message": "Meeting updated successfully and notifications sent",
            "event": updated_event,
            "changes_summary": changes_summary,
            "notifications_sent_to": list(all_attendees) if all_attendees else []
        }

    except Exception as e:
        raise BackendError(status_code=500, detail=f"Error updating meeting: {str(e)}")

def get_upcoming_meetings(user_email: str, limit: int = 10) -> List[MeetingDisplay]:
    """Get upcoming meetings"""
    try:
        if not config.supabase_client:
            raise BackendError(status_code=503, detail="Database service unavailable")
            
        now = datetime.utcnow().isoformat()
        week_later = (datetime.utcnow() + timedelta(days=30)).isoformat()

        # Get meetings with their links and attendees
        response = config.supabase_client.table("meetings") \
            .select("*, meeting_links(meet_link), meeting_attendees(email)") \
            .eq("user_email", user_email) \
            .gte("start_datetime", now) \
            .lte("start_datetime", week_later) \
            .order("start_datetime") \
            .limit(limit) \
            .execute()

        meetings = []
        for m in response.data or []:
            meet_link = ""
            if m.get("meeting_links"):
                meet_link = m["meeting_links"][0].get("meet_link", "") if m["meeting_links"] else ""
            
            attendees = []
            if m.get("meeting_attendees"):
                attendees = [att["email"] for att in m["meeting_attendees"]]
            
            meetings.append(MeetingDisplay(
                id=str(m["id"]),
                subject=m["subject"],
                start_datetime=m["start_datetime"],
                end_datetime=m["end_datetime"],
                description=m.get("description", ""),
                meet_link=meet_link,
                attendees=attendees,
                status=m.get("status", "scheduled")
            ))

        return meetings

    except Exception as e:
        raise BackendError(status_code=500, detail=f"Error fetching meetings: {str(e)}")

def get_calendar_events(user_email: str, start_date: str, end_date: str) -> List[CalendarEvent]:
    """Get meetings in calendar format for calendar view"""
    try:
        if not config.supabase_client:
            raise BackendError(status_code=503, detail="Database service unavailable")
            
        response = config.supabase_client.table("meetings") \
            .select("*, meeting_links(meet_link), meeting_attendees(email)") \
            .eq("user_email", user_email) \
            .gte("start_datetime", start_date) \
            .lte("start_datetime", end_date) \
            .order("start_datetime") \
            .execute()

        events = []
        for m in response.data or []:
            meet_link = ""
            if m.get("meeting_links"):
                meet_link = m["meeting_links"][0].get("meet_link", "") if m["meeting_links"] else ""
            
            attendees = []
            if m.get("meeting_attendees"):
                attendees = [att["email"] for att in m["meeting_attendees"]]
            
            events.append(CalendarEvent(
                id=str(m["id"]),
                title=m["subject"],
                start=m["start_datetime"],
                end=m["end_datetime"],
                description=m.get("description", ""),
                meet_link=meet_link,
                attendees=attendees
            ))

        return events

    except Exception as e:
        raise BackendError(status_code=500, detail=f"Error fetching calendar events: {str(e)}")

# ------------------- Calendar Service Functions -------------------

def get_date_events(date: str, user_email: str) -> DateEventsResponse:
    """Get all meetings and tasks for a specific date"""
    try:
        if not config.supabase_client:
            raise BackendError(status_code=503, detail="Database service unavailable")
            
        # Parse date and create date range for the whole day
        target_date = datetime.fromisoformat(date)
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        print(f"📅 Fetching events for {date} ({start_of_day} to {end_of_day})")

        # Fetch meetings for the date
        meetings_response = config.supabase_client.table("meetings") \
            .select("*, meeting_links(meet_link), meeting_attendees(email)") \
            .eq("user_email", user_email) \
            .gte("start_datetime", start_of_day.isoformat()) \
            .lte("start_datetime", end_of_day.isoformat()) \
            .order("start_datetime") \
            .execute()

        meetings = []
        for m in meetings_response.data or []:
            meet_link = ""
            if m.get("meeting_links"):
                meet_link = m["meeting_links"][0].get("meet_link", "") if m["meeting_links"] else ""
            
            attendees = []
            if m.get("meeting_attendees"):
                attendees = [att["email"] for att in m["meeting_attendees"]]
            
            meetings.append(MeetingDisplay(
                id=str(m["id"]),
                subject=m["subject"],
                start_datetime=m["start_datetime"],
                end_datetime=m["end_datetime"],
                description=m.get("description", ""),
                meet_link=meet_link,
                attendees=attendees,
                status=m.get("status", "scheduled")
            ))

        # Fetch tasks for the date
        tasks_response = config.supabase_client.table("tasks") \
            .select("*") \
            .eq("user_email", user_email) \
            .gte("due_datetime", start_of_day.isoformat()) \
            .lte("due_datetime", end_of_day.isoformat()) \
            .order("due_datetime") \
            .execute()

        tasks = []
        for t in tasks_response.data or []:
            tasks.append(TaskDisplay(
                id=t["id"],
                title=t["title"],
                description=t.get("description", ""),
                due_datetime=t.get("due_datetime"),
                user_email=t["user_email"],
                status=t.get("status", "pending")
            ))

        print(f"✅ Found {len(meetings)} meetings and {len(tasks)} tasks for {date}")

        return DateEventsResponse(
            date=date,
            meetings=meetings,
            tasks=tasks
        )

    except Exception as e:
        print(f"❌ Error fetching date events: {str(e)}")
        raise BackendError(status_code=500, detail=f"Error fetching date events: {str(e)}")

def quick_create_event(data: QuickCreateData, user_id: str = "demo-user-id") -> dict:
    """Quick create meeting or task for a specific date"""
    try:
        if not config.supabase_client:
            raise BackendError(status_code=503, detail="Database service unavailable")
            
        if data.type == "meeting":
            # Create quick meeting
            if not data.time:
                raise BackendError(status_code=400, detail="Time is required for meetings")
            
            # Parse date and time
            meeting_date = datetime.fromisoformat(data.date).date()
            start_time = datetime.strptime(data.time, "%H:%M").time()
            start_datetime = datetime.combine(meeting_date, start_time)
            end_datetime = start_datetime + timedelta(minutes=data.duration)
            
            # Create meeting data
            meeting_data = MeetingData(
                subject=data.title,
                start_datetime=start_datetime.isoformat(),
                end_datetime=end_datetime.isoformat(),
                description="",
                attendees=[],
                user_email=data.user_email
            )
            
            # Create Google Calendar event
            access_token = get_google_access_token(user_id)
            event_response = create_google_calendar_event(meeting_data, access_token)
            
            # Save to database
            meeting_dict = meeting_data.dict()
            db_record = save_to_supabase(meeting_dict, event_response)
            
            meet_link = event_response.get("hangoutLink") or event_response.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri", "")
            
            return {
                "type": "meeting",
                "message": "Meeting created successfully",
                "data": {
                    "id": db_record["id"],
                    "subject": data.title,
                    "start_datetime": start_datetime.isoformat(),
                    "end_datetime": end_datetime.isoformat(),
                    "meet_link": meet_link
                }
            }
            
        elif data.type == "task":
            # Create quick task
            due_datetime = None
            if data.time:
                task_date = datetime.fromisoformat(data.date).date()
                due_time = datetime.strptime(data.time, "%H:%M").time()
                due_datetime = datetime.combine(task_date, due_time).isoformat()
            else:
                # Default to end of day if no time specified
                task_date = datetime.fromisoformat(data.date).date()
                due_datetime = datetime.combine(task_date, datetime.max.time().replace(microsecond=0)).isoformat()
            
            task_data = {
                "title": data.title,
                "description": "",
                "user_email": data.user_email,
                "due_datetime": due_datetime,
                "status": "pending"
            }
            
            # Validate and clean task data
            validated_task_data = validate_task_data(task_data)
            validated_task_data["created_at"] = datetime.utcnow().isoformat()
            
            response = config.supabase_client.table("tasks").insert(validated_task_data).execute()
            if not response.data:
                raise Exception("Failed to create task")
            
            return {
                "type": "task",
                "message": "Task created successfully",
                "data": response.data[0]
            }
        
        else:
            raise BackendError(status_code=400, detail="Invalid type. Must be 'meeting' or 'task'")

    except Exception as e:
        print(f"❌ Error creating quick event: {str(e)}")
        raise BackendError(status_code=500, detail=f"Error creating event: {str(e)}")

def get_month_overview(year: int, month: int, user_email: str) -> dict:
    """Get overview of meetings and tasks for a specific month"""
    try:
        if not config.supabase_client:
            raise BackendError(status_code=503, detail="Database service unavailable")
            
        # Get first and last day of month
        first_day = datetime(year, month, 1)
        last_day_num = calendar.monthrange(year, month)[1]
        last_day = datetime(year, month, last_day_num, 23, 59, 59)
        
        print(f"📅 Fetching month overview for {year}-{month}")

        # Fetch meetings for the month
        meetings_response = config.supabase_client.table("meetings") \
            .select("id, subject, start_datetime, end_datetime") \
            .eq("user_email", user_email) \
            .gte("start_datetime", first_day.isoformat()) \
            .lte("start_datetime", last_day.isoformat()) \
            .execute()
        
        # Fetch tasks for the month
        tasks_response = config.supabase_client.table("tasks") \
            .select("id, title, due_datetime, status") \
            .eq("user_email", user_email) \
            .gte("due_datetime", first_day.isoformat()) \
            .lte("due_datetime", last_day.isoformat()) \
            .execute()
        
        # Group by date
        calendar_data = {}
        
        # Process meetings
        for meeting in meetings_response.data or []:
            meeting_date = datetime.fromisoformat(meeting["start_datetime"].replace('Z', '+00:00')).date().isoformat()
            if meeting_date not in calendar_data:
                calendar_data[meeting_date] = {"meetings": [], "tasks": []}
            calendar_data[meeting_date]["meetings"].append({
                "id": meeting["id"],
                "title": meeting["subject"],
                "start_time": datetime.fromisoformat(meeting["start_datetime"].replace('Z', '+00:00')).strftime("%H:%M")
            })
        
        # Process tasks
        for task in tasks_response.data or []:
            if task.get("due_datetime"):
                task_date = datetime.fromisoformat(task["due_datetime"].replace('Z', '+00:00')).date().isoformat()
                if task_date not in calendar_data:
                    calendar_data[task_date] = {"meetings": [], "tasks": []}
                calendar_data[task_date]["tasks"].append({
                    "id": task["id"],
                    "title": task["title"],
                    "status": task.get("status", "pending")
                })
        
        print(f"✅ Month overview: {len(calendar_data)} days with events")
        
        return {
            "year": year,
            "month": month,
            "calendar_data": calendar_data
        }

    except Exception as e:
        print(f"❌ Error fetching month overview: {str(e)}")
        raise BackendError(status_code=500, detail=f"Error fetching month overview: {str(e)}")

# ------------------- Backend Service Class -------------------

class BackendService:
    """Main backend service class that encapsulates all functionality"""
    
    def __init__(self):
        self.config = config
        print("🚀 Starting Enhanced Meeting Management Backend...")
        print(f"📊 Supabase connected: {config.supabase_client is not None}")
        
        if config.supabase_client:
            print("✅ Backend ready!")
        else:
            print("⚠️ Backend started but Supabase not connected - some features may not work")
        
        create_tables()
    
    # Health endpoints
    def get_health(self):
        return {
            "message": "Enhanced Meeting Management Backend is running",
            "timestamp": datetime.utcnow().isoformat(),
            "supabase_connected": config.supabase_client is not None
        }
    
    def get_health_check(self):
        return health_check()
    
    # Task endpoints
    def create_task(self, task_data: dict):
        try:
            task = TaskCreate(**task_data)
            return create_task(task)
        except ValidationError as e:
            raise BackendError(status_code=400, detail=str(e))
    
    def update_task(self, task_id: int, task_data: dict):
        try:
            task = TaskUpdate(**task_data)
            return update_task(task_id, task)
        except ValidationError as e:
            raise BackendError(status_code=400, detail=str(e))
    
    def delete_task(self, task_id: int):
        return delete_task(task_id)
    
    def list_tasks(self, user_email: str, date: Optional[str] = None, status: Optional[str] = None):
        return list_tasks(user_email, date, status)
    
    # Meeting endpoints
    def create_meeting(self, meeting_data: dict, user_id: str = "demo-user-id"):
        try:
            data = MeetingData(**meeting_data)
            return create_meeting(data, user_id)
        except ValidationError as e:
            raise BackendError(status_code=400, detail=str(e))
    
    def update_meeting(self, meeting_id: int, meeting_data: dict, user_id: str = "demo-user-id"):
        try:
            data = MeetingData(**meeting_data)
            return update_meeting(meeting_id, data, user_id)
        except ValidationError as e:
            raise BackendError(status_code=400, detail=str(e))
    
    def get_upcoming_meetings(self, user_email: str, limit: int = 10):
        return get_upcoming_meetings(user_email, limit)
    
    def get_calendar_events(self, user_email: str, start_date: str, end_date: str):
        return get_calendar_events(user_email, start_date, end_date)
    
    # Calendar endpoints
    def get_date_events(self, date: str, user_email: str):
        return get_date_events(date, user_email)
    
    def quick_create_event(self, quick_data: dict, user_id: str = "demo-user-id"):
        try:
            data = QuickCreateData(**quick_data)
            return quick_create_event(data, user_id)
        except ValidationError as e:
            raise BackendError(status_code=400, detail=str(e))
    
    def get_month_overview(self, year: int, month: int, user_email: str):
        return get_month_overview(year, month, user_email)

# ------------------- Initialize Backend Service -------------------

# Global backend service instance
backend_service = BackendService()
