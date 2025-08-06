# test_pure_backend.py

import streamlit as st
from datetime import datetime, timedelta, date
import re
import calendar
import pandas as pd
from typing import Dict, Any, List
import sys
import os

# Add the current directory to Python path to import the backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the pure Python backend
try:
    from Google_backend import backend_service, BackendError
    st.success("✅ Backend service imported successfully!")
except ImportError as e:
    st.error(f"❌ Failed to import backend service: {e}")
    st.stop()

st.set_page_config(layout="wide", page_title="CRM - Meeting & Task Management", page_icon="📅")

# Custom CSS for better styling (same as before)
st.markdown("""
<style>
    /* Cards */
    .meeting-card, .task-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .meeting-card {
        border-left: 4px solid #4285f4;
    }

    .task-card {
        border-left: 4px solid #34a853;
    }

    .meeting-title, .task-title {
        font-weight: bold;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }

    .meeting-time, .task-due {
        color: #5f6368;
        font-size: 0.9rem;
    }

    /* Buttons */
    .meet-link-button, .task-button {
        color: white;
        text-decoration: none;
        display: inline-block;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }

    .meet-link-button {
        background-color: #4285f4;
        padding: 0.5rem 1rem;
    }

    .task-button {
        background-color: #34a853;
        padding: 0.3rem 0.8rem;
        border-radius: 3px;
    }

    /* Calendar Day Cell */
    .calendar-day {
        border: 1px solid #e0e0e0;
        padding: 6px;
        min-height: 100px;
        max-height: 140px;
        overflow-y: auto;
        background: white;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s, box-shadow 0.2s;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        box-sizing: border-box;
    }

    .calendar-day:hover {
        background-color: #f8f9fa;
        border-color: #4285f4;
    }

    .calendar-day-selected {
        background-color: #e3f2fd;
        border-color: #4285f4;
        box-shadow: 0 2px 8px rgba(66, 133, 244, 0.3);
    }

    /* Events and Tasks */
    .calendar-event, .calendar-task {
        padding: 3px 6px;
        margin: 2px 0;
        border-radius: 4px;
        font-size: 0.75rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        width: 100%;
        box-sizing: border-box;
        cursor: pointer;
    }

    .calendar-event {
        background-color: #4285f4;
        color: white;
    }

    .calendar-task {
        background-color: #34a853;
        color: white;
    }

    /* Scrollbar for overflow */
    .calendar-day::-webkit-scrollbar {
        width: 4px;
    }

    .calendar-day::-webkit-scrollbar-thumb {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 2px;
    }

    /* Notifications & States */
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }

    .update-notification {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
    }

    .overdue-task {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }

    .completed-task {
        background-color: #e2e3e5;
        border-left-color: #6c757d;
        opacity: 0.7;
    }

    /* Panels and Headers */
    .quick-action-panel {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }

    .date-header {
        background: linear-gradient(135deg, #4285f4, #34a853);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }

    .empty-state {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_email' not in st.session_state:
    st.session_state.user_email = "user@example.com"
if 'current_view' not in st.session_state:
    st.session_state.current_view = "create"
if 'selected_meeting' not in st.session_state:
    st.session_state.selected_meeting = None
if 'selected_task' not in st.session_state:
    st.session_state.selected_task = None
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'task_edit_mode' not in st.session_state:
    st.session_state.task_edit_mode = False
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = None
if 'show_date_details' not in st.session_state:
    st.session_state.show_date_details = False
if 'quick_create_mode' not in st.session_state:
    st.session_state.quick_create_mode = False
if 'update_notifications' not in st.session_state:
    st.session_state.update_notifications = []

# Helper functions
def format_datetime_display(dt_str: str) -> str:
    """Format datetime string for display"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%d-%m-%y %I:%M %p")
    except:
        return dt_str

def format_date_display(dt_str: str) -> str:
    """Format date string for display"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%d-%m-%y")
    except:
        return dt_str

def format_time_display(dt_str: str) -> str:
    """Format time string for display"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%I:%M %p")
    except:
        return dt_str

def is_task_overdue(due_datetime: str) -> bool:
    """Check if task is overdue"""
    try:
        due_dt = datetime.fromisoformat(due_datetime.replace('Z', '+00:00'))
        return due_dt < datetime.now()
    except:
        return False

def get_tasks(user_email: str, date_filter: str = None):
    """Fetch tasks from backend"""
    try:
        return backend_service.list_tasks(user_email, date_filter)
    except BackendError as e:
        st.error(f"Error fetching tasks: {e.detail}")
        return []
    except Exception as e:
        st.error(f"Error fetching tasks: {e}")
        return []

def get_date_events(selected_date: str, user_email: str):
    """Fetch events for a specific date"""
    try:
        return backend_service.get_date_events(selected_date, user_email)
    except BackendError as e:
        st.error(f"Error fetching date events: {e.detail}")
        return {"date": selected_date, "meetings": [], "tasks": []}
    except Exception as e:
        st.error(f"Error fetching date events: {e}")
        return {"date": selected_date, "meetings": [], "tasks": []}

def quick_create_event(date_str: str, event_type: str, title: str, time_str: str = None, user_email: str = None):
    """Quick create event for a specific date"""
    try:
        data = {
            "date": date_str,
            "type": event_type,
            "title": title,
            "user_email": user_email or st.session_state.user_email
        }
        
        if time_str:
            data["time"] = time_str
        
        return backend_service.quick_create_event(data, "demo-user-id")
    except BackendError as e:
        st.error(f"Failed to create {event_type}: {e.detail}")
        return None
    except Exception as e:
        st.error(f"Error creating {event_type}: {e}")
        return None

def get_month_overview(year: int, month: int, user_email: str):
    """Get month overview from backend"""
    try:
        return backend_service.get_month_overview(year, month, user_email)
    except BackendError as e:
        st.error(f"Error fetching month overview: {e.detail}")
        return {"year": year, "month": month, "calendar_data": {}}
    except Exception as e:
        st.error(f"Error fetching month overview: {e}")
        return {"year": year, "month": month, "calendar_data": {}}

def create_interactive_calendar_view(meetings: List[Dict], tasks: List[Dict], year: int, month: int, user_email: str):
    """Create an interactive calendar view"""
    cal = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]
    
    # Get month overview data
    month_data = get_month_overview(year, month, user_email)
    calendar_data = month_data.get("calendar_data", {})
    
    st.markdown(f"### 📅 {month_name} {year}")
    st.markdown("*Click on any date to view details or add new events*")
    
    # Create calendar grid
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Header
    cols = st.columns(7)
    for i, day in enumerate(days):
        cols[i].markdown(f"**{day}**")
    
    # Calendar days
    for week in cal:
        cols = st.columns(7)
        for i, day in enumerate(week):
            with cols[i]:
                if day == 0:
                    st.markdown('<div class="calendar-day" style="opacity: 0.3; cursor: default;"></div>', unsafe_allow_html=True)
                else:
                    # Create date string
                    day_date = datetime(year, month, day).date().isoformat()
                    day_data = calendar_data.get(day_date, {"meetings": [], "tasks": []})
                    
                    # Check if this date is selected
                    is_selected = st.session_state.selected_date == day_date
                    day_class = "calendar-day-selected" if is_selected else "calendar-day"
                    
                    # Create clickable day
                    if st.button(f"{day}", key=f"day_{year}_{month}_{day}", 
                               help=f"Click to view/add events for {day_date}"):
                        st.session_state.selected_date = day_date
                        st.session_state.show_date_details = True
                        st.rerun()
                    
                    # Show events preview
                    day_content = f'<div class="{day_class}" style="margin-top: -2rem; padding-top: 2rem;">'
                    
                    # Add meetings (max 2)
                    for meeting in day_data["meetings"][:2]:
                        start_time = meeting.get("start_time", "")
                        title = meeting["title"][:12] + "..." if len(meeting["title"]) > 12 else meeting["title"]
                        day_content += f'<div class="calendar-event" title="Meeting: {meeting["title"]}">{start_time} {title}</div>'
                    
                    # Add tasks (max 2)
                    for task in day_data["tasks"][:2]:
                        task_title = task['title'][:12] + "..." if len(task['title']) > 12 else task['title']
                        status_icon = "✅" if task["status"] == "completed" else "📋"
                        day_content += f'<div class="calendar-task" title="Task: {task["title"]}">{status_icon} {task_title}</div>'
                    
                    # Show count if more items exist
                    total_items = len(day_data["meetings"]) + len(day_data["tasks"])
                    if total_items > 4:
                        day_content += f'<div style="font-size: 0.7rem; color: #666; text-align: center;">+{total_items - 4} more</div>'
                    
                    day_content += '</div>'
                    st.markdown(day_content, unsafe_allow_html=True)

def show_date_details_modal():
    """Show detailed view for selected date"""
    if not st.session_state.show_date_details or not st.session_state.selected_date:
        return
    
    selected_date = st.session_state.selected_date
    date_obj = datetime.fromisoformat(selected_date).date()
    
    st.markdown("---")
    
    # Date header with close button
    col_header, col_close = st.columns([5, 1])
    with col_header:
        st.markdown(f'<div class="date-header"><h3>📅 {date_obj.strftime("%A, %B %d, %Y")}</h3></div>', 
                   unsafe_allow_html=True)
    with col_close:
        if st.button("❌ Close", key="close_date_details"):
            st.session_state.show_date_details = False
            st.session_state.selected_date = None
            st.session_state.quick_create_mode = False
            st.rerun()
    
    # Get events for the selected date
    date_events = get_date_events(selected_date, st.session_state.user_email)
    meetings = date_events.get("meetings", [])
    tasks = date_events.get("tasks", [])
    
    # Quick create panel
    if not st.session_state.quick_create_mode:
        col_quick1, col_quick2 = st.columns(2)
        with col_quick1:
            if st.button("➕ Quick Add Meeting", use_container_width=True):
                st.session_state.quick_create_mode = "meeting"
                st.rerun()
        with col_quick2:
            if st.button("📋 Quick Add Task", use_container_width=True):
                st.session_state.quick_create_mode = "task"
                st.rerun()
    
    # Quick create form
    if st.session_state.quick_create_mode:
        create_type = st.session_state.quick_create_mode
        
        st.markdown(f'<div class="quick-action-panel">', unsafe_allow_html=True)
        st.markdown(f"### ➕ Quick Add {create_type.title()}")
        
        with st.form(f"quick_create_{create_type}_form"):
            col_form1, col_form2 = st.columns([2, 1])
            
            with col_form1:
                quick_title = st.text_input(f"{create_type.title()} Title *", 
                                          placeholder=f"Enter {create_type} title")
            
            with col_form2:
                if create_type == "meeting":
                    quick_time = st.time_input("Start Time", value=datetime.now().replace(second=0, microsecond=0).time())
                    quick_duration = st.selectbox("Duration (minutes)", [30, 60, 90, 120], index=1)
                else:
                    quick_time = st.time_input("Due Time (optional)", value=None)
            
            col_cancel, col_create = st.columns([1, 1])
            with col_cancel:
                cancel_quick = st.form_submit_button("❌ Cancel")
            with col_create:
                create_quick = st.form_submit_button(f"✨ Create {create_type.title()}", type="primary")
            
            if cancel_quick:
                st.session_state.quick_create_mode = False
                st.rerun()
            
            elif create_quick:
                if not quick_title.strip():
                    st.error(f"{create_type.title()} title is required.")
                else:
                    time_str = None
                    if quick_time:
                        time_str = quick_time.strftime("%H:%M")
                    
                    # Create the event
                    result = quick_create_event(
                        date_str=selected_date,
                        event_type=create_type,
                        title=quick_title.strip(),
                        time_str=time_str,
                        user_email=st.session_state.user_email
                    )
                    
                    if result:
                        st.success(f"✅ {create_type.title()} created successfully!")
                        st.session_state.quick_create_mode = False
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display existing events
    col_meetings, col_tasks = st.columns(2)
    
    with col_meetings:
        st.markdown("### 📅 Meetings")
        if not meetings:
            st.markdown('<div class="empty-state">📭 No meetings scheduled for this date</div>', 
                       unsafe_allow_html=True)
        else:
            for meeting in meetings:
                with st.container():
                    st.markdown('<div class="meeting-card">', unsafe_allow_html=True)
                    
                    col_info, col_actions = st.columns([3, 1])
                    
                    with col_info:
                        start_time = format_time_display(meeting.start_datetime)
                        end_time = format_time_display(meeting.end_datetime)
                        
                        st.markdown(f"**📅 {meeting.subject}**")
                        st.write(f"⏰ {start_time} - {end_time}")
                        
                        if meeting.description:
                            st.write(f"📝 {meeting.description}")
                        
                        if meeting.attendees:
                            st.write(f"👥 {', '.join(meeting.attendees)}")
                        
                        if meeting.meet_link:
                            st.markdown(f'<a href="{meeting.meet_link}" target="_blank" class="meet-link-button">🎥 Join</a>', 
                                      unsafe_allow_html=True)
                    
                    with col_actions:
                        if st.button("👁️ View", key=f"view_meeting_{meeting.id}"):
                            st.session_state.selected_meeting = meeting
                            st.session_state.edit_mode = False
                            st.session_state.show_date_details = False
                            st.rerun()
                        
                        if st.button("✏️ Edit", key=f"edit_meeting_{meeting.id}"):
                            st.session_state.selected_meeting = meeting
                            st.session_state.edit_mode = True
                            st.session_state.show_date_details = False
                            st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with col_tasks:
        st.markdown("### 📋 Tasks")
        if not tasks:
            st.markdown('<div class="empty-state">📭 No tasks due on this date</div>', 
                       unsafe_allow_html=True)
        else:
            for task in tasks:
                is_overdue = task.due_datetime and is_task_overdue(task.due_datetime)
                is_completed = task.status == 'completed'
                
                card_class = "task-card"
                if is_completed:
                    card_class += " completed-task"
                elif is_overdue:
                    card_class += " overdue-task"
                
                with st.container():
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    
                    col_info, col_actions = st.columns([3, 1])
                    
                    with col_info:
                        status_icon = "✅" if is_completed else ("⚠️" if is_overdue else "📋")
                        st.markdown(f"**{status_icon} {task.title}**")
                        
                        if task.description:
                            st.write(f"📝 {task.description}")
                        
                        if task.due_datetime:
                            due_time = format_time_display(task.due_datetime)
                            st.write(f"⏰ Due: {due_time}")
                        
                        st.write(f"📊 Status: {task.status.title()}")
                    
                    with col_actions:
                        if st.button("✏️ Edit", key=f"edit_task_date_{task.id}"):
                            st.session_state.selected_task = task
                            st.session_state.task_edit_mode = True
                            st.session_state.show_date_details = False
                            st.rerun()
                        
                        if not is_completed:
                            if st.button("✅ Done", key=f"complete_task_date_{task.id}"):
                                try:
                                    backend_service.update_task(task.id, {"status": "completed"})
                                    st.success("Task completed!")
                                    st.rerun()
                                except BackendError as e:
                                    st.error(f"Failed to complete task: {e.detail}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

def show_upcoming_meetings():
    """Display upcoming meetings in sidebar"""
    st.sidebar.title("📅 Upcoming Meetings")
    try:
        meetings = backend_service.get_upcoming_meetings(st.session_state.user_email, 5)
        
        if not meetings:
            st.sidebar.write("No upcoming meetings")
        else:
            for meeting in meetings:
                start_time = format_datetime_display(meeting.start_datetime)
                
                with st.sidebar.expander(f"{meeting.subject}", expanded=False):
                    st.write(f"**⏰ When:** {start_time}")
                    if meeting.description:
                        st.write(f"**📝 Description:** {meeting.description}")
                    
                    if meeting.meet_link:
                        st.markdown(f'<a href="{meeting.meet_link}" target="_blank" class="meet-link-button">🎥 Join Meeting</a>', unsafe_allow_html=True)
                    
                    if meeting.attendees:
                        st.write(f"**👥 Attendees:** {', '.join(meeting.attendees)}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("View", key=f"view_{meeting.id}"):
                            st.session_state.selected_meeting = meeting
                            st.session_state.edit_mode = False
                            st.rerun()
                    with col2:
                        if st.button("Edit", key=f"edit_{meeting.id}"):
                            st.session_state.selected_meeting = meeting
                            st.session_state.edit_mode = True
                            st.rerun()
    except BackendError as e:
        st.sidebar.error(f"Failed to load upcoming meetings: {e.detail}")
    except Exception as e:
        st.sidebar.error(f"Error loading meetings: {e}")

def show_upcoming_tasks():
    """Display upcoming tasks in sidebar"""
    st.sidebar.title("📋 Today's Tasks")
    try:
        today = date.today().isoformat()
        tasks = get_tasks(st.session_state.user_email, today)
        
        if not tasks:
            st.sidebar.write("No tasks for today")
        else:
            for task in tasks[:5]:  # Show max 5 tasks
                is_overdue = task.due_datetime and is_task_overdue(task.due_datetime)
                is_completed = task.status == 'completed'
                
                status_icon = "✅" if is_completed else ("⚠️" if is_overdue else "📋")
                
                with st.sidebar.expander(f"{status_icon} {task.title}", expanded=False):
                    if task.description:
                        st.write(f"**📝 Description:** {task.description}")
                    
                    if task.due_datetime:
                        due_time = format_datetime_display(task.due_datetime)
                        st.write(f"**⏰ Due:** {due_time}")
                    
                    st.write(f"**Status:** {task.status.title()}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Edit", key=f"edit_task_{task.id}"):
                            st.session_state.selected_task = task
                            st.session_state.task_edit_mode = True
                            st.rerun()
                    with col2:
                        if not is_completed and st.button("Complete", key=f"complete_task_{task.id}"):
                            # Mark task as completed
                            try:
                                backend_service.update_task(task.id, {"status": "completed"})
                                st.success("Task completed!")
                                st.rerun()
                            except BackendError as e:
                                st.error(f"Error completing task: {e.detail}")
                            except Exception as e:
                                st.error(f"Error completing task: {e}")
    except Exception as e:
        st.sidebar.error(f"Error loading tasks: {e}")

def show_update_notifications():
    """Display meeting update notifications"""
    if st.session_state.update_notifications:
        for notification in st.session_state.update_notifications:
            st.markdown(f"""
            <div class="update-notification">
                <h4>🔄 Meeting Updated</h4>
                <p><strong>{notification['subject']}</strong></p>
                <p>📋 Changes: {notification['changes']}</p>
                <p><small>⏰ Updated: {notification['timestamp']}</small></p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Clear Notifications"):
            st.session_state.update_notifications = []
            st.rerun()

def show_meeting_edit_modal():
    """Show meeting edit modal when a meeting is selected for editing"""
    if not (st.session_state.selected_meeting and st.session_state.edit_mode):
        return
    
    meeting = st.session_state.selected_meeting
    
    st.markdown("---")
    st.markdown("### ✏️ Edit Meeting")
    
    with st.form("edit_meeting_form"):
        col_close = st.columns([6, 1])
        with col_close[1]:
            if st.form_submit_button("❌ Close"):
                st.session_state.selected_meeting = None
                st.session_state.edit_mode = False
                st.rerun()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            edit_subject = st.text_input("📝 Subject *", value=meeting.subject)
            
            # Parse existing datetime
            start_dt = datetime.fromisoformat(meeting.start_datetime.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(meeting.end_datetime.replace('Z', '+00:00'))
            
            col_date, col_start, col_end = st.columns([2, 1, 1])
            with col_date:
                edit_start_date = st.date_input("📅 Date", value=start_dt.date())
            with col_start:
                edit_start_time = st.time_input("⏰ Start Time", value=start_dt.time())
            with col_end:
                edit_end_time = st.time_input("⏰ End Time", value=end_dt.time())
            
            edit_description = st.text_area("📋 Meeting Description", 
                                          value=meeting.description or '',
                                          height=120)
        
        with col2:
            st.markdown("#### 👥 Attendees")
            current_attendees = '\n'.join(meeting.attendees or [])
            edit_attendees = st.text_area("Email Addresses", 
                                        value=current_attendees,
                                        height=120,
                                        placeholder="Enter email addresses\n(one per line or comma-separated)")
            
            st.markdown("#### 📧 Update Notifications")
            st.info("All attendees (current and new) will be notified of changes via email.")
        
        # Form buttons
        col_cancel, col_update = st.columns([1, 2])
        with col_cancel:
            cancel_edit = st.form_submit_button("❌ Cancel")
        with col_update:
            update_meeting = st.form_submit_button("💾 Update Meeting & Notify Attendees", type="primary")
        
        if cancel_edit:
            st.session_state.edit_mode = False
            st.rerun()
        
        elif update_meeting:
            # Validation
            errors = []
            
            if not edit_subject.strip():
                errors.append("📝 Subject is required.")
            
            if edit_start_time >= edit_end_time:
                errors.append("⏰ End time must be after start time.")
            
            # Process attendee emails
            email_list = []
            if edit_attendees.strip():
                raw_emails = re.split(r'[,\n]', edit_attendees)
                email_list = [e.strip() for e in raw_emails if e.strip()]
                
                invalid_emails = [e for e in email_list if not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", e)]
                if invalid_emails:
                    errors.append(f"📧 Invalid email addresses: {', '.join(invalid_emails)}")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Build update data
                start_dt_new = datetime.combine(edit_start_date, edit_start_time)
                end_dt_new = datetime.combine(edit_start_date, edit_end_time)
                
                meeting_data = {
                    "subject": edit_subject.strip(),
                    "start_datetime": start_dt_new.isoformat(),
                    "end_datetime": end_dt_new.isoformat(),
                    "description": edit_description.strip(),
                    "attendees": email_list,
                    "user_email": st.session_state.user_email
                }
                
                # Send to backend
                try:
                    with st.spinner("Updating meeting and sending notifications..."):
                        data = backend_service.update_meeting(int(meeting.id), meeting_data, "demo-user-id")
                    
                    st.markdown("""
                    <div class="success-message">
                        <h4>✅ Meeting Updated Successfully!</h4>
                        <p>Your meeting has been updated and notifications have been sent to all attendees.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add notification to session state for display
                    st.session_state.update_notifications.append({
                        'subject': edit_subject,
                        'changes': data.get('changes_summary', 'Meeting details updated'),
                        'timestamp': datetime.now().strftime("%d-%m-%y %I:%M %p")
                    })
                    
                    st.markdown(f"**📋 Changes:** {data.get('changes_summary', 'Meeting updated successfully')}")
                    
                    if data.get('notifications_sent_to'):
                        st.write(f"**📧 Notifications sent to:** {', '.join(data['notifications_sent_to'])}")
                    
                    st.session_state.edit_mode = False
                    st.session_state.selected_meeting = None
                    st.rerun()
                        
                except BackendError as e:
                    st.error(f"❌ Failed to update meeting: {e.detail}")
                except Exception as e:
                    st.error(f"❌ Error updating meeting: {e}")

def show_meeting_view_modal():
    """Show meeting view modal when a meeting is selected for viewing"""
    if not (st.session_state.selected_meeting and not st.session_state.edit_mode):
        return
    
    meeting = st.session_state.selected_meeting
    
    st.markdown("---")
    st.markdown("### 👁️ Meeting Details")
    
    col_close = st.columns([6, 1])
    with col_close[1]:
        if st.button("❌ Close", key="close_meeting_view"):
            st.session_state.selected_meeting = None
            st.rerun()
    
    # Display meeting information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## 📅 {meeting.subject}")
        
        start_time = format_datetime_display(meeting.start_datetime)
        end_time = format_datetime_display(meeting.end_datetime)
        st.write(f"**⏰ When:** {start_time} - {end_time}")
        
        if meeting.description:
            st.write(f"**📝 Description:** {meeting.description}")
        
        if meeting.attendees:
            st.write(f"**👥 Attendees:** {', '.join(meeting.attendees)}")
        
        st.write(f"**📊 Status:** {meeting.status.title()}")
    
    with col2:
        if meeting.meet_link:
            st.markdown("### 🎥 Google Meet")
            st.markdown(f'<a href="{meeting.meet_link}" target="_blank" class="meet-link-button">Join Meeting</a>', 
                       unsafe_allow_html=True)
            st.code(meeting.meet_link, language=None)
        
        st.markdown("### ⚙️ Actions")
        if st.button("✏️ Edit Meeting", use_container_width=True):
            st.session_state.edit_mode = True
            st.rerun()
        
        if st.button("🗑️ Delete Meeting", use_container_width=True, type="secondary"):
            st.warning("Delete functionality would be implemented here")

def show_task_edit_modal():
    """Show task edit modal when a task is selected for editing"""
    if not (st.session_state.selected_task and st.session_state.task_edit_mode):
        return
    
    task = st.session_state.selected_task
    
    st.markdown("---")
    st.markdown("### ✏️ Edit Task")
    
    with st.form("edit_task_form"):
        col_close = st.columns([6, 1])
        with col_close[1]:
            if st.form_submit_button("❌ Close"):
                st.session_state.selected_task = None
                st.session_state.task_edit_mode = False
                st.rerun()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            edit_task_title = st.text_input("📝 Task Title *", value=task.title)
            edit_task_description = st.text_area("📋 Description", 
                                                value=task.description or '',
                                                height=120)
        
        with col2:
            # Parse existing due datetime if exists
            current_due = None
            current_due_date = None
            current_due_time = None
            
            if task.due_datetime:
                current_due = datetime.fromisoformat(task.due_datetime.replace('Z', '+00:00'))
                current_due_date = current_due.date()
                current_due_time = current_due.time()
            
            has_due_date = st.checkbox("Set Due Date", value=bool(current_due))
            
            if has_due_date:
                edit_due_date = st.date_input("📅 Due Date", 
                                            value=current_due_date or date.today())
                edit_due_time = st.time_input("⏰ Due Time", 
                                            value=current_due_time or datetime.now().replace(second=0, microsecond=0).time())
            else:
                edit_due_date = None
                edit_due_time = None
            
            edit_task_status = st.selectbox("📊 Status", 
                                          options=["pending", "completed"],
                                          index=0 if task.status == 'pending' else 1)
        
        # Form buttons
        col_cancel, col_update = st.columns([1, 1])
        with col_cancel:
            cancel_task_edit = st.form_submit_button("❌ Cancel")
        with col_update:
            update_task = st.form_submit_button("💾 Update Task", type="primary")
        
        if cancel_task_edit:
            st.session_state.task_edit_mode = False
            st.rerun()
        
        elif update_task:
            if not edit_task_title.strip():
                st.error("📝 Task title is required.")
            else:
                # Build update data
                update_data = {
                    "title": edit_task_title.strip(),
                    "description": edit_task_description.strip(),
                    "status": edit_task_status
                }
                
                if has_due_date and edit_due_date and edit_due_time:
                    due_datetime = datetime.combine(edit_due_date, edit_due_time)
                    update_data["due_datetime"] = due_datetime.isoformat()
                else:
                    update_data["due_datetime"] = None
                
                try:
                    backend_service.update_task(task.id, update_data)
                    st.success("✅ Task updated successfully!")
                    st.session_state.task_edit_mode = False
                    st.session_state.selected_task = None
                    st.rerun()
                except BackendError as e:
                    st.error(f"❌ Failed to update task: {e.detail}")
                except Exception as e:
                    st.error(f"❌ Error updating task: {e}")

# Show sidebar
show_upcoming_meetings()
show_upcoming_tasks()

# Main content
st.title("📅 CRM - Meeting & Task Management")

# Show update notifications at the top
show_update_notifications()

# Navigation tabs
tab1, tab2, tab3 = st.tabs(["✨ Create Meeting", "📋 Task Management", "📅 Interactive Calendar"])

with tab1:
    st.markdown("### Create New Meeting")
    
    # Meeting form
    with st.form(key="meeting_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            subject = st.text_input("📝 Subject *", placeholder="Enter meeting subject")
            
            # Date and time inputs
            col_date, col_start, col_end = st.columns([2, 1, 1])
            with col_date:
                start_date = st.date_input("📅 Date", value=date.today())
            with col_start:
                start_time = st.time_input("⏰ Start Time", value=datetime.now().replace(second=0, microsecond=0).time())
            with col_end:
                end_time = st.time_input("⏰ End Time", value=(datetime.now() + timedelta(hours=1)).replace(second=0, microsecond=0).time())
            
            description = st.text_area("📋 Meeting Description", height=120, 
                                     placeholder="Enter meeting agenda, notes, or additional details...")
        
        with col2:
            st.markdown("#### 👥 Attendees")
            attendees_input = st.text_area("Email Addresses", height=120,
                                         placeholder="Enter email addresses\n(one per line or comma-separated)")
            
            st.markdown("*Invitations will be sent automatically*")
        
        # Form buttons
        col_cancel, col_save = st.columns([1, 2])
        with col_cancel:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True)
        with col_save:
            save_and_invite = st.form_submit_button("💾 Create Meeting & Send Invites", 
                                                   use_container_width=True, type="primary")
        
        # Handle form submission
        if cancel:
            st.rerun()
        
        elif save_and_invite:
            # Validation
            errors = []
            
            if not subject.strip():
                errors.append("📝 Subject is required.")
            
            if start_time >= end_time:
                errors.append("⏰ End time must be after start time.")
            
            # Process attendee emails
            email_list = []
            if attendees_input.strip():
                raw_emails = re.split(r'[,\n]', attendees_input)
                email_list = [e.strip() for e in raw_emails if e.strip()]
                
                invalid_emails = [e for e in email_list if not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", e)]
                if invalid_emails:
                    errors.append(f"📧 Invalid email addresses: {', '.join(invalid_emails)}")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Build meeting data
                start_dt = datetime.combine(start_date, start_time)
                end_dt = datetime.combine(start_date, end_time)
                
                meeting_data = {
                    "subject": subject.strip(),
                    "start_datetime": start_dt.isoformat(),
                    "end_datetime": end_dt.isoformat(),
                    "description": description.strip(),
                    "attendees": email_list,
                    "user_email": st.session_state.user_email
                }
                
                # Send to backend
                try:
                    with st.spinner("Creating meeting and sending invites..."):
                        data = backend_service.create_meeting(meeting_data, "demo-user-id")
                    
                    st.markdown("""
                    <div class="success-message">
                        <h4>✅ Meeting Created Successfully!</h4>
                        <p>Your meeting has been scheduled and calendar invites have been sent.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_details, col_link = st.columns([2, 1])
                    
                    with col_details:
                        st.markdown("**📋 Meeting Details:**")
                        st.write(f"**Subject:** {subject}")
                        st.write(f"**Date & Time:** {format_datetime_display(meeting_data['start_datetime'])}")
                        if email_list:
                            st.write(f"**Invites Sent To:** {', '.join(email_list)}")
                    
                    with col_link:
                        meet_link = data.get("meet_link")
                        if meet_link:
                            st.markdown("**🎥 Google Meet:**")
                            st.markdown(f'<a href="{meet_link}" target="_blank" class="meet-link-button">Join Meeting</a>', 
                                      unsafe_allow_html=True)
                            st.code(meet_link, language=None)
                        else:
                            st.warning("Google Meet link not generated")
                    
                    st.rerun()
                        
                except BackendError as e:
                    st.error(f"❌ Failed to create meeting: {e.detail}")
                except Exception as e:
                    st.error(f"❌ Error creating meeting: {e}")

with tab2:
    st.markdown("### 📋 Task Management")
    
    # Task creation form
    with st.expander("➕ Create New Task", expanded=False):
        with st.form("task_form"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                task_title = st.text_input("📝 Task Title *", placeholder="Enter task title")
                task_description = st.text_area("📋 Description", height=100, 
                                               placeholder="Enter task description...")
            
            with col2:
                has_due_date = st.checkbox("Set Due Date")
                if has_due_date:
                    due_date = st.date_input("📅 Due Date", value=date.today())
                    due_time = st.time_input("⏰ Due Time", value=datetime.now().replace(second=0, microsecond=0).time())
                else:
                    due_date = None
                    due_time = None
            
            col_cancel, col_create = st.columns([1, 1])
            with col_cancel:
                cancel_task = st.form_submit_button("❌ Cancel")
            with col_create:
                create_task_btn = st.form_submit_button("✨ Create Task", type="primary")
            
            if create_task_btn:
                if not task_title.strip():
                    st.error("📝 Task title is required.")
                else:
                    task_data = {
                        "title": task_title.strip(),
                        "description": task_description.strip(),
                        "user_email": st.session_state.user_email
                    }
                    
                    if has_due_date and due_date:
                        due_datetime = datetime.combine(due_date, due_time)
                        task_data["due_datetime"] = due_datetime.isoformat()
                    
                    try:
                        backend_service.create_task(task_data)
                        st.success("✅ Task created successfully!")
                        st.rerun()
                    except BackendError as e:
                        st.error(f"❌ Failed to create task: {e.detail}")
                    except Exception as e:
                        st.error(f"❌ Error creating task: {e}")
    
    # Display existing tasks
    st.markdown("#### 📋 Your Tasks")
    
    # Task filters
    col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
    with col_filter1:
        task_status_filter = st.selectbox("Filter by Status", 
                                        options=["All", "Pending", "Completed"], 
                                        key="task_status_filter")
    with col_filter2:
        task_date_filter = st.selectbox("Filter by Date", 
                                      options=["All", "Today", "This Week", "Overdue"], 
                                      key="task_date_filter")
    with col_filter3:
        if st.button("🔄 Refresh Tasks"):
            st.rerun()
    
    # Fetch and display tasks
    try:
        date_param = None
        if task_date_filter == "Today":
            date_param = date.today().isoformat()
        
        all_tasks = get_tasks(st.session_state.user_email, date_param)
        
        # Apply filters
        filtered_tasks = all_tasks
        
        if task_status_filter != "All":
            status_map = {"Pending": "pending", "Completed": "completed"}
            filtered_tasks = [t for t in filtered_tasks if t.status == status_map[task_status_filter]]
        
        if task_date_filter == "Overdue":
            filtered_tasks = [t for t in filtered_tasks if t.due_datetime and is_task_overdue(t.due_datetime)]
        elif task_date_filter == "This Week":
            week_start = date.today()
            week_end = week_start + timedelta(days=7)
            filtered_tasks = [t for t in filtered_tasks if t.due_datetime and 
                            week_start.isoformat() <= t.due_datetime[:10] <= week_end.isoformat()]
        
        if not filtered_tasks:
            st.info("📭 No tasks found matching your filters.")
        else:
            for task in filtered_tasks:
                is_overdue = task.due_datetime and is_task_overdue(task.due_datetime)
                is_completed = task.status == 'completed'
                
                # Apply CSS classes based on task status
                card_class = "task-card"
                if is_completed:
                    card_class += " completed-task"
                elif is_overdue:
                    card_class += " overdue-task"
                
                with st.container():
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    
                    col_task_info, col_task_actions = st.columns([3, 1])
                    
                    with col_task_info:
                        status_icon = "✅" if is_completed else ("⚠️" if is_overdue else "📋")
                        st.markdown(f"**{status_icon} {task.title}**")
                        
                        if task.description:
                            st.write(f"📝 {task.description}")
                        
                        if task.due_datetime:
                            due_display = format_datetime_display(task.due_datetime)
                            if is_overdue and not is_completed:
                                st.markdown(f"**⚠️ Overdue:** {due_display}")
                            else:
                                st.write(f"⏰ Due: {due_display}")
                        
                        st.write(f"📊 Status: {task.status.title()}")
                    
                    with col_task_actions:
                        if st.button("✏️ Edit", key=f"edit_task_main_{task.id}"):
                            st.session_state.selected_task = task
                            st.session_state.task_edit_mode = True
                            st.rerun()
                        
                        if not is_completed:
                            if st.button("✅ Complete", key=f"complete_task_main_{task.id}"):
                                try:
                                    backend_service.update_task(task.id, {"status": "completed"})
                                    st.success("Task completed!")
                                    st.rerun()
                                except BackendError as e:
                                    st.error(f"Failed to complete task: {e.detail}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading tasks: {e}")

with tab3:
    st.markdown("### 📅 Interactive Calendar View")
    
    # Calendar controls
    col_month, col_year, col_refresh, col_today = st.columns([2, 1, 1, 1])
    
    with col_month:
        current_month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: calendar.month_name[x],
            index=datetime.now().month - 1
        )
    
    with col_year:
        current_year = st.selectbox(
            "Year",
            options=list(range(2024, 2027)),
            index=1 if datetime.now().year == 2025 else 0
        )
    
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    with col_today:
        if st.button("📅 Today", use_container_width=True):
            today = date.today()
            st.session_state.selected_date = today.isoformat()
            st.session_state.show_date_details = True
            st.rerun()
    
    # Fetch calendar events and tasks
    try:
        start_date = datetime(current_year, current_month, 1).isoformat()
        last_day = calendar.monthrange(current_year, current_month)[1]
        end_date = datetime(current_year, current_month, last_day, 23, 59, 59).isoformat()
        
        # Fetch meetings
        calendar_meetings = backend_service.get_calendar_events(
            st.session_state.user_email,
            start_date,
            end_date
        )
        
        # Fetch tasks for the month
        calendar_tasks = get_tasks(st.session_state.user_email)
        # Filter tasks for current month
        calendar_tasks = [
            t for t in calendar_tasks 
            if t.due_datetime and 
            datetime.fromisoformat(t.due_datetime.replace('Z', '+00:00')).month == current_month and
            datetime.fromisoformat(t.due_datetime.replace('Z', '+00:00')).year == current_year
        ]
        
        # Create interactive calendar view
        create_interactive_calendar_view(calendar_meetings, calendar_tasks, current_year, current_month, st.session_state.user_email)
        
        # Show date details modal if a date is selected
        show_date_details_modal()
            
    except Exception as e:
        st.error(f"Error loading calendar: {e}")

# Show modals (these will only display if the conditions are met)
show_meeting_edit_modal()
show_meeting_view_modal()
show_task_edit_modal()

# Display backend status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Backend Status")
try:
    health = backend_service.get_health_check()
    if health["status"] == "healthy":
        st.sidebar.success("✅ Backend Healthy")
    else:
        st.sidebar.warning("⚠️ Backend Degraded")
    
    st.sidebar.write(f"**Supabase:** {health['services']['supabase']}")
    
    if st.sidebar.button("🔄 Check Health"):
        st.rerun()
        
except Exception as e:
    st.sidebar.error(f"❌ Backend Error: {e}")

# Footer
st.markdown("---")
st.markdown("**📅 Pure Python CRM - Meeting & Task Management System**")
st.markdown("*Powered by Pure Python Backend + Streamlit Frontend*")