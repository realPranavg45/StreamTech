# streamlit_app.py
import streamlit as st
from pathlib import Path
import importlib.util
import sys
import ast
import inspect
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime

MODULES_FOLDER = Path(r"C:\Users\Pranav\OneDrive\Desktop\GIT")
MODULES_FOLDER.mkdir(exist_ok=True)

st.set_page_config(
    layout="wide", 
    page_title=" Python Modules Explorer",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .stButton > button {
        width: 100% !important;
        text-align: left !important;
        padding: 0.5rem 1rem !important;
        margin: 0.2rem 0 !important;
        border-radius: 8px !important;
        border: 1px solid #e1e5e9 !important;
        background: white !important;
        color: #333 !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    .sidebar .stButton > button:hover {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        transform: translateX(5px) !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Code container styling */
    .code-container {
        background: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 1rem 0;
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        gap: 1rem;
    }
    
    .stat-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        flex: 1;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* File title styling */
    .file-title {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Success/Error indicators */
    .success-indicator {
        color: #28a745;
        font-weight: bold;
    }
    
    .error-indicator {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0 2rem 0;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

####################
# Utilities
####################
def list_module_files(folder: Path) -> List[Path]:
    """Return sorted list of .py files in folder."""
    return sorted([p for p in folder.glob("*.py") if p.is_file()])

def get_file_stats(path: Path) -> Dict[str, Any]:
    """Get comprehensive file statistics."""
    stat = path.stat()
    size_kb = stat.st_size / 1024
    return {
        'size_bytes': stat.st_size,
        'size_kb': round(size_kb, 2),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'created': datetime.fromtimestamp(stat.st_ctime)
    }

def count_code_metrics(source: str) -> Dict[str, int]:
    """Count various code metrics."""
    lines = source.splitlines()  # Use splitlines() instead of split('\n')
    return {
        'total_lines': len(lines),
        'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
        'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
        'blank_lines': len([l for l in lines if not l.strip()])
    }

def read_file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def parse_functions_and_classes(source: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse source with ast to extract top-level functions and classes."""
    results = {"functions": [], "classes": []}
    try:
        tree = ast.parse(source)
    except Exception:
        return results

    module_doc = ast.get_docstring(tree)
    results["module_doc"] = module_doc or ""

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            results["functions"].append({"name": node.name})
        elif isinstance(node, ast.ClassDef):
            results["classes"].append({"name": node.name})
    
    return results

####################
# Dashboard UI
####################

# Initialize session state
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

# Main title and refresh button
col_title, col_refresh = st.columns([4, 1])

with col_title:
    st.markdown('<div class="main-title"> Python Modules Explorer</div>', unsafe_allow_html=True)

with col_refresh:
    if st.button("🔄 Refresh", key="refresh_btn"):
        st.session_state.selected_file = None
        st.rerun()

module_files = list_module_files(MODULES_FOLDER)

# Sidebar: File selection
with st.sidebar:
    st.header("📁 Modules")
    
    if not module_files:
        st.info("No Python modules found in the directory.")
        st.write("💡 Add some `.py` files to get started!")
    else:
        st.write(f"Found {len(module_files)} modules:")
        
        # Display files as clickable list in sidebar
        for file_path in module_files:
            file_stats = get_file_stats(file_path)
            
            # Truncate filename if too long for better display
            display_name = file_path.name
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."
            
            # Create a button for each file
            if st.button(
                f"{display_name}",
                key=f"sidebar_btn_{file_path.name}",
                help=f"Full name: {file_path.name}\nSize: {file_stats['size_kb']} KB\nModified: {file_stats['modified'].strftime('%Y-%m-%d %H:%M')}",
                use_container_width=True
            ):
                st.session_state.selected_file = file_path.name
                st.rerun()
        
        st.markdown("---")
        st.caption(f"🕐 **Last scan:** {time.strftime('%H:%M:%S')}")

# Main content area
if st.session_state.selected_file:
    selected_path = MODULES_FOLDER / st.session_state.selected_file
    
    if selected_path.exists():
        try:
            # Read file with different methods to ensure accuracy
            source_text = read_file_text(selected_path)
            
            # Count lines
            line_count = len(source_text.splitlines())
            
            # Get file size and other stats
            file_stats = get_file_stats(selected_path)
            
            code_metrics = count_code_metrics(source_text)
            parsed = parse_functions_and_classes(source_text)
            
            # Display file title
            st.markdown(f'<div class="file-title">📄 {st.session_state.selected_file}</div>', unsafe_allow_html=True)
            
            # Display source code - use text_area for better handling of large files
            st.markdown("### 📝 Source Code")
            st.code(source_text, language="python", line_numbers=True)
            
            # Calculate appropriate height based on line count (minimum 400px, maximum 800px)
            line_count = len(source_text.splitlines())
            text_area_height = min(max(400, line_count * 20), 800)
            
            # Display statistics below the code
            st.markdown("---")
            
            # Create statistics cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{line_count}</div>
                    <div class="stat-label">Total Lines of Code</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                func_count = len(parsed.get('functions', []))
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{func_count}</div>
                    <div class="stat-label">Functions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                class_count = len(parsed.get('classes', []))
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{class_count}</div>
                    <div class="stat-label">Classes</div>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.error("❌ Selected file no longer exists")
        st.session_state.selected_file = None
else:
    # Show info when no file is selected
    st.info("👈 Select a Python file from the sidebar to view its contents")
    st.info(f"📂 **Modules Path:** `{MODULES_FOLDER.resolve()}`")

# Footer
st.markdown("---")
