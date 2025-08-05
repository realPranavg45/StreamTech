# streamlit_app.py
import streamlit as st
from pathlib import Path
import os
import sys
import ast
import time
from typing import Dict, Any, List
from datetime import datetime

st.set_page_config(
    layout="wide", 
    page_title="🐍 Python Modules Explorer - Debug",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    .debug-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
    }
    
    .success-info {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
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
    
    .stat-card {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
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
    
    .file-title {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

####################
# Debug Functions
####################

def get_debug_info():
    """Get comprehensive debug information"""
    current_file = Path(__file__)
    current_dir = current_file.parent
    
    debug_info = {
        'current_file': str(current_file),
        'current_file_name': current_file.name,
        'current_dir': str(current_dir),
        'current_dir_absolute': str(current_dir.resolve()),
        'working_directory': os.getcwd(),
        'python_path': sys.path[0] if sys.path else 'Not available',
        'all_files_in_dir': [],
        'py_files_in_dir': [],
        'dir_exists': current_dir.exists(),
    }
    
    try:
        if current_dir.exists():
            all_files = list(current_dir.iterdir())
            debug_info['all_files_in_dir'] = [f.name for f in all_files if f.is_file()]
            debug_info['py_files_in_dir'] = [f.name for f in all_files if f.is_file() and f.suffix == '.py']
        else:
            debug_info['all_files_in_dir'] = ['Directory does not exist']
            debug_info['py_files_in_dir'] = ['Directory does not exist']
    except Exception as e:
        debug_info['error'] = str(e)
    
    return debug_info

def list_module_files(folder: Path) -> List[Path]:
    """Return sorted list of .py files in folder, excluding this app."""
    if not folder.exists():
        return []
    
    app_file = Path(__file__).name
    try:
        py_files = [p for p in folder.glob("*.py") if p.is_file() and p.name != app_file]
        return sorted(py_files)
    except Exception as e:
        st.error(f"Error scanning directory: {e}")
        return []

def get_file_stats(path: Path) -> Dict[str, Any]:
    """Get comprehensive file statistics."""
    try:
        stat = path.stat()
        size_kb = stat.st_size / 1024
        return {
            'size_bytes': stat.st_size,
            'size_kb': round(size_kb, 2),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'created': datetime.fromtimestamp(stat.st_ctime)
        }
    except Exception as e:
        return {
            'size_bytes': 0,
            'size_kb': 0,
            'modified': datetime.now(),
            'created': datetime.now(),
            'error': str(e)
        }

def count_code_metrics(source: str) -> Dict[str, int]:
    """Count various code metrics."""
    lines = source.splitlines()
    return {
        'total_lines': len(lines),
        'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('#')]),
        'comment_lines': len([l for l in lines if l.strip().startswith('#')]),
        'blank_lines': len([l for l in lines if not l.strip()])
    }

def read_file_text(path: Path) -> str:
    """Read file with proper encoding handling."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:
            return f"Error reading file: {e}"

def parse_functions_and_classes(source: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse source with ast to extract top-level functions and classes."""
    results = {"functions": [], "classes": []}
    try:
        tree = ast.parse(source)
        module_doc = ast.get_docstring(tree)
        results["module_doc"] = module_doc or ""

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                results["functions"].append({"name": node.name})
            elif isinstance(node, ast.ClassDef):
                results["classes"].append({"name": node.name})
    except Exception as e:
        results["error"] = str(e)
    
    return results

####################
# Main App
####################

# Initialize session state
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

# Main title
st.markdown('<div class="main-title">🐍 Python Modules Explorer (Debug Mode)</div>', unsafe_allow_html=True)

# Get debug information
debug_info = get_debug_info()

# Always show debug information at the top
st.markdown("## 🔧 Debug Information")

with st.expander("📊 System Information", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**File System Info:**")
        st.write(f"- Current file: `{debug_info['current_file_name']}`")
        st.write(f"- Current directory: `{debug_info['current_dir']}`")
        st.write(f"- Directory exists: {'✅' if debug_info['dir_exists'] else '❌'}")
        st.write(f"- Working directory: `{debug_info['working_directory']}`")
    
    with col2:
        st.write("**Files Found:**")
        st.write(f"- All files: {len(debug_info['all_files_in_dir'])}")
        st.write(f"- Python files: {len(debug_info['py_files_in_dir'])}")
        
        if debug_info['all_files_in_dir']:
            st.write("**All files in directory:**")
            for file in debug_info['all_files_in_dir']:
                st.write(f"  - {file}")
        
        if debug_info['py_files_in_dir']:
            st.write("**Python files found:**")
            for file in debug_info['py_files_in_dir']:
                st.write(f"  - {file}")

# Show current working directory
current_dir = Path(__file__).parent
MODULES_FOLDER = current_dir

st.markdown("---")

# Get module files
module_files = list_module_files(MODULES_FOLDER)

st.markdown(f"## 📁 Looking for Python files in: `{MODULES_FOLDER.resolve()}`")

if debug_info.get('error'):
    st.error(f"Error occurred: {debug_info['error']}")

# Sidebar: File selection
with st.sidebar:
    st.header("📁 Python Files")
    
    if not module_files:
        st.warning("No Python files found!")
        st.write("**Expected structure:**")
        st.code("""
your-repo/
├── streamlit_app.py    ← This file
├── example1.py        ← Your files
├── example2.py
└── requirements.txt
        """)
        
        st.write("**Debug - Files in current directory:**")
        for file in debug_info['all_files_in_dir'][:10]:
            st.write(f"- {file}")
    else:
        st.success(f"Found {len(module_files)} Python files!")
        
        # Display files as buttons
        for file_path in module_files:
            if st.button(
                f"📄 {file_path.name}",
                key=f"btn_{file_path.name}",
                use_container_width=True
            ):
                st.session_state.selected_file = file_path.name
                st.rerun()

# Main content
if st.session_state.selected_file and module_files:
    selected_path = MODULES_FOLDER / st.session_state.selected_file
    
    if selected_path.exists():
        try:
            source_text = read_file_text(selected_path)
            
            if source_text.startswith("Error reading file:"):
                st.error(source_text)
            else:
                # Display file info
                st.markdown(f'<div class="file-title">📄 {st.session_state.selected_file}</div>', unsafe_allow_html=True)
                
                # File stats
                file_stats = get_file_stats(selected_path)
                code_metrics = count_code_metrics(source_text)
                parsed = parse_functions_and_classes(source_text)
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{code_metrics['total_lines']}</div>
                        <div class="stat-label">Total Lines</div>
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
                
                # Show source code
                st.markdown("### 📝 Source Code")
                st.code(source_text, language="python", line_numbers=True)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.error("Selected file no longer exists")
        st.session_state.selected_file = None
else:
    if not module_files:
        st.info("👆 Check the debug information above to see what files are in your repository")
        st.markdown("""
        ### 🚀 Quick Fix:
        1. Make sure you have `.py` files in the same folder as `streamlit_app.py`
        2. Files should have the `.py` extension
        3. Files should not be empty
        4. Check that your repository structure matches the expected format above
        """)
    else:
        st.info("👈 Select a Python file from the sidebar to view its contents")

st.markdown("---")
st.caption("🔧 This is a debug version that shows detailed information about your file system")
