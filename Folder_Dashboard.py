# streamlit_app.py
import streamlit as st
from pathlib import Path
import ast
import time
from typing import Dict, Any, List
from datetime import datetime
import importlib.util
import sys
import io
import contextlib

st.set_page_config(
    layout="wide", 
    page_title="Python Modules Explorer",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI (your existing CSS remains the same)
st.markdown("""
<style>
    /* Your existing CSS styles here */
</style>
""", unsafe_allow_html=True)

####################
# Utilities
####################
def list_module_files(folder: Path) -> List[Path]:
    """Return sorted list of .py files in folder."""
    if not folder.exists():
        return []
    # Exclude the main streamlit app file itself
    app_file = Path(__file__).name
    return sorted([p for p in folder.glob("*.py") if p.is_file() and p.name != app_file])

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
            'created': datetime.now()
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
        return path.read_text(encoding="latin-1")

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

def execute_python_file(file_path: Path):
    """Execute a Python file and capture its output."""
    try:
        # Create a module name from the file path
        module_name = file_path.stem
        
        # Create a spec from the file location
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        
        if spec is None:
            return False, "Failed to create module spec"
            
        # Create a module from the spec
        module = importlib.util.module_from_spec(spec)
        
        # Add the module to sys.modules
        sys.modules[module_name] = module
        
        # Create a string buffer to capture output
        output_buffer = io.StringIO()
        
        # Execute the module with output captured
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            try:
                spec.loader.exec_module(module)
                return True, output_buffer.getvalue()
            except Exception as e:
                return False, f"Error during execution: {str(e)}\n\n{output_buffer.getvalue()}"
                
    except Exception as e:
        return False, f"Execution failed: {str(e)}"

####################
# Dashboard UI
####################

# Initialize session state
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'execution_output' not in st.session_state:
    st.session_state.execution_output = None
if 'execution_success' not in st.session_state:
    st.session_state.execution_success = None

# Get the current directory (where this script is located)
MODULES_FOLDER = Path(__file__).parent

# Main title and refresh button
col_title, col_refresh = st.columns([4, 1])

with col_title:
    st.markdown('<div class="main-title">Python Modules Explorer</div>', unsafe_allow_html=True)

with col_refresh:
    if st.button("🔄 Refresh", key="refresh_btn"):
        st.session_state.selected_file = None
        st.session_state.execution_output = None
        st.session_state.execution_success = None
        st.rerun()

module_files = list_module_files(MODULES_FOLDER)

# Sidebar: File selection (your existing sidebar code remains the same)
with st.sidebar:
    st.header("📁 Modules")
    
    if not module_files:
        st.info("No Python modules found in the directory.")
        st.write("💡 Add some `.py` files to get started!")
        st.markdown("---")
        st.write("**Expected structure:**")
        st.code("""
your-repo/
├── streamlit_app.py
├── example1.py
├── example2.py
└── requirements.txt
        """)
    else:
        st.write(f"Found {len(module_files)} modules:")
        
        for file_path in module_files:
            file_stats = get_file_stats(file_path)
            
            display_name = file_path.name
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."
            
            if st.button(
                f"{display_name}",
                key=f"sidebar_btn_{file_path.name}",
                help=f"Full name: {file_path.name}\nSize: {file_stats['size_kb']} KB\nModified: {file_stats['modified'].strftime('%Y-%m-%d %H:%M')}",
                use_container_width=True
            ):
                st.session_state.selected_file = file_path.name
                st.session_state.execution_output = None
                st.session_state.execution_success = None
                st.rerun()
        
        st.markdown("---")

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
            
            # Display file title and execute button
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f'<div class="file-title">📄 {st.session_state.selected_file}</div>', unsafe_allow_html=True)
            with col2:
                if st.button("▶️ Execute Module", key="execute_btn", use_container_width=True):
                    with st.spinner("Executing module..."):
                        success, output = execute_python_file(selected_path)
                        st.session_state.execution_output = output
                        st.session_state.execution_success = success
                        st.rerun()
            
            # Display execution results if available
            if st.session_state.execution_output is not None:
                st.markdown("### 🚀 Execution Results")
                if st.session_state.execution_success:
                    st.success("Execution completed successfully!")
                else:
                    st.error("Execution encountered errors!")
                
                st.text_area("Output", value=st.session_state.execution_output, height=200)
                st.markdown("---")
            
            # Display source code
            st.markdown("### 📝 Source Code")
            st.code(source_text, language="python", line_numbers=True)
            
            # Display file stats
            st.markdown("### 📊 File Statistics")
            cols = st.columns(4)
            cols[0].metric("File Size", f"{file_stats['size_kb']} KB")
            cols[1].metric("Modified", file_stats['modified'].strftime("%Y-%m-%d %H:%M"))
            cols[2].metric("Created", file_stats['created'].strftime("%Y-%m-%d %H:%M"))
            cols[3].metric("Lines of Code", line_count)
            
            # Display code metrics
            st.markdown("### 📈 Code Metrics")
            cols = st.columns(4)
            cols[0].metric("Total Lines", code_metrics['total_lines'])
            cols[1].metric("Code Lines", code_metrics['code_lines'])
            cols[2].metric("Comment Lines", code_metrics['comment_lines'])
            cols[3].metric("Blank Lines", code_metrics['blank_lines'])
            
            # Display parsed functions and classes
            if parsed['functions'] or parsed['classes']:
                st.markdown("### 🏗️ Module Structure")
                
                if parsed['functions']:
                    st.markdown("#### Functions")
                    for func in parsed['functions']:
                        st.code(f"def {func['name']}()")
                
                if parsed['classes']:
                    st.markdown("#### Classes")
                    for cls in parsed['classes']:
                        st.code(f"class {cls['name']}")
            
            if parsed['module_doc']:
                st.markdown("### 📖 Module Documentation")
                st.info(parsed['module_doc'])
            
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.error("❌ Selected file no longer exists")
        st.session_state.selected_file = None
else:
    # Show info when no file is selected
    if module_files:
        st.info("👈 Select a Python file from the sidebar to view its contents")
    
    st.info(f"📂 **Current Directory:** `{MODULES_FOLDER.resolve()}`")

# Footer
st.markdown("---")
