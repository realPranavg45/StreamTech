# streamlit_app.py
import streamlit as st
from pathlib import Path
import ast
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import importlib.util
import sys
import io
import contextlib
import subprocess
import threading
import queue
import shlex
import os
import tempfile
import signal

st.set_page_config(
    layout="wide",
    page_title="Python Modules Explorer & Runner",
    initial_sidebar_state="expanded"
)

# ---------- Minimal CSS (customize as needed) ----------
st.markdown("""
<style>
.main-title {font-size:28px; font-weight:700; padding:6px 0;}
.file-title {font-size:20px; font-weight:600;}
</style>
""", unsafe_allow_html=True)

####################
# Utilities
####################
def list_module_files(folder: Path) -> List[Path]:
    """Return sorted list of .py files in folder, excluding this file."""
    if not folder.exists():
        return []
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
    except Exception:
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
    results = {"functions": [], "classes": [], "module_doc": ""}
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
# Execution backend (runs module in separate process)
####################
class SubprocessRunner:
    """
    Runs a python command in a separate process, captures stdout/stderr in realtime,
    supports timeout and graceful kill.
    """
    def __init__(self, cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None):
        self.cmd = cmd
        self.cwd = str(cwd) if cwd else None
        self.env = os.environ.copy()
        if env:
            self.env.update(env)
        self.proc: Optional[subprocess.Popen] = None
        self._stdout_queue = queue.Queue()
        self._stderr_queue = queue.Queue()
        self._alive = False

    def _reader_thread(self, stream, q):
        try:
            for line in iter(stream.readline, b''):
                if not line:
                    break
                q.put(line.decode(errors='replace'))
        except Exception:
            pass

    def start(self):
        # Start the subprocess
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL
        )
        self._alive = True
        # Start reader threads
        threading.Thread(target=self._reader_thread, args=(self.proc.stdout, self._stdout_queue), daemon=True).start()
        threading.Thread(target=self._reader_thread, args=(self.proc.stderr, self._stderr_queue), daemon=True).start()

    def poll_outputs(self) -> List[str]:
        """Return any new lines from stdout/stderr as a list (marked)."""
        lines = []
        try:
            while True:
                l = self._stdout_queue.get_nowait()
                lines.append(l.rstrip("\n"))
        except queue.Empty:
            pass
        try:
            while True:
                l = self._stderr_queue.get_nowait()
                lines.append(l.rstrip("\n"))
        except queue.Empty:
            pass
        return lines

    def is_running(self):
        if self.proc is None:
            return False
        return self.proc.poll() is None

    def terminate(self):
        if self.proc and self.is_running():
            try:
                # try graceful terminate
                self.proc.terminate()
                # give short time then kill
                time.sleep(1)
                if self.is_running():
                    self.proc.kill()
            except Exception:
                pass

    def wait(self, timeout: Optional[float] = None) -> int:
        if not self.proc:
            raise RuntimeError("Process not started")
        try:
            return_code = self.proc.wait(timeout=timeout)
            self._alive = False
            return return_code
        except subprocess.TimeoutExpired:
            return -1

####################
# Streamlit session-state init
####################
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'execution_lines' not in st.session_state:
    st.session_state.execution_lines = []  # list of log lines
if 'runner' not in st.session_state:
    st.session_state.runner = None
if 'last_run_pid' not in st.session_state:
    st.session_state.last_run_pid = None
if 'run_error' not in st.session_state:
    st.session_state.run_error = None

# Get the current directory (where this script is located)
MODULES_FOLDER = Path(__file__).parent

# ---------- Top bar ----------
col_title, col_refresh = st.columns([4, 1])
with col_title:
    st.markdown('<div class="main-title">Python Modules Explorer & Runner</div>', unsafe_allow_html=True)
with col_refresh:
    if st.button("🔄 Refresh", key="refresh_btn"):
        st.session_state.selected_file = None
        st.session_state.execution_lines = []
        st.session_state.run_error = None
        # Attempt to cleanup runner if any
        runner = st.session_state.runner
        if runner:
            try:
                runner.terminate()
            except Exception:
                pass
            st.session_state.runner = None
        st.rerun()

module_files = list_module_files(MODULES_FOLDER)

# ---------- Sidebar: files ----------
with st.sidebar:
    st.header("📁 Modules")
    if not module_files:
        st.info("No Python modules found in the directory.")
        st.write("Add `.py` files next to this Streamlit app to get started.")
    else:
        st.write(f"Found {len(module_files)} modules:")
        for file_path in module_files:
            file_stats = get_file_stats(file_path)
            display_name = file_path.name if len(file_path.name) <= 32 else file_path.name[:29] + "..."
            if st.button(f"{display_name}", key=f"sidebar_btn_{file_path.name}",
                         help=f"{file_path.name}\nSize: {file_stats['size_kb']} KB\nModified: {file_stats['modified'].strftime('%Y-%m-%d %H:%M')}"):
                st.session_state.selected_file = file_path.name
                st.session_state.execution_lines = []
                st.session_state.runner = None
                st.session_state.run_error = None
                st.rerun()
        st.markdown("---")
        st.write("Options")
        st.checkbox("Show line numbers in source display", value=True, key="show_line_numbers")
        st.text_input("Extra CLI args (space-separated)", value="", key="extra_args")
        st.number_input("Timeout (seconds, 0 = no timeout)", min_value=0, value=30, step=5, key="timeout_secs")
        st.checkbox("Install requirements.txt before running (runs pip install -r requirements.txt)", key="install_reqs")

# ---------- Main content ----------
if st.session_state.selected_file:
    selected_path = MODULES_FOLDER / st.session_state.selected_file
    if not selected_path.exists():
        st.error("Selected file no longer exists")
        st.session_state.selected_file = None
    else:
        try:
            source_text = read_file_text(selected_path)
            line_count = len(source_text.splitlines())
            file_stats = get_file_stats(selected_path)
            code_metrics = count_code_metrics(source_text)
            parsed = parse_functions_and_classes(source_text)

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f'<div class="file-title">📄 {st.session_state.selected_file}</div>', unsafe_allow_html=True)
            with col2:
                # Execute button
                if st.button("▶️ Execute Module", key="execute_btn"):
                    # Ensure previous runner is terminated
                    prev = st.session_state.runner
                    if prev:
                        try:
                            prev.terminate()
                        except Exception:
                            pass
                        st.session_state.runner = None
                    st.session_state.execution_lines = []
                    st.session_state.run_error = None

                    # Set up command: use same python interpreter as Streamlit process
                    python_exe = sys.executable or "python"
                    extra_args_raw = st.session_state.extra_args.strip()
                    extra_args_list = shlex.split(extra_args_raw) if extra_args_raw else []
                    cmd = [python_exe, str(selected_path)] + extra_args_list

                    # Optionally install requirements
                    if st.session_state.install_reqs:
                        req_path = MODULES_FOLDER / "requirements.txt"
                        if req_path.exists():
                            # Run pip install (blocking) before running module
                            st.session_state.execution_lines.append("[runner] Installing requirements from requirements.txt ...")
                            try:
                                # call pip as module to use same interpreter
                                install_proc = subprocess.run([python_exe, "-m", "pip", "install", "-r", str(req_path)],
                                                              cwd=str(MODULES_FOLDER),
                                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
                                st.session_state.execution_lines.extend(install_proc.stdout.splitlines())
                                if install_proc.returncode != 0:
                                    st.session_state.run_error = "pip install failed. See logs above."
                                    st.session_state.runner = None
                                    st.experimental_rerun()
                            except subprocess.TimeoutExpired:
                                st.session_state.run_error = "pip install timed out."
                                st.experimental_rerun()
                            except Exception as e:
                                st.session_state.run_error = f"pip install error: {e}"
                                st.experimental_rerun()
                        else:
                            st.session_state.execution_lines.append("[runner] No requirements.txt found; skipping install.")

                    # Create and start runner
                    runner = SubprocessRunner(cmd=cmd, cwd=MODULES_FOLDER)
                    try:
                        runner.start()
                        st.session_state.runner = runner
                        st.session_state.last_run_pid = runner.proc.pid if runner.proc else None
                        st.session_state.execution_lines.append(f"[runner] Started PID={st.session_state.last_run_pid} -> {' '.join(cmd)}")
                    except Exception as e:
                        st.session_state.run_error = f"Failed to start process: {e}"
                        st.session_state.runner = None

                    st.experimental_rerun()

            # Execution area: live logs + controls
            st.markdown("### 🚀 Execution")
            exec_col1, exec_col2, exec_col3 = st.columns([4, 1, 1])
            with exec_col1:
                # Live logs
                log_box = st.empty()
                # Render any buffered lines
                log_text = "\n".join(st.session_state.execution_lines[-1000:])
                log_box.code(log_text, language=None, line_numbers=False)

            with exec_col2:
                if st.session_state.runner and st.session_state.runner.is_running():
                    if st.button("⏹️ Stop", key="stop_btn"):
                        try:
                            st.session_state.runner.terminate()
                            st.session_state.execution_lines.append("[runner] Terminated by user.")
                        except Exception as e:
                            st.session_state.execution_lines.append(f"[runner] Error while terminating: {e}")
                        st.session_state.runner = None
                        st.experimental_rerun()
                else:
                    st.button("⏺️ Not running", disabled=True, key="not_running_btn")

            with exec_col3:
                st.write("Timeout:")
                t_val = st.session_state.timeout_secs
                st.write(f"{t_val} s" if t_val != 0 else "No timeout")

            # Poll runner frequently (non-blocking)
            runner = st.session_state.runner
            if runner:
                # collect new lines
                new_lines = runner.poll_outputs()
                if new_lines:
                    for ln in new_lines:
                        # mark whether from stderr or stdout is mixed; we already mixed them
                        st.session_state.execution_lines.append(ln)
                    # trim history to reasonable size
                    if len(st.session_state.execution_lines) > 5000:
                        st.session_state.execution_lines = st.session_state.execution_lines[-2000:]
                    st.experimental_rerun()  # re-render to show updated logs

                # Check if process finished
                if not runner.is_running():
                    rc = runner.proc.returncode if runner.proc else None
                    st.session_state.execution_lines.append(f"[runner] Process finished with return code: {rc}")
                    st.session_state.runner = None
                    st.experimental_rerun()

            if st.session_state.run_error:
                st.error(st.session_state.run_error)

            # If no runner, show latest logs in text area
            st.markdown("---")
            st.markdown("### 📝 Source Code")
            if st.session_state.show_line_numbers:
                st.code(source_text, language="python", line_numbers=True)
            else:
                st.code(source_text, language="python")

            st.markdown("### 📊 File Statistics")
            cols = st.columns(4)
            cols[0].metric("File Size", f"{file_stats['size_kb']} KB")
            cols[1].metric("Modified", file_stats['modified'].strftime("%Y-%m-%d %H:%M"))
            cols[2].metric("Created", file_stats['created'].strftime("%Y-%m-%d %H:%M"))
            cols[3].metric("Lines of Code", line_count)

            st.markdown("### 📈 Code Metrics")
            cols = st.columns(4)
            cols[0].metric("Total Lines", code_metrics['total_lines'])
            cols[1].metric("Code Lines", code_metrics['code_lines'])
            cols[2].metric("Comment Lines", code_metrics['comment_lines'])
            cols[3].metric("Blank Lines", code_metrics['blank_lines'])

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
    # No file selected view
    if module_files:
        st.info("👈 Select a Python file from the sidebar to view and run it")
    st.info(f"📂 **Current Directory:** `{MODULES_FOLDER.resolve()}`")
    st.info("Tip: Put `.py` files in the same folder as this app. Use the Install requirements option to install dependencies from `requirements.txt`.")

# Footer + housekeeping
st.markdown("---")
st.caption("⚠️ Running code executes in a subprocess on this machine. Use only with trusted code. For production, run inside a container/VM.")
