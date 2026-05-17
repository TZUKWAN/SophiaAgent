"""Process lifecycle guards for CLI and web entry points.

The web server normally runs in the foreground, but terminal close semantics
vary by shell, OS, and launcher. This module makes shutdown less fragile:

- Windows: put the current process in a Job Object configured with
  KILL_ON_JOB_CLOSE, so child processes created by Sophia die with Sophia.
- All platforms: start a lightweight parent monitor; if the launcher process
  disappears unexpectedly, Sophia exits instead of staying orphaned.
- Register signal/atexit cleanup hooks for tracked background processes.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import time
from typing import Iterable, Set

_INSTALLED = False
_CHILD_PIDS: Set[int] = set()
_JOB_HANDLE = None
_ORIGINAL_HANDLERS = {}


def install_process_lifecycle_hooks(
    *,
    monitor_parent: bool = True,
    use_windows_job: bool = True,
) -> None:
    """Install best-effort process cleanup hooks once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    if use_windows_job and os.name == "nt":
        _install_windows_job_object()
        _install_windows_console_handler()

    atexit.register(cleanup_registered_children)
    _install_signal_handlers()

    if monitor_parent:
        parent_pid = os.getppid()
        thread = threading.Thread(
            target=_monitor_parent_process,
            args=(parent_pid,),
            name="sophia-parent-monitor",
            daemon=True,
        )
        thread.start()


def register_child_process(pid: int) -> None:
    if pid and pid > 0:
        _CHILD_PIDS.add(pid)


def unregister_child_process(pid: int) -> None:
    _CHILD_PIDS.discard(pid)


def register_child_processes(pids: Iterable[int]) -> None:
    for pid in pids:
        register_child_process(pid)


def cleanup_registered_children() -> None:
    for pid in list(_CHILD_PIDS):
        _terminate_process_tree(pid)
        _CHILD_PIDS.discard(pid)


def _install_signal_handlers() -> None:
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            previous = signal.getsignal(sig)
            _ORIGINAL_HANDLERS[sig] = previous

            def _handler(signum, frame, previous=previous):
                cleanup_registered_children()
                if callable(previous):
                    previous(signum, frame)
                elif previous == signal.SIG_DFL:
                    raise SystemExit(128 + signum)

            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def _monitor_parent_process(parent_pid: int) -> None:
    if parent_pid <= 0:
        return
    if os.name == "nt":
        _wait_for_windows_parent(parent_pid)
        return

    while True:
        time.sleep(2.0)
        if os.getppid() != parent_pid:
            cleanup_registered_children()
            os._exit(0)


def _wait_for_windows_parent(parent_pid: int) -> None:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        SYNCHRONIZE = 0x00100000
        WAIT_OBJECT_0 = 0x00000000
        WAIT_FAILED = 0xFFFFFFFF
        INFINITE = 0xFFFFFFFF

        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
        if not handle:
            return
        try:
            wait_result = kernel32.WaitForSingleObject(handle, INFINITE)
            if wait_result == WAIT_OBJECT_0:
                cleanup_registered_children()
                os._exit(0)
            if wait_result == WAIT_FAILED:
                return
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return


def _install_windows_job_object() -> None:
    """Assign this process to a kill-on-close Windows Job Object."""
    global _JOB_HANDLE
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(job)
            return

        current = kernel32.GetCurrentProcess()
        if not kernel32.AssignProcessToJobObject(job, current):
            kernel32.CloseHandle(job)
            return

        _JOB_HANDLE = job
    except Exception:
        _JOB_HANDLE = None


def _install_windows_console_handler() -> None:
    """Clean up tracked children when Windows closes the console."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        CTRL_C_EVENT = 0
        CTRL_BREAK_EVENT = 1
        CTRL_CLOSE_EVENT = 2
        CTRL_LOGOFF_EVENT = 5
        CTRL_SHUTDOWN_EVENT = 6

        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _handler(ctrl_type):
            if ctrl_type in {
                CTRL_C_EVENT,
                CTRL_BREAK_EVENT,
                CTRL_CLOSE_EVENT,
                CTRL_LOGOFF_EVENT,
                CTRL_SHUTDOWN_EVENT,
            }:
                cleanup_registered_children()
                if ctrl_type in {CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT}:
                    os._exit(0)
                return False
            return False

        callback = handler_type(_handler)
        # Keep a global reference so ctypes does not garbage collect it.
        globals()["_WINDOWS_CONSOLE_HANDLER"] = callback
        kernel32.SetConsoleCtrlHandler.argtypes = [handler_type, wintypes.BOOL]
        kernel32.SetConsoleCtrlHandler.restype = wintypes.BOOL
        kernel32.SetConsoleCtrlHandler(callback, True)
    except Exception:
        return


def _terminate_process_tree(pid: int) -> None:
    if pid <= 0 or pid == os.getpid():
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
