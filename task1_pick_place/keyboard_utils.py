#!/usr/bin/env python3
"""Shared keyboard utilities for terminal-driven Task 1 tools."""

import os
import sys
import select


class RawKeyboard:
    """Persistent raw keyboard reader that decodes arrow and special keys."""

    def __init__(self):
        self._mode = None
        self._fd = None
        self._old = None
        self._tty_modules = None

        try:
            import msvcrt  # type: ignore
            self._mode = "windows"
            self._msvcrt = msvcrt
            return
        except Exception:
            self._msvcrt = None

        try:
            import tty
            import termios
            if sys.stdin.isatty():
                self._mode = "posix"
                self._fd = sys.stdin.fileno()
                self._old = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self._tty_modules = (tty, termios)
        except Exception:
            self._mode = None

    def close(self):
        if self._mode == "posix" and self._fd is not None and self._old is not None:
            _, termios = self._tty_modules
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._fd = None
            self._old = None

    def read_key(self):
        if self._mode == "windows":
            if not self._msvcrt.kbhit():
                return None
            ch = self._msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                if not self._msvcrt.kbhit():
                    return None
                nxt = self._msvcrt.getwch()
                return {
                    "H": "UP",
                    "P": "DOWN",
                    "K": "LEFT",
                    "M": "RIGHT",
                }.get(nxt, nxt)
            return self._normalize_char(ch)

        if self._mode == "posix":
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return None
            data = os.read(self._fd, 32)
            if not data:
                return None
            return self._decode_posix_bytes(data)

        return None

    def _normalize_char(self, ch):
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == "\x1b":
            return "ESC"
        if ch in ("\x08", "\x7f"):
            return "BACKSPACE"
        return ch

    def _decode_posix_bytes(self, data):
        if data.startswith(b"\x1b[A"):
            return "UP"
        if data.startswith(b"\x1b[B"):
            return "DOWN"
        if data.startswith(b"\x1b[C"):
            return "RIGHT"
        if data.startswith(b"\x1b[D"):
            return "LEFT"
        if data in (b"\r", b"\n", b"\r\n"):
            return "ENTER"
        if data in (b"\x7f", b"\x08"):
            return "BACKSPACE"
        if data == b"\x1b":
            return "ESC"
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            return None
        return text[0] if text else None
