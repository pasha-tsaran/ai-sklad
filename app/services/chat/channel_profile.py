"""Nextcloud identity and employee-confirmed device; never an OS identity guess."""

import re
import unicodedata


def _safe(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        return None
    if any(unicodedata.category(char) == "Cc" for char in value):
        return None
    return value.strip()


def update_nextcloud_profile(room, observed):
    if not isinstance(observed, dict) or observed.get("user_id") != room.employee_id:
        return False
    saved = dict(room.profile or {})
    saved.update(nextcloud_user_id=room.employee_id,
                 nextcloud_display_name=_safe(observed.get("display_name"), 255) or "",
                 identity_source="nextcloud_participants")
    room.profile = saved
    return True


def channel_profile(room, incident_id):
    saved = room.profile or {}
    result = {"account_login": room.employee_id}
    if (saved.get("identity_source") == "nextcloud_participants"
            and saved.get("nextcloud_user_id") == room.employee_id):
        if name := _safe(saved.get("nextcloud_display_name"), 255):
            result["account_holder_name"] = name
    if saved.get("device_incident_id") == incident_id:
        device = _safe(saved.get("device_name"), 63)
        if device and re.fullmatch(r"[A-Za-z0-9_.-]+", device) and any(char.isalnum() for char in device):
            result["computer_name"] = device
    return result


def remember_computer(room, incident_id, name):
    name = _safe(name, 63)
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not any(char.isalnum() for char in name):
        return False
    room.profile = {**(room.profile or {}), "device_name": name, "device_incident_id": incident_id,
                    "device_source": "employee_in_current_incident"}
    return True
