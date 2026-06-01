"""
responder.py
------------
Automated response module.
When a threat is confirmed (by the rule engine or ML classifier),
this module executes three actions within 500ms:
  1. Generate an alert (printed + stored)
  2. Log the event to a CSV file
  3. Simulate IP blocking (adds IP to a blocked set, logs the action)

Usage:
    from responder import Responder
    responder = Responder()
    responder.respond(record, detection_result)
"""

import os
import csv
import time
import threading
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
LOG_DIR        = "logs"
EVENT_LOG      = os.path.join(LOG_DIR, "events.csv")
RESPONSE_LOG   = os.path.join(LOG_DIR, "responses.csv")
BLOCKED_IP_LOG = os.path.join(LOG_DIR, "blocked_ips.csv")

# Only trigger a response for non-benign detections above this threshold
CONFIDENCE_THRESHOLD = 0.70

# ── CSV column headers ─────────────────────────────────────────────────────────
EVENT_HEADERS = [
    "timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol", "label", "confidence", "rule", "description",
    "packet_count", "byte_count", "packets_per_sec", "bytes_per_sec",
    "connection_count",
]

RESPONSE_HEADERS = [
    "timestamp", "src_ip", "label", "confidence",
    "action_alert", "action_logged", "action_ip_blocked",
    "response_time_ms",
]

BLOCKED_IP_HEADERS = [
    "timestamp", "src_ip", "reason", "label", "confidence",
]


# ── Responder ──────────────────────────────────────────────────────────────────

class Responder:
    """
    Handles all automated response actions for detected threats.
    Thread-safe: uses a lock so the dashboard and detector can call
    respond() concurrently without corrupting the log files.
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self.blocked_ips  = set()     # in-memory set of blocked IPs
        self.alert_queue  = []        # recent alerts for the dashboard to read
        self.event_count  = 0
        self._init_logs()

    # ── Log initialisation ─────────────────────────────────────────────────────

    def _init_logs(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        for path, headers in [
            (EVENT_LOG,      EVENT_HEADERS),
            (RESPONSE_LOG,   RESPONSE_HEADERS),
            (BLOCKED_IP_LOG, BLOCKED_IP_HEADERS),
        ]:
            if not os.path.exists(path):
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(headers)

    # ── Main entry point ───────────────────────────────────────────────────────

    def respond(self, record: dict, detection: dict) -> dict:
        """
        Execute the full response pipeline for a single detection result.

        Parameters
        ----------
        record    : dict — original traffic feature record
        detection : dict — result from RuleEngine.detect() or MLClassifier.predict()

        Returns
        -------
        dict summarising which actions were taken and the total response time.
        """
        start_time = time.perf_counter()
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label      = detection.get("label", "BENIGN")
        confidence = detection.get("confidence", 0.0)
        src_ip     = record.get("src_ip", "0.0.0.0")

        actions = {
            "alert_generated": False,
            "event_logged":    False,
            "ip_blocked":      False,
        }

        # Only act on confirmed threats above the confidence threshold
        if label != "BENIGN" and confidence >= CONFIDENCE_THRESHOLD:
            with self._lock:
                actions["alert_generated"] = self._generate_alert(
                    timestamp, src_ip, record, detection
                )
                actions["event_logged"] = self._log_event(
                    timestamp, record, detection
                )
                actions["ip_blocked"] = self._block_ip(
                    timestamp, src_ip, label, confidence
                )
                self.event_count += 1

        else:
            # Still log benign traffic to the event log (quieter)
            with self._lock:
                self._log_event(timestamp, record, detection)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        response_summary = {
            "timestamp":        timestamp,
            "src_ip":           src_ip,
            "label":            label,
            "confidence":       confidence,
            "actions":          actions,
            "response_time_ms": round(elapsed_ms, 3),
        }

        # Log the response summary
        if label != "BENIGN" and confidence >= CONFIDENCE_THRESHOLD:
            with self._lock:
                self._log_response(response_summary)

        return response_summary

    # ── Action 1: Alert ────────────────────────────────────────────────────────

    def _generate_alert(self, timestamp, src_ip, record, detection) -> bool:
        """Print a formatted alert and add it to the in-memory queue."""
        alert = {
            "timestamp":   timestamp,
            "src_ip":      src_ip,
            "dst_ip":      record.get("dst_ip", "?"),
            "label":       detection["label"],
            "confidence":  detection["confidence"],
            "rule":        detection.get("rule", "ML"),
            "description": detection.get("description", "ML classification"),
            "dst_port":    record.get("dst_port", "?"),
        }

        print(
            f"\n[ALERT] {timestamp} | {detection['label']} detected "
            f"(conf={detection['confidence']:.0%}) | "
            f"src={src_ip} → dst={record.get('dst_ip','?')}:{record.get('dst_port','?')} | "
            f"Rule: {detection.get('rule','ML')}"
        )

        # Keep last 200 alerts in memory for the dashboard
        self.alert_queue.append(alert)
        if len(self.alert_queue) > 200:
            self.alert_queue.pop(0)

        return True

    # ── Action 2: Event log ────────────────────────────────────────────────────

    def _log_event(self, timestamp, record, detection) -> bool:
        """Append a row to events.csv."""
        row = [
            timestamp,
            record.get("src_ip", ""),
            record.get("dst_ip", ""),
            record.get("src_port", ""),
            record.get("dst_port", ""),
            record.get("protocol", ""),
            detection.get("label", "BENIGN"),
            round(detection.get("confidence", 0.0), 4),
            detection.get("rule", ""),
            detection.get("description", ""),
            record.get("packet_count", ""),
            record.get("byte_count", ""),
            record.get("packets_per_sec", ""),
            record.get("bytes_per_sec", ""),
            record.get("connection_count", ""),
        ]
        try:
            with open(EVENT_LOG, "a", newline="") as f:
                csv.writer(f).writerow(row)
            return True
        except IOError:
            return False

    # ── Action 3: IP block simulation ─────────────────────────────────────────

    def _block_ip(self, timestamp, src_ip, label, confidence) -> bool:
        """
        Simulate blocking the source IP.
        In production this would call iptables / firewall API.
        Here we add the IP to an in-memory set and log it.
        """
        if src_ip in self.blocked_ips:
            return False   # already blocked

        self.blocked_ips.add(src_ip)
        print(f"[BLOCK] {timestamp} | IP blocked: {src_ip} (reason: {label})")

        row = [timestamp, src_ip, f"Detected as {label}", label, round(confidence, 4)]
        try:
            with open(BLOCKED_IP_LOG, "a", newline="") as f:
                csv.writer(f).writerow(row)
            return True
        except IOError:
            return False

    # ── Response summary log ───────────────────────────────────────────────────

    def _log_response(self, summary: dict):
        """Log the response action summary to responses.csv."""
        row = [
            summary["timestamp"],
            summary["src_ip"],
            summary["label"],
            round(summary["confidence"], 4),
            summary["actions"]["alert_generated"],
            summary["actions"]["event_logged"],
            summary["actions"]["ip_blocked"],
            summary["response_time_ms"],
        ]
        try:
            with open(RESPONSE_LOG, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except IOError:
            pass

    # ── Dashboard helpers ──────────────────────────────────────────────────────

    def get_recent_alerts(self, n=50) -> list:
        """Return the n most recent alerts for the dashboard."""
        return self.alert_queue[-n:]

    def get_blocked_ips(self) -> set:
        """Return the current set of blocked IPs."""
        return set(self.blocked_ips)

    def get_stats(self) -> dict:
        """Return summary statistics for the dashboard."""
        return {
            "total_threats_detected": self.event_count,
            "total_ips_blocked":      len(self.blocked_ips),
            "recent_alert_count":     len(self.alert_queue),
        }


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    responder = Responder()

    test_cases = [
        (
            {"src_ip": "192.168.1.50", "dst_ip": "10.0.0.5",
             "src_port": 54321, "dst_port": 80, "protocol": "TCP",
             "packet_count": 3000, "byte_count": 300000,
             "packets_per_sec": 30000, "bytes_per_sec": 3000000,
             "connection_count": 500},
            {"label": "DoS", "confidence": 0.92,
             "rule": "dos_syn_flood", "description": "SYN flood attack"},
        ),
        (
            {"src_ip": "192.168.2.99", "dst_ip": "10.0.0.5",
             "src_port": 61000, "dst_port": 22, "protocol": "TCP",
             "packet_count": 4, "byte_count": 240,
             "packets_per_sec": 5, "bytes_per_sec": 300,
             "connection_count": 200},
            {"label": "BruteForce", "confidence": 0.88,
             "rule": "brute_auth_port", "description": "SSH brute force"},
        ),
        (
            {"src_ip": "192.168.1.10", "dst_ip": "10.0.0.1",
             "src_port": 55000, "dst_port": 443, "protocol": "TCP",
             "packet_count": 8, "byte_count": 900,
             "packets_per_sec": 4, "bytes_per_sec": 450,
             "connection_count": 2},
            {"label": "BENIGN", "confidence": 0.0,
             "rule": None, "description": "Normal traffic"},
        ),
    ]

    print("=" * 60)
    print("Responder — Test Run")
    print("=" * 60)
    for record, detection in test_cases:
        result = responder.respond(record, detection)
        print(f"\nLabel          : {result['label']}")
        print(f"Actions taken  : {result['actions']}")
        print(f"Response time  : {result['response_time_ms']:.3f} ms")

    print(f"\nStats: {responder.get_stats()}")
    print(f"Blocked IPs: {responder.get_blocked_ips()}")
    print(f"\nLog files written to: {LOG_DIR}/")
