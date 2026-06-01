"""
detector.py
-----------
Rule-based signature detection engine.
Compares incoming network traffic features against a set of hand-crafted
rules for three attack categories:
  - DoS (Denial of Service)
  - BruteForce
  - PortScan

Each rule returns a label and a confidence score (0.0 – 1.0).
The engine applies all rules and returns the highest-confidence match.
If no rule fires, the traffic is classified as BENIGN.

Usage (standalone):
    from detector import RuleEngine
    engine = RuleEngine()
    result = engine.detect(record)   # record is a dict of feature values
"""


# ── Rule thresholds (tune these to improve accuracy) ──────────────────────────
RULES = {
    # ── DoS rules ──────────────────────────────────────────────────────────────
    # High packet rate is the strongest indicator
    "dos_packet_rate": {
        "label":       "DoS",
        "condition":   lambda r: r["packets_per_sec"] > 500,
        "confidence":  0.90,
        "description": "Packet rate exceeds 500 pkt/s — flood indicator",
    },
    "dos_byte_rate": {
        "label":       "DoS",
        "condition":   lambda r: r["bytes_per_sec"] > 100_000,
        "confidence":  0.85,
        "description": "Byte rate exceeds 100 KB/s — bandwidth flood",
    },
    "dos_syn_flood": {
        "label":       "DoS",
        "condition":   lambda r: r["syn_flag_count"] > 100,
        "confidence":  0.92,
        "description": "SYN flag count > 100 — SYN flood attack",
    },
    "dos_high_connections": {
        "label":       "DoS",
        "condition":   lambda r: r["connection_count"] > 200 and r["idle_time"] == 0,
        "confidence":  0.80,
        "description": "High connections with zero idle time — sustained flood",
    },

    # ── BruteForce rules ────────────────────────────────────────────────────────
    # Repeated short connections to authentication ports
    "brute_auth_port": {
        "label":       "BruteForce",
        "condition":   lambda r: int(r["dst_port"]) in (22, 23, 3389, 21, 5900)
                                 and r["connection_count"] > 30,
        "confidence":  0.88,
        "description": "High connection count to auth port (SSH/RDP/FTP/Telnet/VNC)",
    },
    "brute_repeated_syn_fin": {
        "label":       "BruteForce",
        "condition":   lambda r: r["syn_flag_count"] > 0
                                 and r["fin_flag_count"] > 0
                                 and r["connection_count"] > 40
                                 and int(r["dst_port"]) in (22, 23, 3389, 21, 5900),
        "confidence":  0.91,
        "description": "SYN+FIN pattern on auth port — credential brute forcing",
    },
    "brute_low_bytes_high_conn": {
        "label":       "BruteForce",
        "condition":   lambda r: r["byte_count"] < 600
                                 and r["connection_count"] > 50,
        "confidence":  0.75,
        "description": "Low bytes per connection with many connections — brute force",
    },

    # ── PortScan rules ──────────────────────────────────────────────────────────
    # Very short sessions sweeping many ports
    "scan_short_duration": {
        "label":       "PortScan",
        "condition":   lambda r: r["duration"] < 0.05
                                 and r["packet_count"] <= 2
                                 and r["connection_count"] > 100,
        "confidence":  0.89,
        "description": "Very short connections across many ports — port sweep",
    },
    "scan_no_ack": {
        "label":       "PortScan",
        "condition":   lambda r: r["syn_flag_count"] > 0
                                 and r["ack_flag_count"] == 0
                                 and r["connection_count"] > 150,
        "confidence":  0.86,
        "description": "SYN with no ACK across many connections — stealth scan",
    },
    "scan_high_connection_spread": {
        "label":       "PortScan",
        "condition":   lambda r: r["connection_count"] > 500
                                 and r["bytes_per_sec"] < 6000,
        "confidence":  0.82,
        "description": "Many connections with low bytes — port discovery scan",
    },
}


# ── Engine ─────────────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Applies all signature rules to a single network traffic record.
    Returns the highest-confidence rule match, or BENIGN if none fire.
    """

    def __init__(self):
        self.rules = RULES

    def detect(self, record: dict) -> dict:
        """
        Parameters
        ----------
        record : dict
            A single row of network traffic features.

        Returns
        -------
        dict with keys:
            label       – detected class (BENIGN / DoS / BruteForce / PortScan)
            confidence  – float 0.0–1.0
            rule        – name of the rule that fired (or None)
            description – human-readable explanation
        """
        best = {
            "label":       "BENIGN",
            "confidence":  0.0,
            "rule":        None,
            "description": "No rule matched — traffic classified as benign",
        }

        for rule_name, rule in self.rules.items():
            try:
                if rule["condition"](record):
                    if rule["confidence"] > best["confidence"]:
                        best = {
                            "label":       rule["label"],
                            "confidence":  rule["confidence"],
                            "rule":        rule_name,
                            "description": rule["description"],
                        }
            except (KeyError, TypeError, ZeroDivisionError):
                # Skip rules that fail on missing/malformed fields
                continue

        return best

    def detect_batch(self, records: list) -> list:
        """Run detect() on a list of record dicts. Returns a list of results."""
        return [self.detect(r) for r in records]

    def get_rule_summary(self) -> list:
        """Returns a list of rule descriptions for display in the dashboard."""
        return [
            {
                "rule":        name,
                "label":       rule["label"],
                "confidence":  rule["confidence"],
                "description": rule["description"],
            }
            for name, rule in self.rules.items()
        ]


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = RuleEngine()

    test_cases = [
        {
            "name": "Normal web traffic",
            "record": {
                "duration": 1.2, "packet_count": 10, "byte_count": 1200,
                "packets_per_sec": 8.0, "bytes_per_sec": 1000,
                "src_port": 55123, "dst_port": 443, "protocol": "TCP",
                "syn_flag_count": 1, "ack_flag_count": 3, "fin_flag_count": 1,
                "rst_flag_count": 0, "idle_time": 0.5, "connection_count": 2,
            },
        },
        {
            "name": "DoS flood",
            "record": {
                "duration": 0.1, "packet_count": 3000, "byte_count": 300000,
                "packets_per_sec": 30000, "bytes_per_sec": 3000000,
                "src_port": 54321, "dst_port": 80, "protocol": "TCP",
                "syn_flag_count": 1500, "ack_flag_count": 0, "fin_flag_count": 0,
                "rst_flag_count": 10, "idle_time": 0, "connection_count": 500,
            },
        },
        {
            "name": "SSH brute force",
            "record": {
                "duration": 0.8, "packet_count": 4, "byte_count": 240,
                "packets_per_sec": 5.0, "bytes_per_sec": 300,
                "src_port": 61000, "dst_port": 22, "protocol": "TCP",
                "syn_flag_count": 1, "ack_flag_count": 1, "fin_flag_count": 1,
                "rst_flag_count": 0, "idle_time": 0.1, "connection_count": 200,
            },
        },
        {
            "name": "Port scan",
            "record": {
                "duration": 0.01, "packet_count": 1, "byte_count": 60,
                "packets_per_sec": 100, "bytes_per_sec": 6000,
                "src_port": 44000, "dst_port": 8080, "protocol": "TCP",
                "syn_flag_count": 1, "ack_flag_count": 0, "fin_flag_count": 0,
                "rst_flag_count": 0, "idle_time": 0.001, "connection_count": 800,
            },
        },
    ]

    print("=" * 60)
    print("Rule-Based Detection Engine — Test Results")
    print("=" * 60)
    for tc in test_cases:
        result = engine.detect(tc["record"])
        print(f"\nInput   : {tc['name']}")
        print(f"Label   : {result['label']}")
        print(f"Confidence: {result['confidence']:.0%}")
        print(f"Rule    : {result['rule']}")
        print(f"Reason  : {result['description']}")
    print("\n" + "=" * 60)
    print(f"Total rules loaded: {len(engine.rules)}")
