"""Run against the API and worker started by Compose; never connects to real mail."""

import argparse
import os
import time
import uuid

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    token = os.environ.get("API_TOKEN", "local-demo-token")
    with httpx.Client(
        base_url=args.base_url, timeout=20, headers={"Authorization": f"Bearer {token}"}
    ) as client:

        def post(path, data):
            response = client.post(path, json=data, headers={"Idempotency-Key": str(uuid.uuid4())})
            response.raise_for_status()
            return response.json()

        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()

        def wait(incident_id, predicate):
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                incident = get(f"/api/incidents/{incident_id}")
                if predicate(incident):
                    return incident
                time.sleep(0.5)
            raise TimeoutError("Worker did not complete; inspect docker compose logs worker")

        response = post("/api/chat", {"message": "Принтер не печатает"})
        incident_id = response["incident_id"]
        print(f"Incident: {incident_id}")
        for ordinal in range(1, 6):
            incident = wait(incident_id, lambda value: value["state"] == "WAITING_FEEDBACK")
            attempt = incident["attempts"][-1]
            assert attempt["ordinal"] == ordinal
            print(f"Attempt {ordinal}: {attempt['evidence']['title']}")
            post(f"/api/incidents/{incident_id}/feedback", {"attempt_id": attempt["id"], "helped": False})
        incident = wait(incident_id, lambda value: value["pending_action"] == "COLLECT_ESCALATION_DETAILS")
        assert incident["failed_attempts"] == 5
        assert len(incident["attempts"]) == 5
        print("Five failures: escalation form requested")
        post(
            f"/api/incidents/{incident_id}/escalation-details",
            {
                "account_holder_name": "Иванов Иван Иванович",
                "account_login": "ivanov",
                "computer_name": "WS-001",
                "subject": "Принтер не печатает",
                "additional_info": "Демо",
            },
        )
        incident = wait(
            incident_id, lambda value: value["escalation"] and value["escalation"]["status"] == "SENT_MOCK"
        )
        print(f"State: {incident['state']}; mail: {incident['escalation']['status']}")
        print(incident["escalation"]["mock_email"]["subject"])
        print(f"Inspect: GET /api/incidents/{incident_id}")
        print("PASS: no real email was sent")


if __name__ == "__main__":
    main()
