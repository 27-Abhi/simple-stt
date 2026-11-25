import os
import tempfile
import unittest

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = ""
os.environ["SQLITE_PATH"] = os.path.join(tempfile.gettempdir(), "test_stt.db")

from app import app, init_db


class TestSTTApp(unittest.TestCase):
    def setUp(self):
        if os.path.exists(os.environ["SQLITE_PATH"]):
            os.remove(os.environ["SQLITE_PATH"])
        init_db()
        self.client = TestClient(app)

    def test_crud_and_health(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        # Create
        created = self.client.post(
            "/api/transcriptions",
            json={
                "transcript": "Namaste, aaj ka weather bahut अच्छा है",
                "language": "hi-IN,en-IN",
                "engine": "web-speech",
                "duration_seconds": 4.2,
            },
        )
        self.assertEqual(created.status_code, 201)
        item = created.json()
        self.assertGreaterEqual(item["id"], 1)
        self.assertIn("Namaste", item["title"])

        # Read / Search
        listed = self.client.get("/api/transcriptions", params={"q": "weather"})
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()), 1)

        # Update
        updated = self.client.patch(
            f"/api/transcriptions/{item['id']}",
            json={"title": "Hinglish Greeting", "transcript": "Namaste, aaj ka weather bahut accha hai"},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["title"], "Hinglish Greeting")

        # Delete
        deleted = self.client.delete(f"/api/transcriptions/{item['id']}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get(f"/api/transcriptions/{item['id']}").status_code, 404)


if __name__ == "__main__":
    unittest.main()
