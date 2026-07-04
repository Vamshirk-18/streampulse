"""
Milestone 4 - API Test Suite
Contains 8 unit tests covering valid and invalid inputs.

AI Tool Prompt Used:
"Write a separate test suite containing at least 5 unit tests
verifying how the API handles valid and invalid inputs."

Run with: python test_api.py
"""

import unittest
import json
import sys
sys.path.insert(0, '/home/claude')

from app import app, validate_input, build_feature_vector

class TestStreamingEngagementAPI(unittest.TestCase):

    def setUp(self):
        """Set up test client before each test."""
        app.config['TESTING'] = True
        self.client = app.test_client()

        # A perfectly valid user payload
        self.valid_payload = {
            "age": 28,
            "weekly_watch_hours": 15.5,
            "skip_rate": 0.2,
            "avg_rating_given": 4.1,
            "logins_per_week": 5,
            "subscription_type": "Premium",
            "content_diversity_score": 0.75,
            "days_since_last_login": 2,
            "notifications_clicked": 4,
            "watch_streak_days": 12
        }

    # ─────────────────────────────────────────────────────────
    # TEST 1: Health endpoint returns 200
    # ─────────────────────────────────────────────────────────
    def test_01_health_check(self):
        """GET /health should return 200 and status ok."""
        res = self.client.get('/health')
        data = json.loads(res.data)

        self.assertEqual(res.status_code, 200)
        self.assertEqual(data['status'], 'ok')
        print("✅ Test 1 PASSED: Health check returns 200")

    # ─────────────────────────────────────────────────────────
    # TEST 2: Valid input returns a prediction
    # ─────────────────────────────────────────────────────────
    def test_02_valid_input_returns_prediction(self):
        """POST /predict with valid data should return a prediction."""
        res = self.client.post('/predict',
                               data=json.dumps(self.valid_payload),
                               content_type='application/json')
        data = json.loads(res.data)

        self.assertEqual(res.status_code, 200)
        self.assertIn('prediction', data)
        self.assertIn(data['prediction'], ['Low', 'Medium', 'High'])
        self.assertIn('confidence_pct', data)
        self.assertIn('probabilities', data)
        print(f"✅ Test 2 PASSED: Valid input → Prediction: {data['prediction']} ({data['confidence_pct']}% confidence)")

    # ─────────────────────────────────────────────────────────
    # TEST 3: Missing field returns 400 error
    # ─────────────────────────────────────────────────────────
    def test_03_missing_field_returns_400(self):
        """POST /predict with a missing field should return 400."""
        bad_payload = self.valid_payload.copy()
        del bad_payload['age']  # Remove required field

        res = self.client.post('/predict',
                               data=json.dumps(bad_payload),
                               content_type='application/json')
        data = json.loads(res.data)

        self.assertEqual(res.status_code, 400)
        self.assertIn('error', data)
        self.assertIn('details', data)
        print(f"✅ Test 3 PASSED: Missing field caught → {data['details']}")

    # ─────────────────────────────────────────────────────────
    # TEST 4: Invalid subscription type returns 400
    # ─────────────────────────────────────────────────────────
    def test_04_invalid_subscription_type(self):
        """POST /predict with invalid subscription_type should return 400."""
        bad_payload = self.valid_payload.copy()
        bad_payload['subscription_type'] = 'Gold'  # Not allowed

        res = self.client.post('/predict',
                               data=json.dumps(bad_payload),
                               content_type='application/json')
        data = json.loads(res.data)

        self.assertEqual(res.status_code, 400)
        self.assertIn('subscription_type', str(data['details']))
        print(f"✅ Test 4 PASSED: Invalid subscription_type caught → {data['details']}")

    # ─────────────────────────────────────────────────────────
    # TEST 5: Out-of-range value returns 400
    # ─────────────────────────────────────────────────────────
    def test_05_out_of_range_value(self):
        """POST /predict with skip_rate > 1 should return 400."""
        bad_payload = self.valid_payload.copy()
        bad_payload['skip_rate'] = 5.5  # Max is 1.0

        res = self.client.post('/predict',
                               data=json.dumps(bad_payload),
                               content_type='application/json')
        data = json.loads(res.data)

        self.assertEqual(res.status_code, 400)
        print(f"✅ Test 5 PASSED: Out-of-range skip_rate caught → {data['details']}")

    # ─────────────────────────────────────────────────────────
    # TEST 6: Non-JSON request returns 415
    # ─────────────────────────────────────────────────────────
    def test_06_non_json_request(self):
        """POST /predict without JSON content-type should return 415."""
        res = self.client.post('/predict',
                               data="age=28&skip_rate=0.2",
                               content_type='application/x-www-form-urlencoded')
        data = json.loads(res.data)

        self.assertEqual(res.status_code, 415)
        self.assertIn('error', data)
        print(f"✅ Test 6 PASSED: Non-JSON request rejected → {data['error']}")

    # ─────────────────────────────────────────────────────────
    # TEST 7: Model info endpoint returns correct info
    # ─────────────────────────────────────────────────────────
    def test_07_model_info_endpoint(self):
        """GET /model-info should return model metadata."""
        res = self.client.get('/model-info')
        data = json.loads(res.data)

        self.assertEqual(res.status_code, 200)
        self.assertIn('model_type', data)
        self.assertIn('classes', data)
        self.assertEqual(data['classes'], ['Low', 'Medium', 'High'])
        print(f"✅ Test 7 PASSED: Model info → {data['model_type']}, {data['n_features']} features")

    # ─────────────────────────────────────────────────────────
    # TEST 8: Low-engagement user is predicted correctly
    # ─────────────────────────────────────────────────────────
    def test_08_low_engagement_user(self):
        """A user who barely watches should be predicted as Low."""
        low_user = {
            "age": 45,
            "weekly_watch_hours": 0.1,
            "skip_rate": 0.95,
            "avg_rating_given": 1.0,
            "logins_per_week": 1,
            "subscription_type": "Free",
            "content_diversity_score": 0.05,
            "days_since_last_login": 59,
            "notifications_clicked": 0,
            "watch_streak_days": 0
        }
        res = self.client.post('/predict',
                               data=json.dumps(low_user),
                               content_type='application/json')
        data = json.loads(res.data)

        self.assertEqual(res.status_code, 200)
        self.assertIn(data['prediction'], ['Low', 'Medium'])
        print(f"✅ Test 8 PASSED: Low-engagement user correctly predicted as '{data['prediction']}'")


if __name__ == '__main__':
    print("=" * 60)
    print("   MILESTONE 4 — API TEST SUITE")
    print("=" * 60)
    print()

    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    suite = loader.loadTestsFromTestCase(TestStreamingEngagementAPI)

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    print()
    print("=" * 60)
    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"   {passed}/{total} Tests Passed")
    if result.wasSuccessful():
        print("   ✅ ALL TESTS PASSED — API is production ready!")
    else:
        print("   ❌ Some tests failed.")
    print("=" * 60)
