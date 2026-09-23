import unittest

from budget_tool_router import OpenMeteoWeatherAdapter


class AdapterTests(unittest.TestCase):
    def test_open_meteo_adapter_estimates_and_measures_real_call_shape(self):
        calls = []

        def transport(url, timeout, user_agent):
            calls.append((url, timeout, user_agent))
            return 200, {
                "latitude": 55.75,
                "longitude": 37.62,
                "current_units": {"temperature_2m": "°C"},
                "current": {"temperature_2m": 12.0},
            }

        adapter = OpenMeteoWeatherAdapter(timeout=3.0, transport=transport)
        arguments = {"latitude": 55.7558, "longitude": 37.6173}
        before = adapter.estimate(arguments)
        self.assertEqual(before.estimated_cost, 0.0)
        self.assertEqual(before.estimated_latency, 3.0)

        result = adapter.invoke(arguments)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.observed_cost, 0.0)
        self.assertIsNone(result.error_code)
        self.assertIn("latitude=55.7558", calls[0][0])

        after = adapter.estimate(arguments)
        self.assertLess(after.estimated_latency, 3.0)

    def test_invalid_coordinates_fail_before_network(self):
        adapter = OpenMeteoWeatherAdapter(transport=lambda *args: self.fail("network called"))
        with self.assertRaises(ValueError):
            adapter.estimate({"latitude": 100, "longitude": 0})


if __name__ == "__main__":
    unittest.main()
