import unittest
from config.config import DATA_DIR
from problems.vrptw import Customer, VRPTW


class TestVRPTW(unittest.TestCase):
    def setUp(self):
        self.p = VRPTW([Customer(0,0,0,0,0,100,0),
                        Customer(1,3,0,4,0,20,0),
                        Customer(2,6,0,4,0,20,0)], 8, 2)

    def test_valid_and_distance(self):
        result = self.p.evaluate([[1, 2]])
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.distance, 12)
        self.assertEqual(result.details[0].start_times, [3, 6])

    def test_waiting_and_boundary(self):
        p = VRPTW([Customer(0,0,0,0,0,12,0), Customer(1,3,0,8,10,10,0)], 8, 1)
        r = p.evaluate([[1]])
        self.assertFalse(r.feasible)  # 等待后返回时间 13，超过 12
        p.customers[0] = Customer(0,0,0,0,0,13,0)
        self.assertTrue(p.evaluate([[1]]).feasible)
        self.assertEqual(p.evaluate([[1]]).details[0].start_times, [10])

    def test_capacity(self):
        self.p.capacity = 7
        self.assertFalse(self.p.check_route([1, 2]).feasible)
        self.assertTrue(self.p.check_route([1]).feasible)

    def test_time_window(self):
        self.p.customers[2] = Customer(2,6,0,4,0,5,0)
        self.assertFalse(self.p.check_route([1, 2]).feasible)

    def test_customer_coverage(self):
        for routes in ([[1]], [[1,1,2]], [[1,3]], [[1], []]):
            self.assertFalse(self.p.evaluate(routes).feasible)
        self.assertTrue(self.p.evaluate([[1], [2]]).feasible)

    def test_vehicle_limit(self):
        self.p.vehicle_limit = 1
        self.assertFalse(self.p.evaluate([[1], [2]]).feasible)

    def test_decode_deterministic(self):
        self.p.capacity = 4
        a, b = self.p.decode([1,2]), self.p.decode([1,2])
        self.assertEqual(a.routes, [[1],[2]])
        self.assertEqual(a, b)

    def test_solomon(self):
        p = VRPTW.from_solomon(DATA_DIR / "C101.txt", 10)
        self.assertEqual(len(p.customers), 11)
        self.assertEqual(p.capacity, 200)
        self.assertEqual(p.vehicle_limit, 25)
        self.assertTrue(p.evaluate([[5,3,7,8,10,9,6,4,2,1]]).feasible)


if __name__ == "__main__":
    unittest.main()
