"""Regression checks for the Gaussian centroid and the inference dispatch."""

import math
import unittest

import numpy as np
from simpful import FuzzySet, Sigmoid_MF

from benchmark_gaussian_centroid import build_system, output, reference, set_strengths
from fast_mamdani import fast_mamdani
from gaussian_centroid import gaussian_centroid


class GaussianCentroidTests(unittest.TestCase):
    def test_symmetric_clipped_bell(self):
        self.assertAlmostEqual(gaussian_centroid([(0.35, 8, 1.4)], 2, 14), 8, places=12)

    def test_overlap_crossings_against_independent_quadrature(self):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            for _ in range(12):
                terms = [(float(w), float(mu), float(sigma))
                         for w, mu, sigma in zip(rng.uniform(.05, 1, 4),
                                                  rng.uniform(-2, 27, 4), rng.uniform(.1, 5, 4))]
                expected = reference([(mu, sigma, w) for w, mu, sigma in terms])
                self.assertAlmostEqual(gaussian_centroid(terms, 0, 25), expected, places=8)

    def test_crossing_ablation_changes_asymmetric_overlap(self):
        terms = [(1.0, 9, 1.7), (.95, 11.4, 3)]
        expected = reference([(mu, sigma, w) for w, mu, sigma in terms])
        self.assertAlmostEqual(gaussian_centroid(terms, 0, 25), expected, places=9)
        self.assertGreater(abs(gaussian_centroid(terms, 0, 25, include_crossings=False)
                               - expected), .001)

    def test_asymmetric_narrow_case_through_simpful(self):
        bells, levels = ((7.01, .003), (18, .008)), (.85, .55)
        system = build_system(bells)
        set_strengths(system, levels)
        expected = reference([(mu, sigma, w) for (mu, sigma), w in zip(bells, levels)])
        self.assertAlmostEqual(output(system), expected, places=9)
        self.assertAlmostEqual(output(system, merge=False), expected, places=9)
        self.assertGreater(abs(output(system, analytic_gaussian=False) - expected), 1e-10)
        self.assertAlmostEqual(fast_mamdani(system, ["Tip"], exact=False)["Tip"],
                               float(system.Mamdani_inference(["Tip"])["Tip"]), places=10)

    def test_mixed_output_retains_grid(self):
        system = build_system(((7, 2), (18, 2)))
        set_strengths(system, (.85, .55))
        system._lvs["Tip"]._FSlist[1] = FuzzySet(function=Sigmoid_MF(18, 1), term="bell1")
        value = output(system)
        self.assertTrue(math.isfinite(value))
        self.assertEqual(value, output(system, analytic_gaussian=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
