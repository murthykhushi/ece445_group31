import unittest
import numpy as np
import pandas as pd
from evaluate_vision import evaluate, match_events, match_samples


class EvaluationTests(unittest.TestCase):
    def test_event_matching_is_one_to_one(self):
        self.assertEqual(len(match_events([1,1.02],[1.01])),1)
        self.assertEqual(len(match_events([1,2],[1.04,2.05])),2)
        self.assertEqual(match_events([1],[2]),[])

    def test_samples_linear_matching(self):
        pairs=match_samples([0,.1,.3],[0,.1,.2,.3],.01)
        self.assertEqual(pairs,[(0,0),(1,1),(2,3)])

    def test_unknown_predictions_report_coverage_not_perfect_accuracy(self):
        p=pd.DataFrame({'timestamp':[0.,1.], 'eye_state':['OPEN','UNKNOWN']})
        labels=pd.DataFrame({'timestamp':[0.,1.], 'eye_state':['OPEN','CLOSED']})
        result=evaluate(p,labels)
        self.assertEqual(result['eye_state_evaluated_pairs'],1)
        self.assertEqual(result['labeled_frames'],2)

    def test_blink_false_positive_counts(self):
        p=pd.DataFrame({'timestamp':[1.,2.], 'blink_event_duration_ms':[100.,120.]})
        labels=pd.DataFrame({'timestamp':[1.,2.], 'blink_event_duration_ms':[110.,np.nan]})
        result=evaluate(p,labels)
        self.assertEqual(result['blink_fp'],1)
        self.assertAlmostEqual(result['blink_duration_mae_ms'],10)
