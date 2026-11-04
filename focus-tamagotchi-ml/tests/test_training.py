import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from vision.features import FEATURES, SCHEMA_ID, load_focus_model, predict_focus
from train_real import train
from calibrate_vision import fit_eyes


class TrainingTests(unittest.TestCase):
    def test_session_split_and_artifact_round_trip(self):
        rows=[]
        for session in range(4):
            for i in range(5):
                row={feature:0.0 for feature in FEATURES}
                row.update(session_id=f'session-{session}',schema_id=SCHEMA_ID,landmark_schema='ibug68',
                           ear_left=.3,ear_right=.3,closure_fraction_60s=.1,blink_count_60s=5,
                           median_blink_ms=np.nan,yaw_deg=np.nan,pitch_deg=np.nan,roll_deg=np.nan,
                           focus_score=40+session*5+i)
                rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            data=Path(directory)/'data.csv'; output=Path(directory)/'model.joblib'
            pd.DataFrame(rows).to_csv(data,index=False)
            bundle=train(data,output)
            report=bundle['report']
            self.assertFalse(set(report['train_sessions']) & set(report['test_sessions']))
            loaded=load_focus_model(output)
            self.assertTrue(np.isfinite(predict_focus(loaded,{key:rows[0][key] for key in FEATURES})))

    def test_eye_calibration_and_overlap_rejection(self):
        data=pd.DataFrame({'eye_state':['OPEN']*20+['CLOSED']*20,
                           'ear_left':[.3]*20+[.1]*20,'ear_right':[.32]*20+[.12]*20})
        result=fit_eyes(data)
        self.assertLess(result['close_left'],result['open_left'])
        self.assertTrue(result['calibrated_eyes'])
        data['ear_left']=.2
        with self.assertRaises(ValueError):
            fit_eyes(data)
