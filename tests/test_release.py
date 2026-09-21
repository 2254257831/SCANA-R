"""Release-level evidence integrity and archive traversal tests."""
from pathlib import Path
import importlib.util, unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class ReleaseTests(unittest.TestCase):
    def test_archive_paths_cannot_escape(self):
        f=module('import_artifacts').checked_path
        for bad in ['../outside','/absolute','C:/absolute','x/../../outside','x\\..\\outside']:
            with self.assertRaises(ValueError):f(ROOT/'outputs',bad)

    def test_paper_summary_matches_episode_evidence(self):
        df=pd.read_csv(ROOT/'results/independent_test/per_episode.csv')
        self.assertEqual(len(df),3200)
        self.assertFalse(df.duplicated(['task','method','seed','layout']).any())
        summary=pd.read_csv(ROOT/'results/independent_test/summary.csv')
        for row in summary.itertuples():
            d=df[(df.task==row.task)&(df.method==row.method)]
            self.assertEqual(len(d),200);self.assertEqual(d.success.sum(),row.successes)
            self.assertAlmostEqual(d.success.mean(),row.mean,12)

if __name__=='__main__':unittest.main()
