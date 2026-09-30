import importlib.util
from pathlib import Path
import unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('current',ROOT/'scripts/reproduce_current.py')
current=importlib.util.module_from_spec(spec);spec.loader.exec_module(current)

class CurrentGridTests(unittest.TestCase):
    def setUp(self):
        self.dimensions={'bank':[1,2],'seed':[3,4]}
        self.data=pd.DataFrame({'bank':[1,1,2,2],'seed':[3,4,3,4],'success':[1,0,1,1]})
    def test_complete(self):current.validate_grid(self.data,self.dimensions)
    def test_missing(self):
        with self.assertRaises(ValueError):current.validate_grid(self.data.iloc[:-1],self.dimensions)
    def test_duplicate(self):
        with self.assertRaises(ValueError):current.validate_grid(pd.concat([self.data.iloc[:3],self.data.iloc[:1]]),self.dimensions)
    def test_nonbinary(self):
        self.data.loc[0,'success']=2
        with self.assertRaises(ValueError):current.validate_grid(self.data,self.dimensions)

if __name__=='__main__':unittest.main()
