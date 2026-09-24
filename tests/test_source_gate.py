import pathlib,sys,unittest
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/"training"))
from source_gate import admit
class T(unittest.TestCase):
 def test_gold(self):self.assertTrue(admit({"memory_class":"M4_GOLD","validated":True,"object_path":"x/M4/x"})[0])
 def test_m6(self):self.assertFalse(admit({"memory_class":"M6_COLD_BENCHMARK","validated":True,"object_path":"x/M6/x"})[0])
 def test_m6path(self):self.assertFalse(admit({"memory_class":"M4_GOLD","validated":True,"object_path":"x/M6/x"})[0])
if __name__=="__main__":unittest.main()
