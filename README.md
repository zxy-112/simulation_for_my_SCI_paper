# simulation_for_my_SCI_paper
run simulation.py and you can reproduce my simulation reuslts in my paper "Coherent Interference Suppression Algorithm Based on Duvall Structure with with Gain Control of Auxiliary Array"

change some lines in methods in simulation.py to use different algorithms in utils.py or to use different signals in signl.py. There seems to be other signal module in 
python so the name signl.py is used here.

the aray module inculdes a LineArray class which is useful for the generation of array signals. The Array class can be extended to acquire SurfaceArray or 3DArray to get
more complex array structure, but i haven't done it yet.
