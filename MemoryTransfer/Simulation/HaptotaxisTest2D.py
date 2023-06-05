from cc3d import CompuCellSetup

        
from HaptotaxisTest2DSteppables import PhiSolverSteppable
CompuCellSetup.register_steppable(steppable=PhiSolverSteppable(frequency=1))


from HaptotaxisTest2DSteppables import AssignMotilitySteppable
CompuCellSetup.register_steppable(steppable=AssignMotilitySteppable(frequency=1))

from HaptotaxisTest2DSteppables import FiberConcentrationCaseBRandom50
CompuCellSetup.register_steppable(steppable=FiberConcentrationCaseBRandom50(frequency=1))

from HaptotaxisTest2DSteppables import tracks_n_plots
CompuCellSetup.register_steppable(steppable=tracks_n_plots(frequency=100))

CompuCellSetup.run()
